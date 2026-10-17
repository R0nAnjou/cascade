#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-cascade-patch plugin.
#

import torch

from vllm.v1.sample.rejection_sampler import PLACEHOLDER_TOKEN_ID


def rejection_greedy_sample_pytorch_310p(
    output_token_ids,  # [batch_size, max_spec_len + 1]
    cu_num_draft_tokens,  # [batch_size]
    draft_token_ids,  # [num_tokens]
    target_argmax,  # [num_tokens]
    bonus_token_ids,  # [batch_size]
    draft_tokens_per_req,  # [batch_size], list
    max_spec_len,
    is_greedy=None,  # [batch_size] or None
    target_probs=None,  # [num_tokens, vocab_size] or None (cascade verify)
    cascade_threshold=0.15,
):
    batch_size = output_token_ids.size(0)
    num_tokens = draft_token_ids.size(0)
    device = output_token_ids.device
    draft_tokens_per_req = torch.tensor(draft_tokens_per_req).to(device, non_blocking=True)
    if is_greedy is None:
        is_greedy = torch.ones(batch_size, dtype=torch.bool, device=device)

    start_indices = cu_num_draft_tokens - draft_tokens_per_req
    req_ids = torch.arange(batch_size, device=device)
    token_req_ids = torch.repeat_interleave(req_ids, draft_tokens_per_req)
    token_positions = torch.arange(num_tokens, device=device) - start_indices[token_req_ids]

    # Find the first mismatch position of each request.
    if target_probs is not None:
        # Cascade verify: a draft token is rejected when its
        # target probability is below cascade_threshold * the argmax probability.
        valid_draft_mask = draft_token_ids != PLACEHOLDER_TOKEN_ID
        safe_draft_ids = draft_token_ids.masked_fill(~valid_draft_mask, 0)
        draft_target_probs = target_probs.gather(-1, safe_draft_ids.unsqueeze(-1)).squeeze(-1)
        draft_target_probs = torch.where(valid_draft_mask, draft_target_probs, torch.zeros_like(draft_target_probs))
        argmax_target_probs = target_probs.gather(-1, target_argmax.unsqueeze(-1)).squeeze(-1)
        mismatch_global = draft_target_probs / argmax_target_probs.clamp(min=1e-10) < cascade_threshold
    else:
        mismatch_global = draft_token_ids != target_argmax

    if max_spec_len == 0:
        first_mismatch_pos_per_req = torch.zeros(batch_size, dtype=torch.long, device=device)
    else:
        pos_matrix = torch.full((batch_size, max_spec_len), -1, dtype=torch.long, device=device)
        pos_matrix[token_req_ids, token_positions] = token_positions
        mismatch_matrix = torch.full((batch_size, max_spec_len), False, dtype=torch.bool, device=device)
        mismatch_matrix[token_req_ids, token_positions] = mismatch_global
        mismatch_positions = torch.where(mismatch_matrix, pos_matrix, max_spec_len * 2)
        first_mismatch_pos_per_req, _ = torch.min(mismatch_positions, dim=1)
        no_mismatch_mask = first_mismatch_pos_per_req == max_spec_len * 2
        first_mismatch_pos_per_req[no_mismatch_mask] = draft_tokens_per_req[no_mismatch_mask]

    if target_probs is not None:
        # Cascade verify: keep the accepted draft prefix, correct the first
        # reject to the argmax, and append a bonus token when all accepted.
        accept_len = torch.minimum(first_mismatch_pos_per_req, draft_tokens_per_req)
        copy_indices = torch.arange(max_spec_len + 1, device=device).expand(batch_size, -1)
        accept_mask = copy_indices < accept_len.unsqueeze(1)
        greedy_mask = is_greedy.unsqueeze(1)
        final_accept_mask = accept_mask & greedy_mask
        global_idx = start_indices.unsqueeze(1) + copy_indices
        output_token_ids[final_accept_mask] = draft_token_ids[global_idx[final_accept_mask]].to(output_token_ids.dtype)

        # Correct the first reject position to the argmax.
        reject_mask = (
            (copy_indices == first_mismatch_pos_per_req.unsqueeze(1))
            & ((first_mismatch_pos_per_req < draft_tokens_per_req).unsqueeze(1))
            & greedy_mask
        )
        output_token_ids[reject_mask] = target_argmax[global_idx[reject_mask]].to(output_token_ids.dtype)
        # Fill bonus token when all draft tokens were accepted.
        needs_bonus = is_greedy & (first_mismatch_pos_per_req >= draft_tokens_per_req)
        if torch.any(needs_bonus):
            bonus_rows = torch.where(needs_bonus)[0]
            bonus_cols = draft_tokens_per_req[bonus_rows]
            bonus_token_ids = bonus_token_ids.squeeze(1)
            output_token_ids[bonus_rows, bonus_cols] = bonus_token_ids[bonus_rows]
        return

    # Copy matched target tokens into output.
    copy_len = torch.minimum(first_mismatch_pos_per_req + 1, draft_tokens_per_req)
    copy_indices = torch.arange(max_spec_len + 1, device=device).expand(batch_size, -1)
    copy_mask = copy_indices < copy_len.unsqueeze(1)
    greedy_mask = is_greedy.unsqueeze(1)
    final_copy_mask = copy_mask & greedy_mask
    global_idx = start_indices.unsqueeze(1) + copy_indices
    output_token_ids[final_copy_mask] = target_argmax[global_idx[final_copy_mask]].to(output_token_ids.dtype)
    # Fill bonus token.
    needs_bonus = is_greedy & (first_mismatch_pos_per_req >= draft_tokens_per_req)
    if torch.any(needs_bonus):
        bonus_rows = torch.where(needs_bonus)[0]
        bonus_cols = draft_tokens_per_req[bonus_rows]
        bonus_token_ids = bonus_token_ids.squeeze(1)
        output_token_ids[bonus_rows, bonus_cols] = bonus_token_ids[bonus_rows]
