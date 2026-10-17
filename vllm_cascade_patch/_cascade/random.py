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


def rejection_random_sample_pytorch_310p(
    output_token_ids,  # [batch_size, max_spec_len + 1]
    cu_num_draft_tokens,  # [batch_size]
    draft_token_ids,  # [num_tokens]
    draft_probs,  # [num_tokens, vocab_size] or None
    target_probs,  # [num_tokens, vocab_size] or [num_tokens, selected_vocab_size]
    bonus_token_ids,  # [batch_size]
    recovered_token_ids,  # [num_tokens]
    uniform_probs,  # [num_tokens]
    is_greedy,  # [batch_size]
    max_spec_len,
    vocab_size,
    is_ngram=False,
    target_indices=None,  # [num_tokens, selected_vocab_size] global vocab indices
    enable_reduce_sampling=False,
    entropy_verify=False,
    posterior_threshold=0.95,
    posterior_alpha=0.4,
    epsilon=1e-10,
    ori_target_probs=None,
    target_argmax=None,  # [num_tokens] or None (cascade verify)
    cascade_threshold=0.15,
):
    """
    This function implements the Speculative Decoding rejection sampling step.
    Instead of looping through each request and each token (which causes high
    overhead), it uses a fully vectorized approach:

    1.  **Index Mapping**: Converts the flattened 1D token arrays into a 2D
        [batch_size, max_draft_len] grid using 'cu_num_draft_tokens' to handle
        variable-length sequences in the batch.
    2.  **Parallel Validation**: Calculates the acceptance condition
        (target_prob / draft_prob >= uniform_sample) for ALL draft tokens
        simultaneously across the entire batch.
    3.  **Short-circuit Simulation**: In the loop version, once a token is rejected,
        subsequent tokens are ignored. Here, we simulate this by finding the
        'first_reject_pos' using argmax on the rejection mask and creating a
        'should_skip' mask for all indices after the first failure.
    4.  **Token Selection**: Uses 'torch.where' to select:
        - Draft tokens (if accepted)
        - Recovered tokens (at the point of first rejection)
        - Bonus tokens (if all tokens in a sequence were accepted)
    5.  **Masking**: Ensures operations only apply to non-greedy requests and
        within valid sequence lengths.

    When `target_argmax` is provided, the acceptance condition is
    replaced by the cascade verify rule: accept the draft token when
    target_prob[draft] / target_prob[argmax] >= cascade_threshold.
    """

    batch_size = output_token_ids.shape[0]
    device = output_token_ids.device

    zero_cpu = torch.tensor([0], pin_memory=True)
    zero_device = zero_cpu.to(device, non_blocking=True)

    cu_start = torch.cat([zero_device, cu_num_draft_tokens[:-1]])
    cu_end = cu_num_draft_tokens
    num_draft_per_batch = cu_end - cu_start

    max_draft_len = max_spec_len
    pos_indices_cpu = torch.arange(max_draft_len, pin_memory=True)
    pos_indices = pos_indices_cpu.to(device, non_blocking=True)[None, :]

    valid_mask = pos_indices < num_draft_per_batch[:, None]
    global_token_indices = cu_start[:, None] + pos_indices
    global_token_indices = global_token_indices.clamp(0, draft_token_ids.shape[0] - 1)
    draft_tokens = draft_token_ids[global_token_indices]  # [batch_size, max_draft_len]
    placeholder_mask = draft_tokens == PLACEHOLDER_TOKEN_ID
    safe_draft_tokens = draft_tokens.masked_fill(placeholder_mask, 0)

    if is_ngram:
        ones_cpu = torch.ones(1, pin_memory=True, dtype=torch.float32)
        draft_token_probs = ones_cpu.to(device, non_blocking=True).expand_as(draft_tokens)
    else:
        flat_indices = global_token_indices.flatten()
        flat_draft_tokens = safe_draft_tokens.flatten()
        flat_draft_probs = draft_probs[flat_indices, flat_draft_tokens]
        draft_token_probs = flat_draft_probs.view(batch_size, max_draft_len)

    # Get target token probs
    if enable_reduce_sampling:
        # When enable_reduce_sampling, need to search for draft token in candidates
        flat_global_indices = global_token_indices.flatten()
        flat_draft_tokens = draft_tokens.flatten()

        flat_target_indices = target_indices[flat_global_indices]
        flat_target_probs = target_probs[flat_global_indices]

        # Check if draft token is in candidates
        draft_expanded = flat_draft_tokens.unsqueeze(1)
        is_in_candidates = flat_target_indices == draft_expanded

        # Get the probability of draft token from target (if present)
        target_token_probs_flat = torch.where(
            is_in_candidates, flat_target_probs, torch.tensor(0.0, device=device)
        ).sum(dim=1)

        target_token_probs = target_token_probs_flat.view(batch_size, max_draft_len)
    else:
        flat_indices = global_token_indices.flatten()
        flat_draft_tokens = safe_draft_tokens.flatten()
        flat_target_probs = target_probs[flat_indices, flat_draft_tokens]
        target_token_probs = flat_target_probs.view(batch_size, max_draft_len)

    uniform_token_probs = uniform_probs[global_token_indices]
    recovered_tokens = recovered_token_ids[global_token_indices]

    zero_threshold_cpu = torch.tensor([0.0], pin_memory=True, dtype=torch.float32)
    zero_threshold = zero_threshold_cpu.to(device, non_blocking=True)

    if target_argmax is not None:
        # Cascade verify: accept the draft token when its target
        # probability is at least cascade_threshold * the argmax probability.
        argmax_token_probs = target_probs[global_token_indices, target_argmax[global_token_indices]]
        acceptance_condition = target_token_probs / argmax_token_probs.clamp(min=epsilon) >= cascade_threshold
    elif entropy_verify:
        entropy_probs = ori_target_probs if ori_target_probs is not None else target_probs
        all_target_dist = entropy_probs[global_token_indices]
        entropy = -(all_target_dist * torch.log(all_target_dist + epsilon)).sum(dim=-1)
        exp_neg_entropy = torch.exp(-entropy * posterior_alpha)
        posterior_threshold_device = torch.tensor(posterior_threshold, device=device, dtype=torch.float32)
        threshold = torch.minimum(exp_neg_entropy, posterior_threshold_device)
        modified_uniform_token_probs = threshold * uniform_token_probs
        acceptance_condition = (draft_token_probs > zero_threshold) & (
            target_token_probs / draft_token_probs >= modified_uniform_token_probs
        )
    else:
        acceptance_condition = (draft_token_probs > zero_threshold) & (
            target_token_probs / draft_token_probs >= uniform_token_probs
        )
    acceptance_condition = acceptance_condition & (~placeholder_mask)

    first_rejection = (~acceptance_condition) & valid_mask

    default_pos_cpu = torch.full([batch_size, 1], max_draft_len, pin_memory=True)
    default_pos = default_pos_cpu.to(device, non_blocking=True)

    first_reject_pos = torch.where(
        first_rejection.any(dim=1, keepdim=True), first_rejection.float().argmax(dim=1, keepdim=True), default_pos
    )
    pos_mask = pos_indices >= first_reject_pos
    should_skip = pos_mask & valid_mask

    final_acceptance = acceptance_condition & (~should_skip)
    non_greedy_mask = ~is_greedy
    update_mask = non_greedy_mask[:, None] & valid_mask & (~should_skip)

    first_reject_mask = (pos_indices == first_reject_pos) & valid_mask & non_greedy_mask[:, None]
    final_update_mask = update_mask | first_reject_mask
    final_tokens = torch.where(
        first_reject_mask,
        recovered_tokens,
        torch.where(final_acceptance, draft_tokens, output_token_ids[:, :max_draft_len]),
    )

    output_token_ids[:, :max_draft_len] = torch.where(
        final_update_mask, final_tokens, output_token_ids[:, :max_draft_len]
    )

    no_rejection = first_reject_pos.squeeze(1) >= num_draft_per_batch
    should_add_bonus = non_greedy_mask & no_rejection

    bonus_positions = num_draft_per_batch  # [batch_size]

    seq_len = output_token_ids.shape[1]
    all_positions_cpu = torch.arange(seq_len, pin_memory=True)
    all_positions = all_positions_cpu.to(device, non_blocking=True)[None, :]  # [1, seq_len]

    batch_bonus_positions = bonus_positions[:, None]  # [batch_size, 1]

    max_spec_len_cpu = torch.tensor([max_spec_len], pin_memory=True)
    max_spec_len_device = max_spec_len_cpu.to(device, non_blocking=True)

    valid_bonus_pos = bonus_positions < (max_spec_len_device + 1)
    final_bonus_mask = should_add_bonus & valid_bonus_pos

    bonus_pos_match = all_positions == batch_bonus_positions
    bonus_pos_mask = bonus_pos_match & final_bonus_mask[:, None]

    bonus_values_expanded = bonus_token_ids.view(-1, 1).expand(-1, seq_len)
    output_token_ids[:] = torch.where(bonus_pos_mask, bonus_values_expanded, output_token_ids)
