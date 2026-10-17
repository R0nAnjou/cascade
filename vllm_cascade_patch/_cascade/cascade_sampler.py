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
from vllm.logger import logger
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.rejection_sampler import (
    GREEDY_TEMPERATURE,
    PLACEHOLDER_TOKEN_ID,
    generate_uniform_probs,
)

import vllm_ascend.sample.rejection_sampler as rejection_sampler_module
from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.sample.rejection_sampler import (
    greedy_sample,
    rejection_greedy_sample_spec_len_1_pytorch,
    rejection_random_sample_block_verify_pytorch,
)

from vllm_cascade_patch._cascade.greedy import rejection_greedy_sample_pytorch_310p
from vllm_cascade_patch._cascade.random import rejection_random_sample_pytorch_310p


def _unpack_and_validate_inputs(
    draft_token_ids: torch.Tensor,
    cu_num_draft_tokens: torch.Tensor,
    draft_probs: torch.Tensor | None,
    # [num_tokens, vocab_size//tp_size] or tuple of (logits, indices)
    # For greedy: Tensor [num_tokens, vocab_size//tp_size]
    # For random: tuple of (logits [num_tokens, top_k*tp_size], indices [num_tokens, top_k*tp_size])
    target_logits_or_tuple: torch.Tensor | tuple[torch.Tensor, torch.Tensor | None],
    bonus_token_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor | None, int]:
    # Unpack target_logits_or_tuple
    if isinstance(target_logits_or_tuple, tuple):
        target_logits, target_indices = target_logits_or_tuple
    else:
        target_logits = target_logits_or_tuple
        target_indices = None

    if draft_token_ids.ndim != 1:
        raise ValueError(
            f"draft_token_ids 必须是一维张量"
        )
    if draft_probs is not None and draft_probs.ndim != 2:
        raise ValueError(
            f"draft_probs 必须为 None 或二维张量"
        )
    if cu_num_draft_tokens.ndim != 1:
        raise ValueError(
            f"cu_num_draft_tokens.ndim 必须是一维张量"
        )
    if target_logits.ndim != 2:
        raise ValueError(
            f"target_logits 必须是二维张量"
        )
    if not draft_token_ids.is_contiguous():
        raise ValueError(
            f"draft_token_ids 必须是连续张量"
        )
    if draft_probs is not None and not draft_probs.is_contiguous():
        raise ValueError(
            f"draft_probs 必须为 None 或连续张量"
        )
    if not target_logits.is_contiguous():
        raise ValueError(
            f"target_logits 必须是连续张量"
        )
    if not bonus_token_ids.is_contiguous():
        raise ValueError(
            f"bonus_token_ids 必须是连续张量"
        )

    num_tokens = draft_token_ids.shape[0]
    if target_logits.shape[0] != num_tokens:
        raise ValueError(
            f"target_logits.shape[0] 必须等于 num_tokens"
        )

    return target_logits, target_indices, num_tokens


def rejection_sample_cascade(
    draft_token_ids: torch.Tensor,
    num_draft_tokens: list[int],
    max_spec_len: int,
    cu_num_draft_tokens: torch.Tensor,
    draft_probs: torch.Tensor | None,
    # [num_tokens, vocab_size//tp_size] or tuple of (logits, indices)
    # For greedy: Tensor [num_tokens, vocab_size//tp_size]
    # For random: tuple of (logits [num_tokens, top_k*tp_size], indices [num_tokens, top_k*tp_size])
    target_logits_or_tuple: torch.Tensor | tuple[torch.Tensor, torch.Tensor | None],
    bonus_token_ids: torch.Tensor,
    sampling_metadata: SamplingMetadata,
    synthetic_mode: bool = False,
    synthetic_conditional_rates: torch.Tensor | None = None,
    ori_target_logits: torch.Tensor | None = None,
) -> torch.Tensor:
    
    # Unpack and validate inputs.
    target_logits, target_indices, num_tokens = _unpack_and_validate_inputs(
        draft_token_ids=draft_token_ids,
        cu_num_draft_tokens=cu_num_draft_tokens,
        draft_probs=draft_probs,
        target_logits_or_tuple=target_logits_or_tuple,
        bonus_token_ids=bonus_token_ids,
    )

    batch_size = len(num_draft_tokens)
    device = target_logits.device

    # Block verify requires enable_block_verify config and max_spec_len >= 3.
    using_block_verify = max_spec_len >= 3 and bool(get_ascend_config().rejection_sampler_config.enable_block_verify)
    using_entropy_verify = bool(get_ascend_config().rejection_sampler_config.enable_entropy_verify)
    posterior_threshold = float(get_ascend_config().rejection_sampler_config.posterior_threshold)
    posterior_alpha = float(get_ascend_config().rejection_sampler_config.posterior_alpha)
    logger.debug_once(
        "[sample/rejection_sampler] Rejection sampling path: "
        "block_verify=%s, entropy_verify=%s, all_greedy=%s, all_random=%s, "
        "reduce_sample=%s",
        using_block_verify,
        using_entropy_verify,
        sampling_metadata.all_greedy,
        sampling_metadata.all_random,
        get_ascend_config().enable_reduce_sample,
    )

    # Cascade verify config and mutual exclusion. Cascade verify
    # is only valid on the full-vocab PyTorch path and is mutually exclusive
    # with block/entropy verify.
    cascade_config = get_ascend_config().rejection_sampler_config
    enable_cascade_verify = bool(cascade_config.enable_cascade_verify)
    cascade_threshold = float(cascade_config.cascade_threshold)
    if enable_cascade_verify and (using_block_verify or using_entropy_verify):
        logger.warning_once(
            "[sample/rejection_sampler_310p] Cascade verify is mutually exclusive with "
            "block/entropy verify; disabling cascade verify.",
        )
        enable_cascade_verify = False
    if enable_cascade_verify and target_indices is not None:
        logger.warning_once(
            "[sample/rejection_sampler_310p] Cascade verify requires the full-vocab "
            "PyTorch path; disabling cascade verify (reduce_sample=%s).",
            target_indices is not None,
        )
        enable_cascade_verify = False

    cascade_target_probs = None
    cascade_target_argmax = None
    if enable_cascade_verify and ori_target_logits is not None:
        # Use the raw full-vocab target logits so the cascade ratio is
        # independent of TP partitioning and sampling constraints.
        cascade_target_probs = ori_target_logits.softmax(dim=-1, dtype=torch.float32)
        cascade_target_argmax = ori_target_logits.argmax(dim=-1).view(-1)
        logger.debug_once(
            "[sample/rejection_sampler_310p] Cascade verify enabled: threshold=%s",
            cascade_threshold,
        )

    if using_entropy_verify and ori_target_logits is not None:
        ori_target_probs = ori_target_logits.softmax(dim=-1, dtype=torch.float32)
    else:
        ori_target_probs = None

    # Create output buffer.
    output_token_ids = torch.empty(
        (batch_size, max_spec_len + 1),
        dtype=torch.int32,  # Consistent with SamplerOutput.sampled_token_ids.
        device=device,
    )
    output_token_ids.fill_(PLACEHOLDER_TOKEN_ID)

    if sampling_metadata.all_greedy:
        is_greedy = None
    else:
        is_greedy = sampling_metadata.temperature == GREEDY_TEMPERATURE

    if using_block_verify or using_entropy_verify:
        logger.info_once(
            "RejectionSampler config: block_verify=%s, entropy_verify=%s, "
            "posterior_threshold=%s, posterior_alpha=%s, reduce_sample=%s, "
            "all_greedy=%s, all_random=%s",
            using_block_verify,
            using_entropy_verify,
            posterior_threshold,
            posterior_alpha,
            target_indices is not None,
            sampling_metadata.all_greedy,
            sampling_metadata.all_random,
        )

    # For greedy sampling, we need to do allgather first to get global argmax
    if not sampling_metadata.all_random:
        if get_ascend_config().enable_reduce_sample:
            target_argmax = greedy_sample(target_logits)
        else:
            target_argmax = target_logits.argmax(dim=-1).view(-1)

        use_cascade = cascade_target_probs is not None
        is_valid = (
                not use_cascade
                and min(num_draft_tokens) == 1
                and max(num_draft_tokens) == 1
                and sampling_metadata.all_greedy
        )
        sample_kwargs = {
            "output_token_ids": output_token_ids,
            "draft_token_ids": draft_token_ids,
            "target_argmax": target_argmax,
            "bonus_token_ids": bonus_token_ids,
        }
        if is_valid:
            sample_fn = rejection_greedy_sample_spec_len_1_pytorch
        else:
            sample_fn = rejection_greedy_sample_pytorch_310p
            sample_kwargs.update(
                cu_num_draft_tokens=cu_num_draft_tokens,
                target_argmax=cascade_target_argmax if use_cascade else target_argmax,
                draft_tokens_per_req=num_draft_tokens,
                max_spec_len=max_spec_len,
                is_greedy=is_greedy,
                target_probs=cascade_target_probs,
                cascade_threshold=cascade_threshold,
            )
        sample_fn(**sample_kwargs)
        if sampling_metadata.all_greedy:
            return output_token_ids

    # For random sampling with selected logits
    # target_logits is [num_tokens, top_k*tp_size] with indices [num_tokens, top_k*tp_size]
    vocab_size = target_logits.shape[-1]
    global_vocab_size = draft_probs.shape[-1] if draft_probs is not None else vocab_size

    # Compute probability distribution from target logits
    target_probs = target_logits.softmax(dim=-1, dtype=torch.float32)
    if not target_probs.is_contiguous():
        raise ValueError(
            f"target_probs 必须是连续张量"
        )

    # Generate uniform probabilities for rejection sampling
    uniform_probs = generate_uniform_probs(
        num_tokens,
        num_draft_tokens,
        sampling_metadata.generators,
        device,
    )

    # Sample recovered tokens for each position
    recovered_token_ids = rejection_sampler_module.sample_recovered_tokens(
        max_spec_len,
        num_draft_tokens,
        cu_num_draft_tokens,
        draft_token_ids,
        draft_probs,
        target_probs,
        sampling_metadata,
        device,
        target_indices=target_indices,
        global_vocab_size=global_vocab_size,
        enable_reduce_sampling=target_indices is not None,
    )

    if target_indices is None:
        # Fallback to original mode.
        # This path should not be used in the new distributed flow.
        logger.warning_once(
            "[sample/rejection_sampler] Using fallback (non-reduce-sample) path in "
            "rejection_sample. This path should not be used in the new distributed flow. "
            "enable_reduce_sample=%s, has_target_indices=%s",
            get_ascend_config().enable_reduce_sample,
            target_indices is not None,
        )

    random_sample_kwargs = {
        "output_token_ids": output_token_ids,
        "cu_num_draft_tokens": cu_num_draft_tokens,
        "draft_token_ids": draft_token_ids,
        "draft_probs": draft_probs,
        "target_probs": target_probs,
        "bonus_token_ids": bonus_token_ids,
        "recovered_token_ids": recovered_token_ids,
        "uniform_probs": uniform_probs,
        "is_greedy": is_greedy,
        "max_spec_len": max_spec_len,
        "vocab_size": vocab_size,
        "IS_NGRAM": draft_probs is None,
        "target_indices": target_indices,
        "enable_reduce_sampling": target_indices is not None,
        "ENTROPY_VERIFY": using_entropy_verify,
        "POSTERIOR_THRESHOLD": posterior_threshold,
        "POSTERIOR_ALPHA": posterior_alpha,
        "EPSILON": 1e-10,
        "ori_target_probs": ori_target_probs,
    }
    random_sample_fn = (
        rejection_random_sample_block_verify_pytorch
        if using_block_verify
        else rejection_random_sample_pytorch_310p
    )
    if not using_block_verify and target_indices is None:
        # Cascade verify (full-vocab fallback path only).
        random_sample_kwargs.update(
            target_argmax=cascade_target_argmax,
            cascade_threshold=cascade_threshold,
        )

    if not using_block_verify:
        # The local PyTorch implementation uses lowercase keyword names; the
        # vllm-ascend block verify implementation uses uppercase names.
        for upstream_name, local_name in (
            ("IS_NGRAM", "is_ngram"),
            ("ENTROPY_VERIFY", "entropy_verify"),
            ("POSTERIOR_THRESHOLD", "posterior_threshold"),
            ("POSTERIOR_ALPHA", "posterior_alpha"),
            ("EPSILON", "epsilon"),
        ):
            random_sample_kwargs[local_name] = random_sample_kwargs.pop(upstream_name)

    random_sample_fn(**random_sample_kwargs)

    return output_token_ids
