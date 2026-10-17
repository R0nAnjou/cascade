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

from contextlib import contextmanager

import vllm_ascend.sample.rejection_sampler as rejection_sampler_module


@contextmanager
def bind_rejection_sample(fn):
    """Temporarily swap the parent module's module-level ``rejection_sample``
    with ``fn`` for the duration of the ``with`` block.

    The parent ``AscendRejectionSampler.forward`` looks up ``rejection_sample``
    at call time, so this makes it dispatch to the cascade variant without any
    change to the vllm-ascend source.
    """
    original = rejection_sampler_module.rejection_sample
    rejection_sampler_module.rejection_sample = fn
    try:
        yield
    finally:
        rejection_sampler_module.rejection_sample = original
