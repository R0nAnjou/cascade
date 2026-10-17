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



from vllm_ascend.ascend_config import RejectionSamplerConfig

_config_patch_applied = False


def _validate_cascade(self):
    if not isinstance(self.enable_cascade_verify, bool):
        raise ValueError(
            f"rejection_sampler_config.enable_cascade_verify must be a bool, "
            f"got {type(self.enable_cascade_verify).__name__}"
        )
    if not isinstance(self.cascade_threshold, (int, float)):
        raise ValueError(
            f"rejection_sampler_config.cascade_threshold must be a float, "
            f"got {type(self.cascade_threshold).__name__}"
        )
    if not (0 < self.cascade_threshold <= 1):
        raise ValueError(
            f"rejection_sampler_config.cascade_threshold must be in (0, 1], got {self.cascade_threshold}"
        )


def apply_config_patch():
    """Idempotently wrap ``RejectionSamplerConfig.__init__`` to add the cascade
    verify fields (``enable_cascade_verify`` / ``cascade_threshold``)."""
    global _config_patch_applied
    if _config_patch_applied:
        return
    _config_patch_applied = True

    orig_init = RejectionSamplerConfig.__init__

    def patched_init(self, config=None):
        orig_init(self, config)
        if config is None:
            config = {}
        self.enable_cascade_verify = config.get("enable_cascade_verify", False)
        self.cascade_threshold = config.get("cascade_threshold", 0.15)
        _validate_cascade(self)

    RejectionSamplerConfig.__init__ = patched_init
