# Copyright 2026 the LlamaFactory team.
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

import inspect
from importlib.metadata import PackageNotFoundError, version

import pytest


def _require_frozen_kt_stack() -> None:
    try:
        transformers_version = version("transformers-kt")
        accelerate_version = version("accelerate-kt")
    except PackageNotFoundError:
        pytest.skip("frozen KT distributions are not installed")

    assert transformers_version == "5.6.0.post1"
    assert accelerate_version == "1.14.0.post1"


def test_frozen_transformers_accelerate_kt_abi():
    _require_frozen_kt_stack()

    from accelerate import Accelerator
    from accelerate.utils import KTransformersPlugin
    from kt_kernel.sft import KTConfig
    from transformers.integrations.kt import HfTrainerKTConfig

    raw_config = {"kt_backend": "AMXBF16", "kt_model_max_length": 1152}
    hf_config = HfTrainerKTConfig(dict(raw_config))
    plugin = KTransformersPlugin(enabled=True, kt_config=dict(raw_config))

    assert hf_config.kt_backend == "AMXBF16"
    assert isinstance(plugin.kt_config, KTConfig)
    assert plugin.kt_config.kt_model_max_length == 1152
    assert "kt_config" in inspect.signature(Accelerator).parameters
