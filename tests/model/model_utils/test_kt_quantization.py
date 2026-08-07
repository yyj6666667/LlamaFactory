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

from types import SimpleNamespace

import pytest

from llamafactory.model.model_utils.quantization import configure_quantization


def _fp8_config():
    return SimpleNamespace(quantization_config={"quant_method": "fp8", "bits": 8})


def _model_args(**overrides):
    values = {
        "use_kt": True,
        "quantization_bit": None,
        "kt_non_expert_weight_path": "/tmp/nonexpert-cache",
        "_kt_resolved_config": {"kt_expert_weight_format": "int8"},
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_kt_int8_cache_skips_source_fp8_dequantizer():
    init_kwargs = {}

    configure_quantization(
        _fp8_config(),
        tokenizer=None,
        model_args=_model_args(),
        is_trainable=True,
        init_kwargs=init_kwargs,
    )

    assert "quantization_config" not in init_kwargs
    assert "ignore_mismatched_sizes" not in init_kwargs


def test_kt_int8_resolved_config_skips_source_fp8_dequantizer():
    init_kwargs = {}
    configure_quantization(
        _fp8_config(),
        tokenizer=None,
        model_args=_model_args(),
        is_trainable=True,
        init_kwargs=init_kwargs,
    )

    assert "quantization_config" not in init_kwargs


def test_kt_int8_without_non_expert_cache_keeps_source_dequantizer():
    init_kwargs = {}

    configure_quantization(
        _fp8_config(),
        tokenizer=None,
        model_args=_model_args(kt_non_expert_weight_path=None),
        is_trainable=True,
        init_kwargs=init_kwargs,
    )

    assert init_kwargs["quantization_config"].dequantize is True
    assert init_kwargs["ignore_mismatched_sizes"] is True


def test_kt_int8_cache_rejects_on_the_fly_quantization():
    with pytest.raises(ValueError, match="quantization_bit.*KT INT8/BF16"):
        configure_quantization(
            _fp8_config(),
            tokenizer=None,
            model_args=_model_args(quantization_bit=4),
            is_trainable=True,
            init_kwargs={},
        )
