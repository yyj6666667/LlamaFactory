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

from llamafactory.model.model_utils.quantization import configure_quantization


def _fp8_config():
    return SimpleNamespace(quantization_config={"quant_method": "fp8", "bits": 8})


def test_kt_int8_cache_skips_source_fp8_dequantizer(monkeypatch):
    from transformers.integrations import kt

    monkeypatch.setattr(kt, "is_kt_int8_expert_loading_enabled", lambda: True)
    monkeypatch.setattr(kt, "_get_kt_config", lambda: None)
    monkeypatch.setenv("ACCELERATE_KT_NON_EXPERT_WEIGHT_PATH", "/tmp/nonexpert-cache")
    init_kwargs = {}

    configure_quantization(
        _fp8_config(),
        tokenizer=None,
        model_args=SimpleNamespace(use_kt=True, quantization_bit=None),
        is_trainable=True,
        init_kwargs=init_kwargs,
    )

    assert "quantization_config" not in init_kwargs
    assert "ignore_mismatched_sizes" not in init_kwargs


def test_standard_fp8_loading_keeps_source_dequantizer(monkeypatch):
    from transformers.integrations import kt

    monkeypatch.setattr(kt, "is_kt_int8_expert_loading_enabled", lambda: False)
    monkeypatch.setenv("ACCELERATE_KT_NON_EXPERT_WEIGHT_PATH", "/tmp/nonexpert-cache")
    init_kwargs = {}

    configure_quantization(
        _fp8_config(),
        tokenizer=None,
        model_args=SimpleNamespace(use_kt=True, quantization_bit=None),
        is_trainable=True,
        init_kwargs=init_kwargs,
    )

    assert init_kwargs["quantization_config"].dequantize is True
    assert init_kwargs["ignore_mismatched_sizes"] is True
