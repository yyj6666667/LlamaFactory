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
import torch
from transformers import FineGrainedFP8Config
from transformers.integrations.finegrained_fp8 import Fp8Dequantize

from llamafactory.model.model_utils.quantization import (
    _cap_kt_deepseek_rope_cache,
    _patch_deepseek_remote_code_compatibility,
    _patch_fp8_partial_block_dequantization,
    configure_quantization,
)


def test_kt_deepseek_rope_cache_keeps_original_context_capacity():
    config = SimpleNamespace(
        max_position_embeddings=163840,
        rope_scaling={"original_max_position_embeddings": 4096},
    )
    model_args = SimpleNamespace(model_max_length=1152)

    _cap_kt_deepseek_rope_cache(config, model_args)

    assert config.max_position_embeddings == 4096


def test_fp8_partial_block_dequantization():
    _patch_fp8_partial_block_dequantization()
    quantized = torch.ones((3, 5), dtype=torch.float8_e4m3fn)
    scales = torch.tensor([[2.0, 3.0], [4.0, 5.0]])
    quantizer = SimpleNamespace(quantization_config=SimpleNamespace(weight_block_size=(2, 3)))

    result = Fp8Dequantize(quantizer).convert(
        {"weight$": [quantized], "weight_scale_inv": [scales]},
        full_layer_name="weight",
    )

    expected = torch.tensor(
        [
            [2.0, 2.0, 2.0, 3.0, 3.0],
            [2.0, 2.0, 2.0, 3.0, 3.0],
            [4.0, 4.0, 4.0, 5.0, 5.0],
        ]
    )
    torch.testing.assert_close(result["weight"], expected)


def test_fp8_partial_block_dequantization_preserves_scale_dtype():
    _patch_fp8_partial_block_dequantization()
    quantized = torch.ones((3, 5), dtype=torch.float8_e4m3fn)
    scales = torch.ones((2, 2), dtype=torch.bfloat16)
    quantizer = SimpleNamespace(quantization_config=SimpleNamespace(weight_block_size=(2, 3)))

    result = Fp8Dequantize(quantizer).convert(
        {"weight$": [quantized], "weight_scale_inv": [scales]},
        full_layer_name="weight",
    )

    assert result["weight"].dtype == torch.bfloat16


def test_fp8_dequantization_uses_configured_bf16_output_for_complete_blocks():
    _patch_fp8_partial_block_dequantization()
    quantized = torch.ones((4, 6), dtype=torch.float8_e4m3fn)
    scales = torch.ones((2, 2), dtype=torch.float32)
    quantization_config = SimpleNamespace(
        weight_block_size=(2, 3),
        _llamafactory_dequantization_dtype=torch.bfloat16,
    )
    quantizer = SimpleNamespace(quantization_config=quantization_config)

    result = Fp8Dequantize(quantizer).convert(
        {"weight$": [quantized], "weight_scale_inv": [scales]},
        full_layer_name="weight",
    )

    assert result["weight"].dtype == torch.bfloat16


def test_fp8_dequantization_casts_unquantized_weight_to_configured_dtype():
    _patch_fp8_partial_block_dequantization()
    quantization_config = SimpleNamespace(
        weight_block_size=(2, 3),
        _llamafactory_dequantization_dtype=torch.bfloat16,
    )
    quantizer = SimpleNamespace(quantization_config=quantization_config)

    result = Fp8Dequantize(quantizer).convert(
        {"weight$": [torch.ones((2, 2), dtype=torch.float32)]},
        full_layer_name="weight",
    )

    assert result["weight"][0].dtype == torch.bfloat16


def test_fp8_dequantization_dtype_survives_checkpoint_config_merge():
    loading_config = FineGrainedFP8Config(dequantize=True, weight_block_size=(2, 3))
    checkpoint_config = FineGrainedFP8Config(weight_block_size=(2, 3))
    assert not hasattr(checkpoint_config, "_llamafactory_dequantization_dtype")

    _patch_fp8_partial_block_dequantization(loading_config, torch.bfloat16)
    for name, value in loading_config.get_loading_attributes().items():
        setattr(checkpoint_config, name, value)

    quantized = torch.ones((4, 6), dtype=torch.float8_e4m3fn)
    scales = torch.ones((2, 2), dtype=torch.float32)
    quantizer = SimpleNamespace(quantization_config=checkpoint_config)
    result = Fp8Dequantize(quantizer).convert(
        {"weight$": [quantized], "weight_scale_inv": [scales]},
        full_layer_name="weight",
    )

    assert checkpoint_config._llamafactory_dequantization_dtype == torch.bfloat16
    assert result["weight"].dtype == torch.bfloat16


def test_fp8_partial_block_dequantization_rejects_wrong_scale_shape():
    _patch_fp8_partial_block_dequantization()
    quantized = torch.ones((3, 5), dtype=torch.float8_e4m3fn)
    scales = torch.ones((1, 2))
    quantizer = SimpleNamespace(quantization_config=SimpleNamespace(weight_block_size=(2, 3)))

    with pytest.raises(ValueError, match="expected partial-block shape"):
        Fp8Dequantize(quantizer).convert(
            {"weight$": [quantized], "weight_scale_inv": [scales]},
            full_layer_name="weight",
        )


def test_deepseek_remote_code_compatibility_bridge(monkeypatch):
    from transformers.utils import import_utils

    monkeypatch.delattr(import_utils, "is_torch_fx_available", raising=False)
    _patch_deepseek_remote_code_compatibility()

    assert import_utils.is_torch_fx_available is import_utils.is_torch_available


def test_deepseek_remote_code_compatibility_rejects_unknown_version(monkeypatch):
    import transformers
    from transformers.utils import import_utils

    monkeypatch.delattr(import_utils, "is_torch_fx_available", raising=False)
    monkeypatch.setattr(transformers, "__version__", "6.0.0")

    with pytest.raises(RuntimeError, match="requires Transformers 5.x"):
        _patch_deepseek_remote_code_compatibility()


def test_kt_int8_dequantizes_non_expert_fp8_weights():
    config = SimpleNamespace(model_type="deepseek_v3", quantization_config={"quant_method": "fp8"})
    model_args = SimpleNamespace(
        quantization_bit=None,
        use_kt=True,
        kt_weight_path="/tmp/int8-weights",
        kt_expert_weight_format="int8",
    )
    init_kwargs = {}

    configure_quantization(config, None, model_args, True, init_kwargs)

    assert init_kwargs["quantization_config"].dequantize is True
    assert init_kwargs["quantization_config"]._llamafactory_dequantization_dtype == torch.bfloat16
    assert init_kwargs["ignore_mismatched_sizes"] is True
