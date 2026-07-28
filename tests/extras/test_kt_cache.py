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

import hashlib
import json
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from llamafactory.extras.kt_cache import prepare_kt_cache


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_prepare_kt_cache_is_atomic_and_excludes_experts(tmp_path):
    model_path = tmp_path / "model"
    output_path = tmp_path / "cache"
    model_path.mkdir()
    config = {
        "model_type": "deepseek_v3",
        "quantization_config": {"quant_method": "fp8", "weight_block_size": [2, 3]},
    }
    (model_path / "config.json").write_text(json.dumps(config), encoding="utf-8")

    quantized = torch.ones((3, 5), dtype=torch.float8_e4m3fn)
    scales = torch.tensor([[2.0, 3.0], [4.0, 5.0]], dtype=torch.float32)
    source_tensors = {
        "model.embed_tokens.weight": torch.arange(12, dtype=torch.bfloat16).reshape(3, 4),
        "model.layers.0.mlp.gate.e_score_correction_bias": torch.arange(2, dtype=torch.float32),
        "model.layers.0.self_attn.q_proj.weight": quantized,
        "model.layers.0.self_attn.q_proj.weight_scale_inv": scales,
        "model.layers.3.mlp.experts.0.gate_proj.weight": torch.ones((2, 2), dtype=torch.float8_e4m3fn),
        "model.layers.61.self_attn.q_proj.weight": torch.ones((2, 2), dtype=torch.bfloat16),
    }
    shard_name = "model-00001-of-00001.safetensors"
    save_file(source_tensors, model_path / shard_name)
    index = {
        "metadata": {},
        "weight_map": dict.fromkeys(source_tensors, shard_name),
    }
    (model_path / "model.safetensors.index.json").write_text(json.dumps(index), encoding="utf-8")
    training_config = tmp_path / "train.yaml"
    training_config.write_text(
        "\n".join(
            [
                f"model_name_or_path: {model_path}",
                "trust_remote_code: false",
                "kt_expert_weight_format: int8",
                f"kt_non_expert_weight_path: {output_path}",
            ]
        ),
        encoding="utf-8",
    )

    manifest = prepare_kt_cache(str(training_config), shard_bytes=128)

    assert manifest["status"] == "ready"
    assert manifest["tensors"]["count"] == 3
    assert not list(tmp_path.glob(".cache.tmp-*"))
    cache_manifest = json.loads((output_path / "kt_non_expert_manifest.json").read_text(encoding="utf-8"))
    assert len(cache_manifest["fingerprint"]) == 64
    assert cache_manifest["source"]["model_name_or_path"] == str(model_path.resolve())
    assert cache_manifest["tensors"]["dtypes"] == {"BF16": 2, "F32": 1}
    for file_record in cache_manifest["files"]:
        assert _sha256(output_path / file_record["name"]) == file_record["sha256"]

    cache_index = json.loads((output_path / "model.safetensors.index.json").read_text(encoding="utf-8"))
    assert set(cache_index["weight_map"]) == {
        "model.embed_tokens.weight",
        "model.layers.0.mlp.gate.e_score_correction_bias",
        "model.layers.0.self_attn.q_proj.weight",
    }
    q_proj_shard = output_path / cache_index["weight_map"]["model.layers.0.self_attn.q_proj.weight"]
    with safe_open(q_proj_shard, framework="pt", device="cpu") as file:
        converted = file.get_tensor("model.layers.0.self_attn.q_proj.weight")

    expected = torch.tensor(
        [
            [2.0, 2.0, 2.0, 3.0, 3.0],
            [2.0, 2.0, 2.0, 3.0, 3.0],
            [4.0, 4.0, 4.0, 5.0, 5.0],
        ],
        dtype=torch.bfloat16,
    )
    torch.testing.assert_close(converted, expected)
    bias_shard = output_path / cache_index["weight_map"]["model.layers.0.mlp.gate.e_score_correction_bias"]
    with safe_open(bias_shard, framework="pt", device="cpu") as file:
        assert file.get_tensor("model.layers.0.mlp.gate.e_score_correction_bias").dtype == torch.float32


def test_launcher_preserves_cache_config_argument(monkeypatch):
    from llamafactory import launcher
    from llamafactory.extras import kt_cache

    observed = []
    monkeypatch.setattr(sys, "argv", ["llamafactory-cli", "prepare-kt-cache", "train.yaml"])
    monkeypatch.setattr(kt_cache, "prepare_kt_cache_cli", lambda: observed.append(list(sys.argv)))

    launcher.launch()

    assert observed == [["llamafactory-cli", "train.yaml"]]
