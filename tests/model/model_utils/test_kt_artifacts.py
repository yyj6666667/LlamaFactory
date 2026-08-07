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
from types import SimpleNamespace

import torch
from safetensors.torch import save_file

from llamafactory.model.model_utils.kt_artifacts import publish_kt_int8_adapter_manifest


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_publish_kt_int8_adapter_manifest(tmp_path):
    routed = tmp_path / "routed"
    routed.mkdir()
    routed_manifest = routed / "kt-ephemeral-manifest.json"
    routed_manifest.write_text('{"state":"ready"}\n', encoding="utf-8")

    cache = tmp_path / "cache"
    cache.mkdir()
    cache_manifest = {
        "fingerprint": "b" * 64,
        "source": {"model_name_or_path": "/models/base", "fingerprint": "a" * 64},
    }
    (cache / "kt_non_expert_manifest.json").write_text(
        json.dumps(cache_manifest), encoding="utf-8"
    )

    output = tmp_path / "adapter"
    output.mkdir()
    (output / "adapter_config.json").write_text('{"r":8}\n', encoding="utf-8")
    save_file({"router.lora_B.weight": torch.ones(2, 2, dtype=torch.bfloat16)}, output / "adapter_model.safetensors")
    fused = {
        f"layers.3.experts.{name}": torch.ones(2, 2, dtype=torch.bfloat16)
        for name in (
            "gate_lora_a",
            "gate_lora_b",
            "up_lora_a",
            "up_lora_b",
            "down_lora_a",
            "down_lora_b",
        )
    }
    save_file(fused, output / "fused_expert_lora.safetensors")

    model = torch.nn.Module()
    model._kt_non_expert_cache_path = str(cache)
    model._kt_non_expert_cache_manifest = cache_manifest
    model._kt_routed_int8_manifest_path = str(routed_manifest)
    model._kt_routed_int8_manifest = {"state": "ready"}
    model_args = SimpleNamespace(
        kt_weight_path=str(routed),
        _kt_resolved_config={
            "kt_expert_weight_format": "int8",
            "kt_lora_rank": 8,
            "kt_lora_alpha": 16,
        },
    )

    manifest = publish_kt_int8_adapter_manifest(model, output, model_args)

    assert manifest is not None
    assert manifest["status"] == "ready"
    assert manifest["int8_experts"]["manifest_sha256"] == _sha256(routed_manifest)
    assert manifest["lora"] == {"rank": 8, "alpha": 16.0}
    fused_record = manifest["artifacts"]["fused_expert_lora.safetensors"]
    assert fused_record["tensor_count"] == 6
    assert set(fused_record["tensors"]) == set(fused)
    saved = json.loads((output / "kt_adapter_manifest.json").read_text(encoding="utf-8"))
    assert saved == manifest


def test_non_int8_adapter_does_not_publish_manifest(tmp_path):
    model_args = SimpleNamespace(_kt_resolved_config={"kt_expert_weight_format": "bf16"})
    assert publish_kt_int8_adapter_manifest(torch.nn.Module(), tmp_path, model_args) is None
