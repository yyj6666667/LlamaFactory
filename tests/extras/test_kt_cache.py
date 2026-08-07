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

import pytest
import torch
from safetensors.torch import save_file
from transformers import PretrainedConfig

from llamafactory.extras.kt_cache import (
    KT_NON_EXPERT_MANIFEST_NAME,
    _cache_fingerprint,
    prepare_kt_int8_cache_loading,
    validate_kt_int8_loaded_model,
    validate_kt_non_expert_cache,
)


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def _make_artifacts(tmp_path):
    source = tmp_path / "base"
    source.mkdir()
    source_config = {
        "model_type": "deepseek_v3",
        "hidden_size": 8,
        "moe_intermediate_size": 16,
        "n_routed_experts": 2,
        "num_hidden_layers": 61,
        "first_k_dense_replace": 3,
        "quantization_config": {"quant_method": "fp8", "weight_block_size": [128, 128]},
    }
    _write_json(source / "config.json", source_config)
    _write_json(
        source / "generation_config.json",
        {"bos_token_id": 11, "eos_token_id": 12, "max_new_tokens": 13},
    )
    _write_json(source / "model.safetensors.index.json", {"metadata": {}, "weight_map": {}})

    cache = tmp_path / "cache"
    cache.mkdir()
    shard_name = "model-00001-of-00001.safetensors"
    tensors = {
        "model.embed_tokens.weight": torch.ones(2, 3, dtype=torch.bfloat16),
        "model.layers.3.mlp.gate.e_score_correction_bias": torch.ones(2, dtype=torch.float32),
    }
    save_file(tensors, cache / shard_name)
    tensor_bytes = sum(tensor.numel() * tensor.element_size() for tensor in tensors.values())
    _write_json(
        cache / "model.safetensors.index.json",
        {"metadata": {"total_size": tensor_bytes}, "weight_map": dict.fromkeys(tensors, shard_name)},
    )
    records = [
        {"name": path.name, "size": path.stat().st_size, "sha256": _sha256(path)} for path in sorted(cache.iterdir())
    ]
    source_fingerprint = "a" * 64
    dtype_counts = {"BF16": 1, "F32": 1}
    manifest = {
        "version": 1,
        "status": "ready",
        "fingerprint": _cache_fingerprint(
            source_fingerprint,
            records,
            len(tensors),
            tensor_bytes,
            dtype_counts,
        ),
        "source": {
            "model_name_or_path": str(source),
            "fingerprint": source_fingerprint,
            "config_sha256": _sha256(source / "config.json"),
            "index_sha256": _sha256(source / "model.safetensors.index.json"),
        },
        "converter": {
            "name": "llamafactory.prepare-kt-cache",
            "version": 1,
            "default_dtype": "BF16",
            "fp32_exceptions": ["model.layers.*.mlp.gate.e_score_correction_bias"],
            "weight_block_size": [128, 128],
        },
        "files": records,
        "tensors": {"count": len(tensors), "bytes": tensor_bytes, "dtypes": dtype_counts},
    }
    _write_json(cache / KT_NON_EXPERT_MANIFEST_NAME, manifest)

    routed = tmp_path / "int8"
    routed.mkdir()
    routed_manifest = {
        "schema_version": 1,
        "state": "ready",
        "expert_weight_format": "int8",
        "backend": "AMXINT8",
        "threadpool_count": 2,
        "expert_num": 2,
        "hidden_size": 8,
        "intermediate_size": 16,
        "layers": [{"index": index} for index in range(3, 61)],
        "bytes": 1,
    }
    _write_json(routed / "kt-ephemeral-manifest.json", routed_manifest)
    return source, cache, routed


def _model_args(source, cache, routed):
    return SimpleNamespace(
        use_kt=True,
        model_name_or_path=str(source),
        kt_non_expert_weight_path=str(cache),
        kt_weight_path=str(routed),
        _kt_resolved_config={
            "kt_expert_weight_format": "int8",
            "kt_threadpool_count": 2,
            "kt_weight_lifecycle": "persistent",
        },
    )


def test_validates_existing_nonexpert_and_routed_int8_contract(tmp_path):
    source, cache, routed = _make_artifacts(tmp_path)

    resolved = validate_kt_non_expert_cache(
        cache,
        source,
        routed,
        {"kt_threadpool_count": 2},
    )

    assert resolved.path == str(cache)
    assert resolved.weight_keys == {
        "model.embed_tokens.weight",
        "model.layers.3.mlp.gate.e_score_correction_bias",
    }
    assert resolved.routed_manifest["expert_weight_format"] == "int8"


def test_rejects_tampered_cache_shard(tmp_path):
    source, cache, routed = _make_artifacts(tmp_path)
    with (cache / "model-00001-of-00001.safetensors").open("ab") as handle:
        handle.write(b"tampered")

    with pytest.raises(RuntimeError, match="size mismatch"):
        validate_kt_non_expert_cache(cache, source, routed, {"kt_threadpool_count": 2})


def test_rejects_routed_manifest_from_different_architecture(tmp_path):
    source, cache, routed = _make_artifacts(tmp_path)
    manifest_path = routed / "kt-ephemeral-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["hidden_size"] = 16
    _write_json(manifest_path, manifest)

    with pytest.raises(RuntimeError, match="hidden_size.*does not match"):
        validate_kt_non_expert_cache(cache, source, routed, {"kt_threadpool_count": 2})


def test_prepare_switches_only_weight_source_and_clears_source_quantizer(tmp_path):
    source, cache, routed = _make_artifacts(tmp_path)
    config = SimpleNamespace(quantization_config={"quant_method": "fp8"}, name_or_path=str(source))
    init_kwargs = {"config": config, "pretrained_model_name_or_path": str(source)}

    resolved = prepare_kt_int8_cache_loading(config, _model_args(source, cache, routed), init_kwargs)

    assert resolved is not None
    assert not hasattr(config, "quantization_config")
    assert config.name_or_path == str(source)
    assert init_kwargs["pretrained_model_name_or_path"] == str(cache)
    assert init_kwargs["generation_config"].bos_token_id == 11
    assert init_kwargs["generation_config"].eos_token_id == 12
    assert init_kwargs["generation_config"].max_new_tokens == 13
    assert init_kwargs["output_loading_info"] is True
    assert init_kwargs["local_files_only"] is True
    assert not (cache / "generation_config.json").exists()
    assert not (cache / "config.json").exists()


def test_prepare_uses_model_config_when_source_has_no_generation_config(tmp_path):
    source, cache, routed = _make_artifacts(tmp_path)
    (source / "generation_config.json").unlink()
    config = PretrainedConfig(bos_token_id=21, eos_token_id=22)
    config.quantization_config = {"quant_method": "fp8"}
    config.name_or_path = str(source)
    init_kwargs = {"config": config, "pretrained_model_name_or_path": str(source)}

    resolved = prepare_kt_int8_cache_loading(config, _model_args(source, cache, routed), init_kwargs)

    assert resolved is not None
    assert init_kwargs["pretrained_model_name_or_path"] == str(cache)
    assert init_kwargs["generation_config"].bos_token_id == 21
    assert init_kwargs["generation_config"].eos_token_id == 22


def test_filtered_frozen_loading_info_still_requires_exact_model_keys(tmp_path):
    source, cache, routed = _make_artifacts(tmp_path)
    resolved = validate_kt_non_expert_cache(cache, source, routed, {"kt_threadpool_count": 2})

    class LoadedModel:
        config = SimpleNamespace(name_or_path=str(cache))

        @staticmethod
        def state_dict():
            return {
                "model.embed_tokens.weight": torch.ones(2, 3, dtype=torch.bfloat16),
                "model.layers.3.mlp._original_router.weight": torch.empty(2, 8),
                "model.layers.3.mlp._original_router.e_score_correction_bias": torch.ones(2),
                "model.layers.3.mlp.experts.gate_up_proj": torch.empty(0),
            }

    model = LoadedModel()
    validate_kt_int8_loaded_model(
        model,
        {"missing_keys": set(), "unexpected_keys": set(), "mismatched_keys": set(), "error_msgs": []},
        resolved,
        str(source),
    )

    assert model.config.name_or_path == str(source)
    assert model._kt_non_expert_cache_manifest["status"] == "ready"


def test_loaded_model_rejects_nonexpert_missing_key(tmp_path):
    source, cache, routed = _make_artifacts(tmp_path)
    resolved = validate_kt_non_expert_cache(cache, source, routed, {"kt_threadpool_count": 2})
    model = SimpleNamespace(config=SimpleNamespace(), state_dict=lambda: {})

    with pytest.raises(RuntimeError, match="keys do not match"):
        validate_kt_int8_loaded_model(
            model,
            {"missing_keys": set(), "unexpected_keys": set(), "mismatched_keys": set(), "error_msgs": []},
            resolved,
            str(source),
        )
