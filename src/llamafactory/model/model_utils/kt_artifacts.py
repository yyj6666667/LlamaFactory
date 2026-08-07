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

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from safetensors import safe_open

from ...extras.kt_cache import KT_NON_EXPERT_MANIFEST_NAME


if TYPE_CHECKING:
    from ...hparams import ModelArguments


KT_ADAPTER_MANIFEST_NAME = "kt_adapter_manifest.json"
_FUSED_EXPERT_LORA_NAME = "fused_expert_lora.safetensors"
_STANDARD_ADAPTER_NAMES = ("adapter_model.safetensors", "adapter_model.bin")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _regular_file(path: Path) -> Path:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"KT adapter artifact must be a regular file: {path}.")
    return path


def _artifact_record(path: Path) -> dict[str, Any]:
    _regular_file(path)
    return {"size": path.stat().st_size, "sha256": _sha256_file(path)}


def _read_json(path: Path) -> dict[str, Any]:
    _regular_file(path)
    try:
        with path.open(encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Could not read KT artifact manifest {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"KT artifact manifest must contain a JSON object: {path}.")
    return value


def _fused_tensor_contract(path: Path) -> dict[str, dict[str, Any]]:
    contract: dict[str, dict[str, Any]] = {}
    with safe_open(path, framework="pt", device="cpu") as handle:
        for name in handle.keys():
            tensor = handle.get_slice(name)
            dtype = tensor.get_dtype()
            if dtype != "BF16":
                raise RuntimeError(f"KT fused adapter tensor {name!r} must be BF16, got {dtype}.")
            contract[name] = {"shape": list(tensor.get_shape()), "dtype": dtype}
    if not contract or len(contract) % 6:
        raise RuntimeError("KT fused adapter must contain six non-empty tensors per MoE layer.")
    return contract


def _find_cache_owner(model: torch.nn.Module) -> torch.nn.Module:
    owners = [module for module in model.modules() if "_kt_non_expert_cache_manifest" in vars(module)]
    if len(owners) != 1:
        raise RuntimeError(f"KT INT8 adapter save expected one cache provenance owner, found {len(owners)}.")
    return owners[0]


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def publish_kt_int8_adapter_manifest(
    model: torch.nn.Module,
    output_dir: str | os.PathLike[str],
    model_args: ModelArguments,
) -> dict[str, Any] | None:
    r"""Publish DeepSeek routed-INT8 adapter provenance after all adapter files are complete."""
    runtime_config = getattr(model_args, "_kt_resolved_config", {})
    if not isinstance(runtime_config, dict) or runtime_config.get("kt_expert_weight_format") != "int8":
        return None

    output_path = Path(output_dir).absolute()
    if output_path.is_symlink() or not output_path.is_dir():
        raise RuntimeError(f"KT adapter output must be a real directory: {output_path}.")

    owner = _find_cache_owner(model)
    cache_manifest = getattr(owner, "_kt_non_expert_cache_manifest", None)
    cache_path = getattr(owner, "_kt_non_expert_cache_path", None)
    routed_manifest_path_value = getattr(owner, "_kt_routed_int8_manifest_path", None)
    routed_manifest = getattr(owner, "_kt_routed_int8_manifest", None)
    if (
        not isinstance(cache_manifest, dict)
        or not isinstance(cache_path, str)
        or not cache_path
        or not isinstance(routed_manifest_path_value, str)
        or not routed_manifest_path_value
        or not isinstance(routed_manifest, dict)
    ):
        raise RuntimeError("KT INT8 adapter save requires validated cache provenance.")

    cache_root = Path(cache_path).absolute()
    routed_manifest_path = Path(routed_manifest_path_value).absolute()
    current_cache_manifest = _read_json(cache_root / KT_NON_EXPERT_MANIFEST_NAME)
    if current_cache_manifest != cache_manifest:
        raise RuntimeError("KT non-expert cache manifest changed after model loading.")
    current_routed_manifest = _read_json(routed_manifest_path)
    if current_routed_manifest != routed_manifest:
        raise RuntimeError("KT routed INT8 manifest changed after model loading.")

    weight_path = getattr(model_args, "kt_weight_path", None)
    if not isinstance(weight_path, str) or not weight_path:
        raise RuntimeError("KT INT8 adapter provenance requires `kt_weight_path`.")
    if os.path.realpath(weight_path) != os.path.realpath(routed_manifest_path.parent):
        raise RuntimeError("KT routed INT8 manifest does not belong to the configured weight path.")

    source = cache_manifest.get("source")
    if not isinstance(source, dict):
        raise RuntimeError("KT non-expert cache manifest has no source provenance.")
    for field, value in {
        "source.model_name_or_path": source.get("model_name_or_path"),
        "source.fingerprint": source.get("fingerprint"),
        "cache.fingerprint": cache_manifest.get("fingerprint"),
        "cache.path": cache_path,
    }.items():
        if not isinstance(value, str) or not value:
            raise RuntimeError(f"KT INT8 adapter provenance requires `{field}`.")

    routed_digest = _sha256_file(routed_manifest_path)
    routed_fingerprint = routed_manifest.get("fingerprint") or routed_digest
    rank = runtime_config.get("kt_lora_rank")
    alpha = runtime_config.get("kt_lora_alpha")
    if isinstance(rank, bool) or not isinstance(rank, int) or rank <= 0:
        raise RuntimeError("KT INT8 adapter provenance requires a positive LoRA rank.")
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
        raise RuntimeError("KT INT8 adapter provenance requires numeric LoRA alpha.")

    standard_names = [name for name in _STANDARD_ADAPTER_NAMES if (output_path / name).is_file()]
    if len(standard_names) != 1:
        raise RuntimeError(f"KT INT8 adapter save expected one standard adapter file, found {standard_names}.")
    fused_path = _regular_file(output_path / _FUSED_EXPERT_LORA_NAME)
    fused_contract = _fused_tensor_contract(fused_path)

    artifacts = {
        "adapter_config.json": _artifact_record(output_path / "adapter_config.json"),
        standard_names[0]: _artifact_record(output_path / standard_names[0]),
        _FUSED_EXPERT_LORA_NAME: {
            **_artifact_record(fused_path),
            "tensor_count": len(fused_contract),
            "tensors": fused_contract,
        },
    }
    payload = {
        "version": 1,
        "status": "ready",
        "base": {
            "model_name_or_path": source["model_name_or_path"],
            "fingerprint": source["fingerprint"],
        },
        "non_expert_cache": {
            "path": str(cache_root),
            "fingerprint": cache_manifest["fingerprint"],
        },
        "int8_experts": {
            "path": os.path.abspath(weight_path),
            "manifest": routed_manifest_path.name,
            "manifest_sha256": routed_digest,
            "fingerprint": routed_fingerprint,
        },
        "lora": {"rank": rank, "alpha": float(alpha)},
        "artifacts": artifacts,
    }
    _write_json_atomic(output_path / KT_ADAPTER_MANIFEST_NAME, payload)
    return payload
