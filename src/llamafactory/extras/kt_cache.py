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
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch.distributed as dist
from safetensors import safe_open
from transformers import GenerationConfig

from . import logging


if TYPE_CHECKING:
    from transformers import PretrainedConfig, PreTrainedModel

    from ..hparams import ModelArguments


logger = logging.get_logger(__name__)

KT_NON_EXPERT_MANIFEST_NAME = "kt_non_expert_manifest.json"
KT_NON_EXPERT_INDEX_NAME = "model.safetensors.index.json"
_MANIFEST_VERSION = 1
_CONVERTER_NAME = "llamafactory.prepare-kt-cache"
_FP32_ROUTER_BIAS = re.compile(r"^model\.layers\.\d+\.mlp\.gate\.e_score_correction_bias$")
_ROUTED_EXPERT = re.compile(r"\.(?:mlp|block_sparse_moe)\.experts\.")
_ROUTED_MANIFEST_NAMES = ("kt-weight-manifest.json", "kt-ephemeral-manifest.json")


@dataclass(frozen=True)
class KTNonExpertCache:
    path: str
    manifest: dict[str, Any]
    checkpoint_files: tuple[str, ...]
    weight_keys: frozenset[str]
    source_config: dict[str, Any]
    routed_manifest_path: str
    routed_manifest: dict[str, Any]

    @property
    def fingerprint(self) -> str:
        return self.manifest["fingerprint"]


def is_kt_int8_cache_requested(model_args: ModelArguments) -> bool:
    config = getattr(model_args, "_kt_resolved_config", {})
    weight_format = config.get("kt_expert_weight_format") if isinstance(config, dict) else None
    return bool(
        getattr(model_args, "use_kt", False)
        and weight_format == "int8"
        and getattr(model_args, "kt_non_expert_weight_path", None)
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path, description: str) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Could not read {description} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"{description.capitalize()} {path} must contain a JSON object.")
    return value


def _require_string(value: Any, field: str, manifest_path: Path) -> str:
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"{manifest_path}: `{field}` must be a non-empty string.")
    return value


def _require_sha256(value: Any, field: str, manifest_path: Path) -> str:
    digest = _require_string(value, field, manifest_path)
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise RuntimeError(f"{manifest_path}: `{field}` must be a lowercase SHA256 digest.")
    return digest


def _same_source(expected: str | os.PathLike[str], actual: str) -> bool:
    expected = os.fspath(expected)
    if os.path.exists(expected) or os.path.exists(actual):
        return os.path.realpath(expected) == os.path.realpath(actual)
    return expected == actual


def _cache_fingerprint(
    source_fingerprint: str,
    file_records: list[dict[str, Any]],
    tensor_count: int,
    tensor_bytes: int,
    dtype_counts: dict[str, int],
) -> str:
    digest = hashlib.sha256()
    digest.update(f"kt-non-expert-cache-v{_MANIFEST_VERSION}\0".encode())
    digest.update(source_fingerprint.encode())
    digest.update(b"\0")
    digest.update(str(tensor_count).encode())
    digest.update(b"\0")
    digest.update(str(tensor_bytes).encode())
    digest.update(b"\0")
    digest.update(json.dumps(dtype_counts, sort_keys=True, separators=(",", ":")).encode())
    digest.update(b"\0")
    for record in sorted(file_records, key=lambda item: item["name"]):
        digest.update(record["name"].encode())
        digest.update(b"\0")
        digest.update(str(record["size"]).encode())
        digest.update(b"\0")
        digest.update(record["sha256"].encode())
        digest.update(b"\0")
    return digest.hexdigest()


def _positive_int(value: Any, field: str, path: Path) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RuntimeError(f"{path}: `{field}` must be a positive integer.")
    return value


def _validate_routed_manifest(
    weight_path: str | os.PathLike[str],
    source_config: dict[str, Any],
    runtime_config: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    root = Path(weight_path).expanduser()
    if not root.is_absolute() or root.is_symlink() or not root.is_dir():
        raise RuntimeError(f"KT routed INT8 weight path must be an absolute, non-symlink directory: {root}.")
    candidates = [root / name for name in _ROUTED_MANIFEST_NAMES if (root / name).exists()]
    if len(candidates) != 1:
        raise RuntimeError(
            f"{root} must contain exactly one routed INT8 manifest from {_ROUTED_MANIFEST_NAMES}; "
            f"found {[path.name for path in candidates]}."
        )
    manifest_path = candidates[0]
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise RuntimeError(f"KT routed INT8 manifest must be a regular file: {manifest_path}.")
    manifest = _read_json(manifest_path, "KT routed INT8 manifest")
    if manifest.get("schema_version") not in {1, 2}:
        raise RuntimeError(f"{manifest_path}: unsupported `schema_version` {manifest.get('schema_version')!r}.")
    if manifest.get("state") != "ready" or manifest.get("expert_weight_format") != "int8":
        raise RuntimeError(f"{manifest_path}: routed weights must be ready INT8 artifacts.")

    if source_config.get("model_type") != "deepseek_v3":
        raise RuntimeError("Routed INT8 non-expert cache loading currently requires a DeepSeek-V3 checkpoint.")
    contract = {
        "expert_num": source_config.get("n_routed_experts"),
        "hidden_size": source_config.get("hidden_size"),
        "intermediate_size": source_config.get("moe_intermediate_size"),
    }
    for field, expected in contract.items():
        _positive_int(expected, f"source.{field}", manifest_path)
        if manifest.get(field) != expected:
            raise RuntimeError(
                f"{manifest_path}: `{field}` does not match the source model: "
                f"expected {expected!r}, got {manifest.get(field)!r}."
            )

    configured_threadpools = runtime_config.get("kt_threadpool_count")
    if configured_threadpools is not None and manifest.get("threadpool_count") != configured_threadpools:
        raise RuntimeError(
            f"{manifest_path}: `threadpool_count` does not match kt_config: "
            f"expected {configured_threadpools!r}, got {manifest.get('threadpool_count')!r}."
        )

    layer_count = _positive_int(source_config.get("num_hidden_layers"), "source.num_hidden_layers", manifest_path)
    first_moe = _positive_int(
        source_config.get("first_k_dense_replace"), "source.first_k_dense_replace", manifest_path
    )
    layers = manifest.get("layers")
    if not isinstance(layers, list) or any(not isinstance(layer, dict) for layer in layers):
        raise RuntimeError(f"{manifest_path}: `layers` must be a list of objects.")
    layer_indices = [layer.get("index") for layer in layers]
    expected_layers = list(range(first_moe, layer_count))
    if layer_indices != expected_layers:
        raise RuntimeError(
            f"{manifest_path}: routed layer set does not match the source model: "
            f"expected {expected_layers}, got {layer_indices}."
        )
    return str(manifest_path), manifest


def _validate_cache_local(
    cache_path: str | os.PathLike[str],
    source_model_name_or_path: str | os.PathLike[str],
    weight_path: str | os.PathLike[str],
    runtime_config: dict[str, Any],
    *,
    verify_shard_hashes: bool,
) -> KTNonExpertCache:
    root = Path(cache_path).expanduser().absolute()
    if root.is_symlink() or not root.is_dir():
        raise RuntimeError(f"KT non-expert cache must be a real local directory, got {root}.")

    manifest_path = root / KT_NON_EXPERT_MANIFEST_NAME
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise RuntimeError(f"KT non-expert cache manifest must be a regular file: {manifest_path}.")
    manifest = _read_json(manifest_path, "KT non-expert cache manifest")
    if manifest.get("version") != _MANIFEST_VERSION:
        raise RuntimeError(f"{manifest_path}: unsupported version {manifest.get('version')!r}.")
    if manifest.get("status") != "ready":
        raise RuntimeError(f"{manifest_path}: `status` must be `ready`.")
    fingerprint = _require_sha256(manifest.get("fingerprint"), "fingerprint", manifest_path)

    source = manifest.get("source")
    if not isinstance(source, dict):
        raise RuntimeError(f"{manifest_path}: `source` must be an object.")
    source_path_value = _require_string(source.get("model_name_or_path"), "source.model_name_or_path", manifest_path)
    source_fingerprint = _require_sha256(source.get("fingerprint"), "source.fingerprint", manifest_path)
    if not _same_source(source_model_name_or_path, source_path_value):
        raise RuntimeError(
            f"{manifest_path}: cache source {source_path_value!r} does not match requested base "
            f"{os.fspath(source_model_name_or_path)!r}."
        )
    source_path = Path(source_path_value)
    source_files = {
        "config_sha256": source_path / "config.json",
        "index_sha256": source_path / KT_NON_EXPERT_INDEX_NAME,
    }
    for field, path in source_files.items():
        expected = _require_sha256(source.get(field), f"source.{field}", manifest_path)
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"{manifest_path}: source artifact must be a regular file: {path}.")
        if _sha256_file(path) != expected:
            raise RuntimeError(f"{manifest_path}: source.{field} does not match {path}.")

    source_config = _read_json(source_files["config_sha256"], "source model config")
    source_layer_count = _positive_int(
        source_config.get("num_hidden_layers"), "source.num_hidden_layers", manifest_path
    )
    mtp_prefix = f"model.layers.{source_layer_count}."
    source_quantization = source_config.get("quantization_config")
    if not isinstance(source_quantization, dict) or source_quantization.get("quant_method") != "fp8":
        raise RuntimeError(f"{manifest_path}: source config must describe an FP8 checkpoint.")
    block_size = source_quantization.get("weight_block_size")
    if (
        not isinstance(block_size, (list, tuple))
        or len(block_size) != 2
        or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in block_size)
    ):
        raise RuntimeError(f"{manifest_path}: source config must contain a positive two-element FP8 block size.")

    expected_converter = {
        "name": _CONVERTER_NAME,
        "version": _MANIFEST_VERSION,
        "default_dtype": "BF16",
        "fp32_exceptions": ["model.layers.*.mlp.gate.e_score_correction_bias"],
        "weight_block_size": list(block_size),
    }
    if manifest.get("converter") != expected_converter:
        raise RuntimeError(f"{manifest_path}: converter contract does not match the supported cache producer.")

    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise RuntimeError(f"{manifest_path}: `files` must be a non-empty list.")
    file_records: dict[str, dict[str, Any]] = {}
    for index, record in enumerate(files):
        if not isinstance(record, dict):
            raise RuntimeError(f"{manifest_path}: files[{index}] must be an object.")
        name = _require_string(record.get("name"), f"files[{index}].name", manifest_path)
        if name != os.path.basename(name) or name in file_records:
            raise RuntimeError(f"{manifest_path}: invalid or duplicate cache filename {name!r}.")
        _require_sha256(record.get("sha256"), f"files[{index}].sha256", manifest_path)
        _positive_int(record.get("size"), f"files[{index}].size", manifest_path)
        file_records[name] = record

    index_path = root / KT_NON_EXPERT_INDEX_NAME
    if KT_NON_EXPERT_INDEX_NAME not in file_records:
        raise RuntimeError(f"{manifest_path}: cache manifest does not list {KT_NON_EXPERT_INDEX_NAME}.")
    cache_index = _read_json(index_path, "KT non-expert safetensors index")
    weight_map = cache_index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise RuntimeError(f"{index_path}: `weight_map` must be a non-empty object.")
    shard_names = set()
    for key, shard_name in weight_map.items():
        if not isinstance(key, str) or not key:
            raise RuntimeError(f"{index_path}: tensor names must be non-empty strings.")
        if not isinstance(shard_name, str) or shard_name != os.path.basename(shard_name):
            raise RuntimeError(f"{index_path}: invalid shard name {shard_name!r}.")
        shard_names.add(shard_name)
    if not all(name.endswith(".safetensors") for name in shard_names):
        raise RuntimeError(f"{index_path}: every weight shard must use safetensors.")
    expected_files = shard_names | {KT_NON_EXPERT_INDEX_NAME}
    if set(file_records) != expected_files:
        raise RuntimeError(
            f"{manifest_path}: file inventory differs from the index: "
            f"expected {sorted(expected_files)}, got {sorted(file_records)}."
        )

    tensors = manifest.get("tensors")
    if not isinstance(tensors, dict):
        raise RuntimeError(f"{manifest_path}: `tensors` must be an object.")
    tensor_count = tensors.get("count")
    tensor_bytes = tensors.get("bytes")
    dtype_counts = tensors.get("dtypes")
    _positive_int(tensor_count, "tensors.count", manifest_path)
    if tensor_count != len(weight_map):
        raise RuntimeError(f"{manifest_path}: tensor count does not match the cache index.")
    _positive_int(tensor_bytes, "tensors.bytes", manifest_path)
    if (
        not isinstance(dtype_counts, dict)
        or not dtype_counts
        or set(dtype_counts) - {"BF16", "F32"}
        or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in dtype_counts.values())
        or sum(dtype_counts.values()) != tensor_count
    ):
        raise RuntimeError(f"{manifest_path}: tensors.dtypes must be positive BF16/F32 counts.")
    expected_fingerprint = _cache_fingerprint(
        source_fingerprint, list(file_records.values()), tensor_count, tensor_bytes, dtype_counts
    )
    if fingerprint != expected_fingerprint:
        raise RuntimeError(f"{manifest_path}: fingerprint does not match the source, files, and tensors.")

    observed_keys = set()
    observed_bytes = 0
    observed_dtypes: dict[str, int] = {}
    checkpoint_files = []
    for name, record in file_records.items():
        path = root / name
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"{manifest_path}: cache file must be a regular file: {path}.")
        if path.stat().st_size != record["size"]:
            raise RuntimeError(f"{manifest_path}: size mismatch for {name}.")
        if name == KT_NON_EXPERT_INDEX_NAME or verify_shard_hashes:
            if _sha256_file(path) != record["sha256"]:
                raise RuntimeError(f"{manifest_path}: SHA256 mismatch for {name}.")
        if name == KT_NON_EXPERT_INDEX_NAME:
            continue

        checkpoint_files.append(str(path))
        with safe_open(path, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key in observed_keys:
                    raise RuntimeError(f"{manifest_path}: duplicate tensor {key!r} across cache shards.")
                tensor_slice = handle.get_slice(key)
                dtype = tensor_slice.get_dtype()
                expected_dtype = "F32" if _FP32_ROUTER_BIAS.fullmatch(key) else "BF16"
                if dtype != expected_dtype:
                    raise RuntimeError(f"{manifest_path}: tensor {key!r} must be {expected_dtype}, got {dtype}.")
                if _ROUTED_EXPERT.search(key) or key.endswith(".weight_scale_inv") or key.startswith(mtp_prefix):
                    raise RuntimeError(f"{manifest_path}: cache contains excluded tensor {key!r}.")
                elements = 1
                for dimension in tensor_slice.get_shape():
                    elements *= dimension
                observed_bytes += elements * (4 if dtype == "F32" else 2)
                observed_dtypes[dtype] = observed_dtypes.get(dtype, 0) + 1
                observed_keys.add(key)

    if observed_keys != set(weight_map):
        raise RuntimeError(f"{manifest_path}: cache shard keys do not match its index.")
    if observed_bytes != tensor_bytes or observed_dtypes != dtype_counts:
        raise RuntimeError(f"{manifest_path}: observed tensor metadata does not match the manifest.")

    routed_manifest_path, routed_manifest = _validate_routed_manifest(weight_path, source_config, runtime_config)
    return KTNonExpertCache(
        path=str(root),
        manifest=manifest,
        checkpoint_files=tuple(sorted(checkpoint_files)),
        weight_keys=frozenset(weight_map),
        source_config=source_config,
        routed_manifest_path=routed_manifest_path,
        routed_manifest=routed_manifest,
    )


def validate_kt_non_expert_cache(
    cache_path: str | os.PathLike[str],
    source_model_name_or_path: str | os.PathLike[str],
    weight_path: str | os.PathLike[str],
    runtime_config: dict[str, Any],
) -> KTNonExpertCache:
    """Validate the existing BF16 cache once per shared-filesystem distributed job."""
    distributed = dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1
    rank = dist.get_rank() if distributed else 0
    cache = None
    error = None
    try:
        cache = _validate_cache_local(
            cache_path,
            source_model_name_or_path,
            weight_path,
            runtime_config,
            verify_shard_hashes=rank == 0,
        )
    except BaseException as exc:
        error = f"rank {rank}: {type(exc).__name__}: {exc}"

    if distributed:
        errors: list[str | None] = [None] * dist.get_world_size()
        dist.all_gather_object(errors, error)
        error = next((item for item in errors if item is not None), None)
    if error is not None:
        raise RuntimeError(f"KT INT8 artifact validation failed: {error}")
    assert cache is not None
    logger.info_rank0(f"Validated KT non-expert cache {cache.path} ({cache.fingerprint}).")
    return cache


def prepare_kt_int8_cache_loading(
    config: PretrainedConfig,
    model_args: ModelArguments,
    init_kwargs: dict[str, Any],
) -> KTNonExpertCache | None:
    """Switch only the checkpoint weights to the validated cache, retaining the source architecture config."""
    if not is_kt_int8_cache_requested(model_args):
        return None
    if init_kwargs.get("quantization_config") is not None:
        raise RuntimeError("KT INT8 non-expert cache loading cannot use an explicit quantization config.")

    runtime_config = getattr(model_args, "_kt_resolved_config", {})
    cache = validate_kt_non_expert_cache(
        model_args.kt_non_expert_weight_path,
        model_args.model_name_or_path,
        model_args.kt_weight_path,
        runtime_config,
    )
    if hasattr(config, "quantization_config"):
        delattr(config, "quantization_config")
    config.name_or_path = model_args.model_name_or_path
    generation_kwargs = {
        name: init_kwargs[name] for name in ("cache_dir", "revision", "token") if init_kwargs.get(name) is not None
    }
    try:
        generation_config = GenerationConfig.from_pretrained(
            model_args.model_name_or_path,
            local_files_only=True,
            **generation_kwargs,
        )
    except OSError:
        generation_config = GenerationConfig.from_model_config(config)
    init_kwargs.update(
        {
            "config": config,
            "generation_config": generation_config,
            "local_files_only": True,
            "output_loading_info": True,
            "pretrained_model_name_or_path": cache.path,
            "use_safetensors": True,
        }
    )
    return cache


def _canonical_loaded_key(key: str) -> str | None:
    if "._original_router.weight" in key:
        return None
    if "._original_router.e_score_correction_bias" in key:
        return key.replace("._original_router.e_score_correction_bias", ".gate.e_score_correction_bias")
    return key


def validate_kt_int8_loaded_model(
    model: PreTrainedModel,
    loading_info: dict[str, Any],
    cache: KTNonExpertCache,
    source_model_name_or_path: str,
) -> None:
    """Fail closed if frozen Transformers loaded anything other than the validated non-expert contract."""
    if not isinstance(loading_info, dict):
        raise RuntimeError("Frozen Transformers did not return the requested KT loading diagnostics.")
    missing_keys = set(loading_info.get("missing_keys") or ())
    invalid_missing = sorted(key for key in missing_keys if not _ROUTED_EXPERT.search(key))
    failures = {}
    if invalid_missing:
        failures["missing_keys"] = invalid_missing
    for field in ("unexpected_keys", "mismatched_keys", "error_msgs"):
        value = loading_info.get(field)
        if value:
            failures[field] = sorted(value) if field != "error_msgs" else list(value)
    if failures:
        raise RuntimeError(f"KT INT8 cache did not exactly load the non-expert model: {failures}.")

    loaded_keys = set()
    for key in model.state_dict():
        canonical = _canonical_loaded_key(key)
        if canonical is not None and not _ROUTED_EXPERT.search(canonical):
            loaded_keys.add(canonical)
    if loaded_keys != cache.weight_keys:
        missing = sorted(cache.weight_keys - loaded_keys)
        unexpected = sorted(loaded_keys - cache.weight_keys)
        raise RuntimeError(
            "KT INT8 cache keys do not match the instantiated non-expert architecture: "
            f"missing={missing[:16]}, unexpected={unexpected[:16]}."
        )

    model.config.name_or_path = source_model_name_or_path
    model._kt_base_model_name_or_path = source_model_name_or_path
    model._kt_non_expert_cache_path = cache.path
    model._kt_non_expert_cache_manifest = cache.manifest
    model._kt_routed_int8_manifest_path = cache.routed_manifest_path
    model._kt_routed_int8_manifest = cache.routed_manifest
