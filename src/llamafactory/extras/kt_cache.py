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
import shutil
import sys
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import torch
import yaml
from safetensors import safe_open
from safetensors.torch import save_file


_CACHE_MANIFEST = "kt_non_expert_manifest.json"
_CACHE_INDEX = "model.safetensors.index.json"
_CONVERTER_VERSION = 1
_DEFAULT_SHARD_BYTES = 4 * 1024**3
_MIN_DISK_HEADROOM_BYTES = 10 * 1024**3
_MTP_LAYER_PREFIX = "model.layers.61."
_FP32_ROUTER_BIAS = re.compile(r"^model\.layers\.\d+\.mlp\.gate\.e_score_correction_bias$")
_FP8_DTYPES = {
    dtype
    for dtype in (
        getattr(torch, "float8_e4m3fn", None),
        getattr(torch, "float8_e4m3fnuz", None),
        getattr(torch, "float8_e5m2", None),
        getattr(torch, "float8_e5m2fnuz", None),
    )
    if dtype is not None
}
_SAFETENSORS_FLOAT_DTYPES = {
    "F8_E4M3",
    "F8_E5M2",
    "F8_E4M3FNUZ",
    "F8_E5M2FNUZ",
    "F16",
    "BF16",
    "F32",
    "F64",
}


def _sha256_file(path: Path, chunk_bytes: int = 8 * 1024**2) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(chunk_bytes):
            digest.update(chunk)

    return digest.hexdigest()


def _fsync_path(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _is_routed_expert_key(key: str) -> bool:
    return ".mlp.experts." in key or ".block_sparse_moe.experts." in key


def _should_cache_key(key: str) -> bool:
    return not _is_routed_expert_key(key) and not key.startswith(_MTP_LAYER_PREFIX)


def _preserve_fp32(key: str) -> bool:
    return _FP32_ROUTER_BIAS.fullmatch(key) is not None


def _scale_key(weight_key: str) -> str:
    return f"{weight_key.removesuffix('.weight')}.weight_scale_inv"


def _weight_key(scale_key: str) -> str:
    return f"{scale_key.removesuffix('.weight_scale_inv')}.weight"


def _load_source_index(model_path: Path) -> tuple[dict[str, str], Path]:
    index_path = model_path / _CACHE_INDEX
    if index_path.is_file():
        with index_path.open(encoding="utf-8") as file:
            index = json.load(file)

        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise RuntimeError(f"{index_path} does not contain a non-empty `weight_map`.")

        return {str(key): str(value) for key, value in weight_map.items()}, index_path

    shard_paths = sorted(model_path.glob("*.safetensors"))
    if len(shard_paths) != 1:
        raise RuntimeError(
            f"{model_path} must contain {_CACHE_INDEX} or exactly one safetensors file; "
            f"found {len(shard_paths)} files."
        )

    with safe_open(shard_paths[0], framework="pt", device="cpu") as file:
        return dict.fromkeys(file.keys(), shard_paths[0].name), shard_paths[0]


def _source_fingerprint(model_path: Path, index_path: Path, weight_map: dict[str, str]) -> dict[str, str]:
    config_path = model_path / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing source model config: {config_path}")

    digest = hashlib.sha256()
    digest.update(f"kt-non-expert-cache-v{_CONVERTER_VERSION}\0".encode())
    for path in (config_path, index_path):
        digest.update(path.name.encode())
        digest.update(b"\0")
        digest.update(bytes.fromhex(_sha256_file(path)))

    for shard_name in sorted(set(weight_map.values())):
        shard_path = model_path / shard_name
        stat = shard_path.stat()
        digest.update(shard_name.encode())
        digest.update(b"\0")
        digest.update(str(stat.st_size).encode())
        digest.update(b"\0")

    return {
        "model_name_or_path": str(model_path.resolve()),
        "fingerprint": digest.hexdigest(),
        "config_sha256": _sha256_file(config_path),
        "index_sha256": _sha256_file(index_path),
    }


def _cache_fingerprint(
    source: dict[str, str],
    file_records: list[dict[str, Any]],
    tensor_count: int,
    tensor_bytes: int,
    dtype_counts: dict[str, int],
) -> str:
    digest = hashlib.sha256()
    digest.update(f"kt-non-expert-cache-v{_CONVERTER_VERSION}\0".encode())
    digest.update(source["fingerprint"].encode())
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


def _load_tensor(model_path: Path, weight_map: dict[str, str], key: str) -> torch.Tensor:
    try:
        shard_path = model_path / weight_map[key]
    except KeyError as error:
        raise KeyError(f"Tensor {key!r} is absent from the source weight index.") from error

    with safe_open(shard_path, framework="pt", device="cpu") as file:
        return file.get_tensor(key)


def _estimate_output_bytes(model_path: Path, weight_map: dict[str, str]) -> int:
    keys_by_shard: dict[str, list[str]] = {}
    for key, shard_name in weight_map.items():
        if _should_cache_key(key) and not key.endswith(".weight_scale_inv"):
            keys_by_shard.setdefault(shard_name, []).append(key)

    total_bytes = 0
    for shard_name, keys in keys_by_shard.items():
        with safe_open(model_path / shard_name, framework="pt", device="cpu") as file:
            for key in keys:
                tensor_slice = file.get_slice(key)
                element_count = 1
                for dimension in tensor_slice.get_shape():
                    element_count *= int(dimension)

                dtype = tensor_slice.get_dtype()
                if dtype not in _SAFETENSORS_FLOAT_DTYPES:
                    raise RuntimeError(
                        f"KT non-expert cache supports floating-point tensors only; "
                        f"tensor {key} has source dtype {dtype!r}."
                    )

                output_element_size = torch.float32.itemsize if _preserve_fp32(key) else torch.bfloat16.itemsize
                total_bytes += element_count * output_element_size

    return total_bytes


def _dequantize_fp8(
    quantized: torch.Tensor,
    scales: torch.Tensor,
    block_size: tuple[int, int],
    *,
    output_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    if quantized.ndim != 2 or scales.ndim != 2:
        raise ValueError(
            f"KT non-expert FP8 conversion expects 2-D weight and scale tensors, got "
            f"{tuple(quantized.shape)} and {tuple(scales.shape)}."
        )

    rows, cols = quantized.shape
    block_m, block_n = block_size
    expected_scale_shape = ((rows + block_m - 1) // block_m, (cols + block_n - 1) // block_n)
    if tuple(scales.shape) != expected_scale_shape:
        raise ValueError(
            f"FP8 scale shape {tuple(scales.shape)} does not match expected partial-block shape "
            f"{expected_scale_shape} for weight {tuple(quantized.shape)} and block size {block_size}."
        )

    output = torch.empty((rows, cols), dtype=output_dtype)
    block_rows_per_chunk = max(1, (256 * 1024**2) // max(cols * scales.element_size() * block_m, 1))
    for scale_row_begin in range(0, scales.shape[0], block_rows_per_chunk):
        scale_row_end = min(scale_row_begin + block_rows_per_chunk, scales.shape[0])
        row_begin = scale_row_begin * block_m
        row_end = min(scale_row_end * block_m, rows)
        expanded_scales = scales[scale_row_begin:scale_row_end].repeat_interleave(block_m, dim=0)
        expanded_scales = expanded_scales.repeat_interleave(block_n, dim=1)[: row_end - row_begin, :cols]
        output[row_begin:row_end].copy_(
            (quantized[row_begin:row_end].to(scales.dtype) * expanded_scales).to(output_dtype)
        )

    return output


def _iter_cached_tensors(
    model_path: Path,
    weight_map: dict[str, str],
    block_size: tuple[int, int],
) -> Iterator[tuple[str, torch.Tensor]]:
    for key in sorted(weight_map):
        if not _should_cache_key(key) or key.endswith(".weight_scale_inv"):
            continue

        tensor = _load_tensor(model_path, weight_map, key)
        if _preserve_fp32(key):
            if tensor.dtype != torch.float32:
                raise RuntimeError(
                    f"DeepSeek router correction bias {key} must be FP32 in the source checkpoint, got {tensor.dtype}."
                )
            tensor = tensor.contiguous()
        elif tensor.dtype in _FP8_DTYPES:
            if not key.endswith(".weight"):
                raise RuntimeError(f"Unsupported FP8 non-weight tensor in source checkpoint: {key}")

            scale_key = _scale_key(key)
            if scale_key not in weight_map:
                raise RuntimeError(f"FP8 tensor {key} is missing its scale tensor {scale_key}.")

            tensor = _dequantize_fp8(tensor, _load_tensor(model_path, weight_map, scale_key), block_size)
        elif tensor.is_floating_point():
            tensor = tensor.to(torch.bfloat16).contiguous()
        else:
            raise RuntimeError(
                f"KT non-expert cache supports floating-point tensors only; tensor {key} has dtype {tensor.dtype}."
            )

        yield key, tensor

    orphan_scales = [
        key
        for key in weight_map
        if key.endswith(".weight_scale_inv") and _should_cache_key(key) and _weight_key(key) not in weight_map
    ]
    if orphan_scales:
        raise RuntimeError(f"Found FP8 scale tensors without matching weights: {orphan_scales[:8]}")


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True)
        file.write("\n")
        file.flush()
        os.fsync(file.fileno())

    os.replace(temporary_path, path)


def _publish_cache(
    model_path: Path,
    output_path: Path,
    weight_map: dict[str, str],
    source: dict[str, str],
    block_size: tuple[int, int],
    shard_bytes: int,
) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    staging_path = Path(tempfile.mkdtemp(prefix=f".{output_path.name}.tmp-", dir=output_path.parent))
    shard_tensors: dict[str, torch.Tensor] = {}
    shard_tensor_names: list[list[str]] = []
    tensor_count = 0
    tensor_bytes = 0
    dtype_counts: dict[str, int] = {}

    def flush_shard() -> None:
        nonlocal shard_tensors
        if not shard_tensors:
            return

        shard_index = len(shard_tensor_names) + 1
        temporary_name = f"model-part-{shard_index:05d}.safetensors"
        save_file(shard_tensors, staging_path / temporary_name)
        shard_tensor_names.append(list(shard_tensors))
        shard_tensors = {}

    try:
        current_bytes = 0
        for key, tensor in _iter_cached_tensors(model_path, weight_map, block_size):
            size_bytes = tensor.numel() * tensor.element_size()
            if shard_tensors and current_bytes + size_bytes > shard_bytes:
                flush_shard()
                current_bytes = 0

            shard_tensors[key] = tensor
            current_bytes += size_bytes
            tensor_count += 1
            tensor_bytes += size_bytes
            dtype_name = {
                torch.bfloat16: "BF16",
                torch.float32: "F32",
            }.get(tensor.dtype)
            if dtype_name is None:
                raise RuntimeError(f"Unsupported KT cache output dtype {tensor.dtype} for tensor {key}.")
            dtype_counts[dtype_name] = dtype_counts.get(dtype_name, 0) + 1

        flush_shard()
        if not shard_tensor_names:
            raise RuntimeError("No non-expert tensors were selected for the KT cache.")

        final_weight_map: dict[str, str] = {}
        file_records: list[dict[str, Any]] = []
        shard_count = len(shard_tensor_names)
        for index, tensor_names in enumerate(shard_tensor_names, start=1):
            temporary_path = staging_path / f"model-part-{index:05d}.safetensors"
            final_name = f"model-{index:05d}-of-{shard_count:05d}.safetensors"
            final_path = staging_path / final_name
            os.replace(temporary_path, final_path)
            for tensor_name in tensor_names:
                final_weight_map[tensor_name] = final_name

            file_records.append(
                {
                    "name": final_name,
                    "size": final_path.stat().st_size,
                    "sha256": _sha256_file(final_path),
                }
            )
            _fsync_path(final_path)

        _write_json_atomic(
            staging_path / _CACHE_INDEX,
            {
                "metadata": {"total_size": tensor_bytes},
                "weight_map": final_weight_map,
            },
        )
        index_path = staging_path / _CACHE_INDEX
        file_records.append(
            {
                "name": _CACHE_INDEX,
                "size": index_path.stat().st_size,
                "sha256": _sha256_file(index_path),
            }
        )
        manifest = {
            "version": 1,
            "status": "ready",
            "fingerprint": _cache_fingerprint(
                source,
                file_records,
                tensor_count,
                tensor_bytes,
                dtype_counts,
            ),
            "converter": {
                "name": "llamafactory.prepare-kt-cache",
                "version": _CONVERTER_VERSION,
                "default_dtype": "BF16",
                "fp32_exceptions": ["model.layers.*.mlp.gate.e_score_correction_bias"],
                "weight_block_size": list(block_size),
            },
            "source": source,
            "files": file_records,
            "tensors": {
                "count": tensor_count,
                "bytes": tensor_bytes,
                "dtypes": dict(sorted(dtype_counts.items())),
            },
        }
        _write_json_atomic(staging_path / _CACHE_MANIFEST, manifest)
        _fsync_path(staging_path / _CACHE_MANIFEST)
        _fsync_path(staging_path)

        if output_path.exists():
            raise FileExistsError(f"Refusing to replace existing KT cache directory: {output_path}")

        os.replace(staging_path, output_path)
        _fsync_path(output_path.parent)
        return manifest
    except BaseException:
        shutil.rmtree(staging_path, ignore_errors=True)
        raise


def prepare_kt_cache(config_path: str, *, shard_bytes: int = _DEFAULT_SHARD_BYTES) -> dict[str, Any]:
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise RuntimeError("KT non-expert cache preparation must run as one offline process, not under torchrun.")

    with Path(config_path).absolute().open(encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise TypeError(f"Training configuration {config_path} must contain a mapping.")

    if config.get("kt_expert_weight_format") != "int8":
        raise ValueError("`prepare-kt-cache` requires `kt_expert_weight_format: int8`.")
    if config.get("trust_remote_code") is not False:
        raise ValueError("DeepSeek INT8 cache preparation requires `trust_remote_code: false`.")

    model_path = Path(str(config.get("model_name_or_path", ""))).expanduser()
    output_path = Path(str(config.get("kt_non_expert_weight_path", ""))).expanduser()
    if not model_path.is_dir():
        raise FileNotFoundError(f"Source model directory does not exist: {model_path}")
    if not str(config.get("kt_non_expert_weight_path", "")).strip():
        raise ValueError("Training configuration must set `kt_non_expert_weight_path`.")
    if output_path.exists():
        raise FileExistsError(f"KT cache destination already exists: {output_path}")

    with (model_path / "config.json").open(encoding="utf-8") as file:
        model_config = json.load(file)

    if model_config.get("model_type") != "deepseek_v3":
        raise ValueError(f"Expected a DeepSeek-V3 checkpoint, got model_type={model_config.get('model_type')!r}.")

    quantization_config = model_config.get("quantization_config") or {}
    if quantization_config.get("quant_method") != "fp8":
        raise ValueError(
            "DeepSeek INT8 cache preparation requires an FP8 source checkpoint "
            "with `quantization_config.quant_method: fp8`."
        )
    block_size_value = quantization_config.get("weight_block_size")
    if not isinstance(block_size_value, (list, tuple)) or len(block_size_value) != 2:
        raise ValueError("Source config must provide a two-element `quantization_config.weight_block_size`.")

    block_size = (int(block_size_value[0]), int(block_size_value[1]))
    if min(block_size) <= 0:
        raise ValueError(f"Invalid FP8 weight block size: {block_size}")

    weight_map, index_path = _load_source_index(model_path)
    source = _source_fingerprint(model_path, index_path, weight_map)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    estimated_output_bytes = _estimate_output_bytes(model_path, weight_map)
    free_bytes = shutil.disk_usage(output_path.parent).free
    required_bytes = estimated_output_bytes + _MIN_DISK_HEADROOM_BYTES
    if free_bytes < required_bytes:
        raise RuntimeError(
            f"Insufficient free space to prepare KT cache: {free_bytes / 1024**3:.1f} GiB available, "
            f"{required_bytes / 1024**3:.1f} GiB required."
        )

    lock_path = output_path.parent / f".{output_path.name}.lock"
    try:
        lock_descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as error:
        raise RuntimeError(f"Another KT cache preparation owns {lock_path}.") from error

    try:
        os.write(lock_descriptor, f"{os.getpid()}\n".encode())
        os.fsync(lock_descriptor)
        return _publish_cache(model_path, output_path, weight_map, source, block_size, shard_bytes)
    finally:
        os.close(lock_descriptor)
        lock_path.unlink(missing_ok=True)


def prepare_kt_cache_cli(arguments: list[str] | None = None) -> None:
    arguments = list(sys.argv[1:] if arguments is None else arguments)
    if len(arguments) != 1 or arguments[0] in {"-h", "--help"}:
        print("Usage: llamafactory-cli prepare-kt-cache TRAINING_CONFIG.yaml")
        if len(arguments) == 1 and arguments[0] in {"-h", "--help"}:
            return

        raise SystemExit(2)

    manifest = prepare_kt_cache(arguments[0])
    print(json.dumps(manifest, indent=2, sort_keys=True))
