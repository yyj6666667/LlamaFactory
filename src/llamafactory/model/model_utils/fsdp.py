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
import json
import logging
import re
import threading
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from functools import wraps
from typing import TYPE_CHECKING

import torch


if TYPE_CHECKING:
    from ...hparams import ModelArguments


_FSDP2_PATCH_LOCK = threading.RLock()
_SUPPORTED_ACCELERATE_MIN = (1, 14, 0)
_SUPPORTED_ACCELERATE_MAX = (1, 15, 0)
_DEEPSEEK_V31_DECODER_COUNT = 61


@contextmanager
def _preserve_parameter_identity_on_conversion() -> Iterator[None]:
    get_swap = getattr(torch.__future__, "get_swap_module_params_on_conversion", None)
    set_swap = getattr(torch.__future__, "set_swap_module_params_on_conversion", None)
    if get_swap is None or set_swap is None:
        yield
        return

    previous = get_swap()
    set_swap(True)
    try:
        yield
    finally:
        set_swap(previous)


def _parse_release(version: str) -> tuple[int, int, int]:
    match = re.match(r"^(\d+)\.(\d+)\.(\d+)", version)
    if match is None:
        raise RuntimeError(f"Cannot parse Accelerate version {version!r}.")

    return tuple(int(part) for part in match.groups())


def _require_signature(function, name: str, expected_names: tuple[str, ...]) -> None:
    try:
        parameters = tuple(inspect.signature(function).parameters)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"Cannot inspect Accelerate {name} signature.") from exc

    if parameters != expected_names:
        raise RuntimeError(f"Unsupported Accelerate {name} signature: expected {expected_names}, got {parameters}.")


def _require_source_contract(function, name: str, markers: tuple[str, ...]) -> None:
    try:
        source = inspect.getsource(function)
    except (OSError, TypeError) as exc:
        raise RuntimeError(f"Cannot inspect Accelerate {name} implementation.") from exc

    missing = [marker for marker in markers if marker not in source]
    if missing:
        raise RuntimeError(
            f"Unsupported Accelerate {name} implementation; missing KT compatibility markers: {missing}."
        )


def _validate_accelerate_kt_fsdp2(accelerate_module, fsdp_utils, prepare_model) -> None:
    version = getattr(accelerate_module, "__version__", None)
    if not isinstance(version, str):
        raise RuntimeError("Accelerate does not expose a valid __version__.")

    release = _parse_release(version)
    if not (_SUPPORTED_ACCELERATE_MIN <= release < _SUPPORTED_ACCELERATE_MAX):
        raise RuntimeError(
            f"KT INT8 FSDP2 requires the Accelerate KT 1.14.x integration; found accelerate=={version}."
        )

    _require_signature(prepare_model, "fsdp2_prepare_model", ("accelerator", "model"))
    _require_source_contract(
        prepare_model,
        "fsdp2_prepare_model",
        (
            "from torch.distributed.fsdp import FSDPModule, MixedPrecisionPolicy, fully_shard",
            "fsdp2_plugin.set_auto_wrap_policy(model)",
            "auto_wrap_policy_func = fsdp2_prepare_auto_wrap_policy(fsdp2_plugin, model)",
            "fully_shard(module, **fsdp2_kwargs)",
            "fully_shard(model, **fsdp2_kwargs)",
        ),
    )

    load_full_state_dict = getattr(fsdp_utils, "fsdp2_load_full_state_dict", None)
    switch_optimizer_parameters = getattr(fsdp_utils, "fsdp2_switch_optimizer_parameters", None)
    if load_full_state_dict is None or switch_optimizer_parameters is None:
        raise RuntimeError("Accelerate KT FSDP2 state loading helpers are unavailable.")

    _require_signature(
        load_full_state_dict,
        "fsdp2_load_full_state_dict",
        ("accelerator", "model", "full_sd", "cpu_offload"),
    )
    _require_source_contract(
        load_full_state_dict,
        "fsdp2_load_full_state_dict",
        (
            "def _is_dtensor(param):",
            "if not _is_dtensor(sharded_param):",
            "model.load_state_dict(sharded_sd, assign=True)",
        ),
    )
    _require_signature(
        switch_optimizer_parameters,
        "fsdp2_switch_optimizer_parameters",
        ("optimizer", "mapping"),
    )
    _require_source_contract(
        switch_optimizer_parameters,
        "fsdp2_switch_optimizer_parameters",
        (
            "if ptr in mapping:",
            "new_params.append(p)",
        ),
    )


def _collect_kt_fsdp_ignored_params(model: torch.nn.Module) -> set[torch.nn.Parameter]:
    ignored_params: set[torch.nn.Parameter] = set()
    wrapper_count = 0

    for module in model.modules():
        if not getattr(module, "_is_kt_moe_wrapper", False):
            continue

        wrapper_count += 1
        experts_attr = getattr(module, "_experts_attr", None)
        if not isinstance(experts_attr, str) or not experts_attr:
            raise RuntimeError("KT MoE wrapper does not declare a valid _experts_attr.")

        experts = getattr(module, experts_attr, None)
        if not isinstance(experts, torch.nn.Module):
            raise RuntimeError(f"KT MoE wrapper expert subtree {experts_attr!r} is not an nn.Module.")

        ignored_params.update(experts.parameters(recurse=True))

    if wrapper_count == 0:
        raise RuntimeError("KT INT8 FSDP2 preparation found no explicit KT MoE wrappers.")
    if not ignored_params:
        raise RuntimeError("KT INT8 FSDP2 preparation found no expert parameters to ignore.")

    zero_storage_params = {param for param in model.parameters() if getattr(param, "_kt_zero_storage", False)}
    missing_placeholders = zero_storage_params - ignored_params
    if missing_placeholders:
        raise RuntimeError("Found KT zero-storage expert placeholders outside the explicit KT expert subtrees.")

    return ignored_params


def _collect_configured_ignored_params(accelerator, model: torch.nn.Module) -> set[torch.nn.Parameter]:
    fsdp_plugin = accelerator.state.fsdp_plugin
    ignored_modules = fsdp_plugin.ignored_modules
    if ignored_modules is None:
        return set()

    if isinstance(ignored_modules, str):
        try:
            pattern = re.compile(ignored_modules)
        except re.error as exc:
            raise RuntimeError(f"Invalid FSDP ignored_modules regex {ignored_modules!r}.") from exc

        modules = [module for name, module in model.named_modules() if pattern.fullmatch(name)]
    else:
        try:
            modules = list(ignored_modules)
        except TypeError as exc:
            raise RuntimeError("FSDP ignored_modules must be a regex or an iterable of nn.Module.") from exc

    invalid_modules = [type(module).__name__ for module in modules if not isinstance(module, torch.nn.Module)]
    if invalid_modules:
        raise RuntimeError(f"FSDP ignored_modules contains non-module entries: {invalid_modules}.")

    return {param for module in modules for param in module.parameters(recurse=True)}


def _validate_shardable_frozen_params(
    model: torch.nn.Module,
    ignored_params: set[torch.nn.Parameter],
) -> None:
    incompatible = []
    for name, param in model.named_parameters():
        if param in ignored_params or param.requires_grad:
            continue
        if param.__class__.__name__ == "Params4bit" or not (param.is_floating_point() or param.is_complex()):
            incompatible.append(name)

    if incompatible:
        preview = incompatible[:8]
        raise RuntimeError(
            "KT INT8 selective FSDP2 cannot shard non-floating frozen parameters outside "
            f"the KT expert subtrees: {preview}" + ("..." if len(incompatible) > len(preview) else "")
        )


def _collect_native_deepseek_decoder_layers(
    model: torch.nn.Module,
) -> list[tuple[str, torch.nn.Module]] | None:
    if getattr(getattr(model, "config", None), "model_type", None) != "deepseek_v3":
        return None

    layers = [
        (name, module)
        for name, module in model.named_modules()
        if type(module).__name__ == "DeepseekV3DecoderLayer"
        and type(module).__module__.startswith("transformers.models.deepseek_v3.")
    ]
    if len(layers) != _DEEPSEEK_V31_DECODER_COUNT:
        raise RuntimeError(
            "KT DeepSeek-V3.1 FSDP2 requires exactly "
            f"{_DEEPSEEK_V31_DECODER_COUNT} native decoder layers, got {len(layers)}."
        )

    return layers


def _validate_deepseek_fsdp2_parameter_ownership(
    model: torch.nn.Module,
    decoder_layers: list[tuple[str, torch.nn.Module]],
    explicit_kt_params: set[torch.nn.Parameter],
    configured_params: set[torch.nn.Parameter],
) -> dict[int, str]:
    unexpected_configured_params = configured_params - explicit_kt_params
    if unexpected_configured_params:
        unexpected_names = sorted(
            name
            for name, parameter in model.named_parameters(remove_duplicate=False)
            if parameter in unexpected_configured_params
        )
        raise RuntimeError(
            "KT DeepSeek-V3.1 FSDP2 only permits explicit KT expert parameters to be ignored; "
            f"configured non-KT parameters: {unexpected_names}."
        )

    ignored_ids = {id(param) for param in explicit_kt_params}
    owners: dict[int, set[str]] = defaultdict(set)
    for layer_name, layer in decoder_layers:
        for parameter in layer.parameters(recurse=True):
            if id(parameter) not in ignored_ids:
                owners[id(parameter)].add(layer_name)

    duplicate_owners = {
        parameter_id: sorted(parameter_owners)
        for parameter_id, parameter_owners in owners.items()
        if len(parameter_owners) != 1
    }
    if duplicate_owners:
        raise RuntimeError(f"DeepSeek decoder parameters belong to multiple FSDP units: {duplicate_owners}.")

    ownership = {parameter_id: next(iter(parameter_owners)) for parameter_id, parameter_owners in owners.items()}
    for parameter in model.parameters():
        parameter_id = id(parameter)
        if parameter_id not in ignored_ids and parameter_id not in ownership:
            ownership[parameter_id] = "<root>"

    expected_ids = {id(parameter) for parameter in model.parameters()} - ignored_ids
    if set(ownership) != expected_ids:
        raise RuntimeError("Every non-KT DeepSeek parameter must belong to exactly one FSDP unit.")

    return ownership


def _frozen_bf16_parameter_bytes(
    model: torch.nn.Module,
    ignored_params: set[torch.nn.Parameter],
    *,
    local_dtensors: bool,
) -> int:
    ignored_ids = {id(parameter) for parameter in ignored_params}
    seen_ids = set()
    total_bytes = 0
    for parameter in model.parameters():
        parameter_id = id(parameter)
        if parameter_id in seen_ids or parameter_id in ignored_ids or parameter.requires_grad:
            continue
        seen_ids.add(parameter_id)
        if parameter.dtype != torch.bfloat16:
            raise RuntimeError(
                f"KT DeepSeek-V3.1 FSDP2 requires every frozen non-expert parameter to be BF16; got {parameter.dtype}."
            )
        tensor = parameter.to_local() if local_dtensors else parameter
        total_bytes += tensor.numel() * tensor.element_size()

    return total_bytes


def _validate_prepared_deepseek_fsdp2(
    prepared_model: torch.nn.Module,
    decoder_layers: list[tuple[str, torch.nn.Module]],
    explicit_kt_params: set[torch.nn.Parameter],
    call_records: list[tuple[int, str, object]],
    expected_global_frozen_bf16_bytes: int,
) -> None:
    from torch.distributed.fsdp import FSDPModule
    from torch.distributed.tensor import DTensor, Shard

    expected_modules = {id(prepared_model): "<root>"}
    expected_modules.update({id(module): name for name, module in decoder_layers})
    actual_modules = [module_id for module_id, _, _ in call_records]
    if len(actual_modules) != len(expected_modules) or set(actual_modules) != set(expected_modules):
        actual_names = [name for _, name, _ in call_records]
        raise RuntimeError(
            "KT DeepSeek-V3.1 FSDP2 requires 61 decoder units plus one root unit; "
            f"got {len(actual_modules)} calls: {actual_names}."
        )
    if len(actual_modules) != len(set(actual_modules)):
        raise RuntimeError("KT DeepSeek-V3.1 FSDP2 attempted to shard an FSDP unit more than once.")

    invalid_reshard = [name for _, name, reshard in call_records if reshard is not True]
    if invalid_reshard:
        raise RuntimeError(f"Every DeepSeek FSDP2 unit must use `reshard_after_forward: true`: {invalid_reshard}.")

    if not isinstance(prepared_model, FSDPModule):
        raise RuntimeError("The DeepSeek root module was not converted to an FSDPModule.")
    non_fsdp_layers = [name for name, module in decoder_layers if not isinstance(module, FSDPModule)]
    if non_fsdp_layers:
        raise RuntimeError(f"DeepSeek decoder layers were not converted to FSDPModule: {non_fsdp_layers}.")

    ignored_ids = {id(parameter) for parameter in explicit_kt_params}
    prepared_ids = {id(parameter) for parameter in prepared_model.parameters()}
    if not ignored_ids.issubset(prepared_ids):
        raise RuntimeError("FSDP2 replaced explicitly ignored KT expert Parameter objects.")

    invalid_shards = []
    for name, parameter in prepared_model.named_parameters(remove_duplicate=False):
        parameter_id = id(parameter)
        if parameter_id in ignored_ids:
            if isinstance(parameter, DTensor):
                invalid_shards.append(f"{name}: KT-owned parameter became a DTensor")
            continue

        placements = tuple(parameter.placements) if isinstance(parameter, DTensor) else ()
        if (
            not isinstance(parameter, DTensor)
            or len(placements) != 1
            or not isinstance(placements[0], Shard)
            or placements[0].dim != 0
        ):
            invalid_shards.append(f"{name}: expected DTensor Shard(0), got {placements or type(parameter).__name__}")
            continue

    if invalid_shards:
        raise RuntimeError("KT DeepSeek-V3.1 FSDP2 parameter sharding contract failed: " + "; ".join(invalid_shards))

    observed_global_frozen_bf16_bytes = _frozen_bf16_parameter_bytes(
        prepared_model,
        explicit_kt_params,
        local_dtensors=False,
    )
    if observed_global_frozen_bf16_bytes != expected_global_frozen_bf16_bytes:
        raise RuntimeError(
            "DeepSeek-V3.1 frozen BF16 global inventory changed during FSDP2 preparation: "
            f"expected {expected_global_frozen_bf16_bytes} bytes, "
            f"got {observed_global_frozen_bf16_bytes}."
        )
    local_frozen_bf16_bytes = _frozen_bf16_parameter_bytes(
        prepared_model,
        explicit_kt_params,
        local_dtensors=True,
    )
    logging.getLogger(__name__).info(
        "KT DeepSeek-V3.1 frozen BF16 inventory: global_bytes=%d, local_bytes=%d",
        observed_global_frozen_bf16_bytes,
        local_frozen_bf16_bytes,
    )


def _install_first_step_fsdp_telemetry(
    prepared_model: torch.nn.Module,
    decoder_layers: list[tuple[str, torch.nn.Module]],
) -> None:
    if not torch.cuda.is_available():
        return

    try:
        from kt_kernel.sft.dist_utils import _checkpoint_hook_mode
    except (ImportError, ModuleNotFoundError):
        _checkpoint_hook_mode = lambda: "unavailable"

    targets = [("<root>", prepared_model), *decoder_layers[:4]]
    target_layer_names = {name for name, _ in decoder_layers[:4]}
    handles: list[torch.utils.hooks.RemovableHandle] = []
    emitted: set[tuple[str, str, str]] = set()
    recompute_posts: set[str] = set()
    outer_forward_count = 0

    def remove_hooks() -> None:
        while handles:
            handles.pop().remove()

    def emit(name: str, module: torch.nn.Module, event: str) -> str:
        nonlocal outer_forward_count
        phase = _checkpoint_hook_mode()
        key = (name, phase, event)
        if key in emitted:
            return phase
        emitted.add(key)

        if name == "<root>" and event == "pre" and phase == "none":
            outer_forward_count += 1

        device = torch.cuda.current_device()
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        payload = {
            "event": event,
            "free_bytes": free_bytes,
            "module": name,
            "phase": phase,
            "rank": torch.distributed.get_rank() if torch.distributed.is_initialized() else 0,
            "reserved_bytes": torch.cuda.memory_reserved(device),
            "allocated_bytes": torch.cuda.memory_allocated(device),
            "total_bytes": total_bytes,
            "fsdp_module": any(type(base).__name__ == "FSDPModule" for base in type(module).__mro__),
        }
        logging.getLogger(__name__).info("KT_FSDP_FIRST_STEP %s", json.dumps(payload, sort_keys=True))
        return phase

    def make_pre_hook(name: str):
        def pre_hook(module, args):
            del args
            emit(name, module, "pre")
            if name == "<root>" and outer_forward_count > 1:
                remove_hooks()

        return pre_hook

    def make_post_hook(name: str):
        def post_hook(module, args, output):
            del args, output
            phase = emit(name, module, "post")
            if phase == "recompute" and name in target_layer_names:
                recompute_posts.add(name)
                if recompute_posts == target_layer_names:
                    remove_hooks()

        return post_hook

    for name, module in targets:
        handles.append(module.register_forward_pre_hook(make_pre_hook(name)))
        handles.append(module.register_forward_hook(make_post_hook(name)))


def _kt_fsdp2_cpu_shard(tensor: torch.Tensor, rank: int, world_size: int) -> torch.Tensor:
    chunk_size = (tensor.size(0) + world_size - 1) // world_size
    start = min(rank * chunk_size, tensor.size(0))
    length = min(chunk_size, tensor.size(0) - start)
    return tensor.narrow(0, start, length)


def _get_registered_state_tensor(model: torch.nn.Module, name: str) -> torch.Tensor:
    try:
        return model.get_parameter(name)
    except AttributeError:
        pass

    try:
        return model.get_buffer(name)
    except AttributeError as error:
        raise RuntimeError(f"{name}: state-dict entry is not a registered parameter or buffer.") from error


def _same_tensor_view(left: torch.Tensor, right: torch.Tensor) -> bool:
    if (
        left.device != right.device
        or left.dtype != right.dtype
        or tuple(left.shape) != tuple(right.shape)
        or tuple(left.stride()) != tuple(right.stride())
        or left.storage_offset() != right.storage_offset()
    ):
        return False

    if left.device.type == "meta":
        return True

    return left.untyped_storage().data_ptr() == right.untyped_storage().data_ptr()


def _install_loaded_state_tensor(target: torch.Tensor, loaded: torch.Tensor) -> None:
    """Install one loaded tensor while preserving the registered Tensor/Parameter identity."""
    from torch.distributed.tensor import DTensor

    if isinstance(target, DTensor) and isinstance(loaded, DTensor) and target.device.type != "meta":
        with torch.no_grad():
            target.to_local().copy_(loaded.to_local())
        return
    if (
        not isinstance(target, torch.nn.Parameter)
        and not isinstance(target, DTensor)
        and type(target) is type(loaded)
        and target.device.type != "meta"
    ):
        with torch.no_grad():
            target.copy_(loaded)
        return

    if isinstance(target, torch.nn.Parameter) and not isinstance(loaded, torch.nn.Parameter):
        loaded = torch.nn.Parameter(loaded, requires_grad=target.requires_grad)
    elif not isinstance(target, torch.nn.Parameter) and isinstance(loaded, torch.nn.Parameter):
        loaded = loaded.detach()

    if type(target) is not type(loaded) or isinstance(target, torch.nn.Parameter) != isinstance(
        loaded, torch.nn.Parameter
    ):
        raise RuntimeError(
            "Cannot install FSDP2 state tensor with a different Python tensor type: "
            f"target={type(target).__name__}, loaded={type(loaded).__name__}."
        )

    target_attributes = dict(getattr(target, "__dict__", {}))
    torch.utils.swap_tensors(target, loaded)
    target.__dict__.clear()
    target.__dict__.update(target_attributes)


def _kt_fsdp2_streaming_load_full_state_dict(
    accelerator,
    model: torch.nn.Module,
    full_sd: dict,
    cpu_offload: bool = False,
    *,
    rank_zero_only_names: frozenset[str] = frozenset(),
):
    """Load a KT FSDP2 state dict without ever materializing a full parameter on GPU."""
    import torch.distributed as dist
    from torch.distributed.tensor import DTensor, Shard

    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("KT INT8 streaming FSDP2 loading requires an initialized process group.")
    if not dist.is_gloo_available():
        raise RuntimeError("KT INT8 streaming FSDP2 loading requires the Gloo backend.")

    world_size = dist.get_world_size()
    rank = dist.get_rank()
    if world_size != 2:
        raise RuntimeError(f"KT INT8 streaming FSDP2 loading currently requires exactly two ranks, got {world_size}.")

    default_is_gloo = str(dist.get_backend()).lower() == "gloo"
    cpu_group = dist.group.WORLD if default_is_gloo else dist.new_group(backend="gloo")
    owns_cpu_group = cpu_group is not dist.group.WORLD

    try:
        sharded_state = model.state_dict()
        errors = []
        signature = []
        registered_tensors = {}
        aliases_by_id: dict[int, list[str]] = defaultdict(list)
        parameter_requires_grad = {
            name: parameter.requires_grad for name, parameter in model.named_parameters(remove_duplicate=False)
        }

        if cpu_offload:
            errors.append("CPU offload is not supported.")
        if bool(accelerator.is_main_process) != (rank == 0):
            errors.append("Accelerator main-process ownership must be global rank 0.")
        if torch.device(accelerator.device).type != "cpu" and torch.device(accelerator.device).index is None:
            errors.append("Accelerator device must identify the local accelerator device.")
        if rank == 0:
            missing = sorted(set(sharded_state) - set(full_sd))
            unexpected = sorted(set(full_sd) - set(sharded_state))
            if missing:
                errors.append(f"rank-0 full state dict is missing keys: {missing}.")
            if unexpected:
                errors.append(f"rank-0 full state dict has unexpected keys: {unexpected}.")

        for name, sharded_value in sharded_state.items():
            try:
                registered_tensor = _get_registered_state_tensor(model, name)
            except RuntimeError as error:
                errors.append(str(error))
            else:
                registered_tensors[name] = registered_tensor
                aliases_by_id[id(registered_tensor)].append(name)

            if not isinstance(sharded_value, torch.Tensor):
                errors.append(f"{name}: sharded model state value must be a Tensor.")
                continue

            if not isinstance(sharded_value, DTensor):
                if name not in rank_zero_only_names:
                    signature.append(
                        (
                            name,
                            tuple(sharded_value.shape),
                            tuple(sharded_value.stride()),
                            str(sharded_value.dtype),
                            "replicated",
                        )
                    )
                if rank == 0 and name in full_sd:
                    full_value = full_sd[name]
                    if not isinstance(full_value, torch.Tensor):
                        errors.append(f"{name}: rank-0 full state value must be a Tensor.")
                    elif full_value.device.type != "cpu":
                        errors.append(f"{name}: rank-0 full state value must remain on CPU, got {full_value.device}.")
                    elif full_value.layout != torch.strided:
                        errors.append(f"{name}: only dense strided rank-0 tensors are supported.")
                    elif full_value.dtype != sharded_value.dtype:
                        errors.append(
                            f"{name}: rank-0 dtype {full_value.dtype} does not match model dtype {sharded_value.dtype}."
                        )
                    elif tuple(full_value.shape) != tuple(sharded_value.shape):
                        errors.append(
                            f"{name}: rank-0 shape {tuple(full_value.shape)} does not match "
                            f"model shape {tuple(sharded_value.shape)}."
                        )
                    elif name not in rank_zero_only_names and not full_value.is_contiguous():
                        errors.append(f"{name}: rank-0 full state value must be contiguous.")
                continue

            mesh = sharded_value.device_mesh
            placements = tuple(sharded_value.placements)
            mesh_ranks = tuple(int(item) for item in mesh.mesh.reshape(-1).tolist())
            if mesh.ndim != 1 or mesh_ranks != tuple(range(world_size)):
                errors.append(f"{name}: expected a one-dimensional WORLD device mesh, got ranks {mesh_ranks}.")
            if len(placements) != 1 or not isinstance(placements[0], Shard) or placements[0].dim != 0:
                errors.append(f"{name}: only a single Shard(0) placement is supported, got {placements}.")
            if mesh.device_type != torch.device(accelerator.device).type:
                errors.append(
                    f"{name}: DTensor mesh device type {mesh.device_type!r} does not match "
                    f"Accelerator device {accelerator.device}."
                )
            if sharded_value.dtype != torch.bfloat16:
                errors.append(f"{name}: KT INT8 FSDP2 requires BF16 DTensors, got {sharded_value.dtype}.")

            local_shape = list(sharded_value.shape)
            chunk_size = (sharded_value.shape[0] + world_size - 1) // world_size
            start = min(rank * chunk_size, sharded_value.shape[0])
            local_shape[0] = min(chunk_size, sharded_value.shape[0] - start)
            if tuple(local_shape) != tuple(sharded_value.to_local().shape):
                errors.append(
                    f"{name}: computed local shape {tuple(local_shape)} does not match "
                    f"FSDP local shape {tuple(sharded_value.to_local().shape)}."
                )
            if not sharded_value.to_local().is_contiguous():
                errors.append(f"{name}: KT INT8 streaming loading requires contiguous local DTensor shards.")

            signature.append(
                (
                    name,
                    tuple(sharded_value.shape),
                    tuple(sharded_value.stride()),
                    str(sharded_value.dtype),
                    mesh.device_type,
                )
            )

            if rank != 0 or name not in full_sd:
                continue
            full_value = full_sd[name]
            if not isinstance(full_value, torch.Tensor):
                errors.append(f"{name}: rank-0 full state value must be a Tensor.")
            elif isinstance(full_value, DTensor):
                errors.append(f"{name}: rank-0 full state value must not be a DTensor.")
            elif full_value.device.type != "cpu":
                errors.append(f"{name}: rank-0 full state value must remain on CPU, got {full_value.device}.")
            elif full_value.layout != torch.strided:
                errors.append(f"{name}: only dense strided rank-0 tensors are supported.")
            elif full_value.dtype != torch.bfloat16:
                errors.append(f"{name}: KT INT8 FSDP2 requires BF16 rank-0 tensors, got {full_value.dtype}.")
            elif not full_value.is_contiguous():
                errors.append(f"{name}: rank-0 full state value must be contiguous.")
            elif tuple(full_value.shape) != tuple(sharded_value.shape):
                errors.append(
                    f"{name}: full shape {tuple(full_value.shape)} does not match DTensor shape "
                    f"{tuple(sharded_value.shape)}."
                )

        alias_groups = []
        for names in aliases_by_id.values():
            rank_zero_flags = {name in rank_zero_only_names for name in names}
            if len(rank_zero_flags) != 1:
                errors.append(f"Tied/shared state aliases mix rank ownership: {names}.")
                continue
            if rank_zero_flags == {False}:
                alias_groups.append(tuple(names))

            if rank == 0 and len(names) > 1 and all(name in full_sd for name in names):
                source = full_sd[names[0]]
                if not isinstance(source, torch.Tensor):
                    continue
                for alias in names[1:]:
                    alias_source = full_sd[alias]
                    if not isinstance(alias_source, torch.Tensor) or not _same_tensor_view(source, alias_source):
                        errors.append(
                            f"Rank-0 full state dict does not preserve tied/shared tensor storage for aliases {names}."
                        )
                        break

        gathered = [None] * world_size
        dist.all_gather_object(gathered, (errors, signature, alias_groups), group=cpu_group)
        all_errors = [
            f"rank {peer_rank}: {error}"
            for peer_rank, (peer_errors, _, _) in enumerate(gathered)
            for error in peer_errors
        ]
        reference_signature = gathered[0][1]
        reference_alias_groups = gathered[0][2]
        for peer_rank, (_, peer_signature, peer_alias_groups) in enumerate(gathered[1:], start=1):
            if peer_signature != reference_signature:
                all_errors.append(f"rank {peer_rank}: DTensor state signature differs from rank 0.")
            if peer_alias_groups != reference_alias_groups:
                all_errors.append(f"rank {peer_rank}: tied/shared state aliases differ from rank 0.")
        if all_errors:
            raise RuntimeError("KT INT8 streaming FSDP2 preflight failed: " + "; ".join(all_errors))

        local_dtype_bytes = {}
        local_max_tensor_bytes = 0
        global_dtype_bytes = {}
        global_max_tensor_bytes = 0
        installed_ids = set()

        if rank == 0:
            for name, sharded_value in sharded_state.items():
                if name not in rank_zero_only_names:
                    continue
                target = registered_tensors[name]
                target_id = id(target)
                if target_id in installed_ids:
                    continue
                _install_loaded_state_tensor(target, full_sd[name].detach())
                installed_ids.add(target_id)

        for name, sharded_value in sharded_state.items():
            if name in rank_zero_only_names:
                continue

            target = registered_tensors[name]
            target_id = id(target)
            if target_id in installed_ids:
                continue

            if not isinstance(sharded_value, DTensor):
                if rank == 0:
                    local_cpu = full_sd[name].detach().to(dtype=sharded_value.dtype).contiguous()
                else:
                    local_cpu = torch.empty(
                        tuple(sharded_value.shape),
                        dtype=sharded_value.dtype,
                        device="cpu",
                    )
                if local_cpu.numel() != 0:
                    dist.broadcast(local_cpu, src=0, group=cpu_group)
                local_tensor = local_cpu.to(accelerator.device)
                _install_loaded_state_tensor(target, local_tensor)
                installed_ids.add(target_id)
                dtype_name = str(local_cpu.dtype)
                tensor_bytes = local_cpu.numel() * local_cpu.element_size()
                local_dtype_bytes[dtype_name] = local_dtype_bytes.get(dtype_name, 0) + tensor_bytes
                local_max_tensor_bytes = max(local_max_tensor_bytes, tensor_bytes)
                if rank == 0:
                    global_dtype_bytes[dtype_name] = global_dtype_bytes.get(dtype_name, 0) + tensor_bytes
                    global_max_tensor_bytes = max(global_max_tensor_bytes, tensor_bytes)
                del local_tensor, local_cpu
                continue

            global_shape = tuple(sharded_value.shape)
            global_stride = tuple(sharded_value.stride())
            target_dtype = sharded_value.dtype
            mesh = sharded_value.device_mesh
            placements = tuple(sharded_value.placements)

            if rank == 0:
                full_value = full_sd[name].detach()
                for destination in range(1, world_size):
                    destination_shard = _kt_fsdp2_cpu_shard(full_value, destination, world_size)
                    destination_shard = destination_shard.to(dtype=target_dtype).contiguous()
                    if destination_shard.numel() != 0:
                        dist.send(destination_shard, dst=destination, group=cpu_group)

                local_cpu = _kt_fsdp2_cpu_shard(full_value, 0, world_size)
                local_cpu = local_cpu.to(dtype=target_dtype).contiguous()
            else:
                local_shape = list(global_shape)
                chunk_size = (global_shape[0] + world_size - 1) // world_size
                start = min(rank * chunk_size, global_shape[0])
                local_shape[0] = min(chunk_size, global_shape[0] - start)
                local_cpu = torch.empty(local_shape, dtype=target_dtype, device="cpu")
                if local_cpu.numel() != 0:
                    dist.recv(local_cpu, src=0, group=cpu_group)

            local_tensor = local_cpu.to(accelerator.device)
            loaded_value = DTensor.from_local(
                local_tensor,
                device_mesh=mesh,
                placements=placements,
                run_check=False,
                shape=torch.Size(global_shape),
                stride=global_stride,
            )
            _install_loaded_state_tensor(target, loaded_value)
            installed_ids.add(target_id)
            dtype_name = str(local_tensor.dtype)
            local_tensor_bytes = local_tensor.numel() * local_tensor.element_size()
            local_dtype_bytes[dtype_name] = local_dtype_bytes.get(dtype_name, 0) + local_tensor_bytes
            local_max_tensor_bytes = max(local_max_tensor_bytes, local_tensor_bytes)
            if rank == 0:
                global_tensor_bytes = sharded_value.numel() * sharded_value.element_size()
                global_dtype_bytes[dtype_name] = global_dtype_bytes.get(dtype_name, 0) + global_tensor_bytes
                global_max_tensor_bytes = max(global_max_tensor_bytes, global_tensor_bytes)
            del loaded_value, local_tensor, local_cpu

        inventory = [None] * world_size
        dist.all_gather_object(
            inventory,
            {"by_dtype": local_dtype_bytes, "max_tensor_bytes": local_max_tensor_bytes},
            group=cpu_group,
        )
        if rank == 0:
            logging.getLogger(__name__).info(
                "KT INT8 streaming FSDP2 state inventory: global_by_dtype=%s, global_max_tensor_bytes=%d, "
                "local_by_rank=%s",
                global_dtype_bytes,
                global_max_tensor_bytes,
                inventory,
            )

        expected_installed_ids = {
            id(tensor) for name, tensor in registered_tensors.items() if rank == 0 or name not in rank_zero_only_names
        }
        if installed_ids != expected_installed_ids:
            raise RuntimeError(
                "KT INT8 streaming FSDP2 did not install every locally owned state tensor: "
                f"missing_ids={sorted(expected_installed_ids - installed_ids)}, "
                f"unexpected_ids={sorted(installed_ids - expected_installed_ids)}."
            )
    finally:
        if owns_cpu_group:
            dist.destroy_process_group(cpu_group)

    loaded_requires_grad = {
        name: parameter.requires_grad for name, parameter in model.named_parameters(remove_duplicate=False)
    }
    if loaded_requires_grad != parameter_requires_grad:
        raise RuntimeError("KT INT8 streaming FSDP2 loading changed Parameter requires_grad ownership.")
    return model


def patch_fsdp2_kt_parameter_identity(model_args: "ModelArguments") -> None:
    if not model_args.use_kt or model_args.kt_expert_weight_format not in {"int8", "fp8"}:
        return

    import accelerate
    import accelerate.accelerator as accelerator_module
    import accelerate.utils as accelerate_utils
    from accelerate.utils import fsdp_utils

    original_prepare = getattr(fsdp_utils, "fsdp2_prepare_model", None)
    if original_prepare is None:
        raise RuntimeError("Accelerate does not expose fsdp2_prepare_model.")
    if getattr(original_prepare, "_llamafactory_kt_identity_patch", False):
        return

    for namespace_name, namespace in (
        ("accelerate.accelerator", accelerator_module),
        ("accelerate.utils", accelerate_utils),
    ):
        alias = getattr(namespace, "fsdp2_prepare_model", None)
        if alias is not original_prepare:
            raise RuntimeError(f"{namespace_name}.fsdp2_prepare_model does not match accelerate.utils.fsdp_utils.")

    _validate_accelerate_kt_fsdp2(accelerate, fsdp_utils, original_prepare)

    @wraps(original_prepare)
    def prepare_model(accelerator, model):
        explicit_kt_params = _collect_kt_fsdp_ignored_params(model)
        configured_params = _collect_configured_ignored_params(accelerator, model)
        allowed_ignored_params = explicit_kt_params | configured_params
        _validate_shardable_frozen_params(model, allowed_ignored_params)
        decoder_layers = _collect_native_deepseek_decoder_layers(model)
        expected_global_frozen_bf16_bytes = None
        if decoder_layers is not None:
            _validate_deepseek_fsdp2_parameter_ownership(
                model,
                decoder_layers,
                explicit_kt_params,
                configured_params,
            )
            expected_global_frozen_bf16_bytes = _frozen_bf16_parameter_bytes(
                model,
                explicit_kt_params,
                local_dtensors=False,
            )

        kt_parameter_ids = {id(param) for param in explicit_kt_params}
        rank_zero_only_names = frozenset(
            name for name, param in model.named_parameters(remove_duplicate=False) if id(param) in kt_parameter_ids
        )

        import torch.distributed.fsdp as torch_fsdp

        original_fully_shard = torch_fsdp.fully_shard
        original_load_full_state_dict = fsdp_utils.fsdp2_load_full_state_dict
        if "ignored_params" not in inspect.signature(original_fully_shard).parameters:
            raise RuntimeError("This PyTorch fully_shard does not support ignored_params.")
        if accelerate_utils.fsdp2_load_full_state_dict is not original_load_full_state_dict:
            raise RuntimeError(
                "accelerate.utils.fsdp2_load_full_state_dict does not match accelerate.utils.fsdp_utils."
            )

        module_names = {id(module): name or "<root>" for name, module in model.named_modules()}
        fully_shard_calls = 0
        fully_shard_records: list[tuple[int, str, object]] = []

        @wraps(original_fully_shard)
        def selective_fully_shard(module, *args, **kwargs):
            nonlocal fully_shard_calls
            fully_shard_calls += 1
            fully_shard_records.append(
                (
                    id(module),
                    module_names.get(id(module), f"<unknown:{type(module).__name__}>"),
                    kwargs.get("reshard_after_forward"),
                )
            )
            kwargs["ignored_params"] = allowed_ignored_params
            return original_fully_shard(module, *args, **kwargs)

        @wraps(_kt_fsdp2_streaming_load_full_state_dict)
        def streaming_load_full_state_dict(accelerator, model, full_sd, cpu_offload=False):
            return _kt_fsdp2_streaming_load_full_state_dict(
                accelerator,
                model,
                full_sd,
                cpu_offload,
                rank_zero_only_names=rank_zero_only_names,
            )

        with _FSDP2_PATCH_LOCK:
            torch_fsdp.fully_shard = selective_fully_shard
            fsdp_utils.fsdp2_load_full_state_dict = streaming_load_full_state_dict
            accelerate_utils.fsdp2_load_full_state_dict = streaming_load_full_state_dict
            try:
                with _preserve_parameter_identity_on_conversion():
                    prepared_model = original_prepare(accelerator, model)
            finally:
                torch_fsdp.fully_shard = original_fully_shard
                fsdp_utils.fsdp2_load_full_state_dict = original_load_full_state_dict
                accelerate_utils.fsdp2_load_full_state_dict = original_load_full_state_dict

        if fully_shard_calls == 0:
            raise RuntimeError("Accelerate fsdp2_prepare_model did not call fully_shard.")

        if decoder_layers is not None:
            if expected_global_frozen_bf16_bytes is None:
                raise RuntimeError("DeepSeek FSDP2 frozen BF16 inventory was not captured before preparation.")
            _validate_prepared_deepseek_fsdp2(
                prepared_model,
                decoder_layers,
                explicit_kt_params,
                fully_shard_records,
                expected_global_frozen_bf16_bytes,
            )
            _install_first_step_fsdp_telemetry(prepared_model, decoder_layers)
        else:
            prepared_parameter_ids = {id(param) for param in prepared_model.parameters()}
            if not kt_parameter_ids.issubset(prepared_parameter_ids):
                raise RuntimeError("FSDP2 replaced explicitly ignored KT expert Parameter objects.")

        return prepared_model

    prepare_model._llamafactory_kt_identity_patch = True
    accelerator_module.fsdp2_prepare_model = prepare_model
    accelerate_utils.fsdp2_prepare_model = prepare_model
    fsdp_utils.fsdp2_prepare_model = prepare_model
