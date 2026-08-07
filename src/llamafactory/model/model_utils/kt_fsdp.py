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

import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING, Any, Optional

import torch
import torch.distributed as dist

from ...extras import logging


if TYPE_CHECKING:
    from torch.utils.hooks import RemovableHandle

    from ...hparams import ModelArguments


logger = logging.get_logger(__name__)

_FROZEN_ACCELERATE_KT_VERSION = "1.14.0.post1"


@dataclass(frozen=True)
class _BufferSpec:
    name: str
    shape: tuple[int, ...]
    dtype: torch.dtype


def _is_fsdp2(accelerator: Any) -> bool:
    plugin = getattr(getattr(accelerator, "state", None), "fsdp_plugin", None)
    return bool(getattr(accelerator, "is_fsdp2", False) or getattr(plugin, "fsdp_version", 1) == 2)


def _installed_accelerate_kt_version() -> Optional[str]:
    try:
        return version("accelerate-kt")
    except PackageNotFoundError:
        return None


def _persistent_buffer_specs(model: torch.nn.Module) -> tuple[_BufferSpec, ...]:
    specs: list[_BufferSpec] = []
    for module_name, module in model.named_modules():
        for local_name, buffer in module._buffers.items():
            if buffer is None or local_name in module._non_persistent_buffers_set:
                continue

            name = f"{module_name}.{local_name}" if module_name else local_name
            specs.append(_BufferSpec(name=name, shape=tuple(buffer.shape), dtype=buffer.dtype))

    return tuple(sorted(specs, key=lambda spec: spec.name))


def _all_rank_errors(local_error: Optional[str]) -> tuple[str, ...]:
    if not dist.is_available() or not dist.is_initialized():
        return (local_error,) if local_error is not None else ()

    errors: list[Optional[str]] = [None] * dist.get_world_size()
    dist.all_gather_object(errors, local_error)
    return tuple(f"rank {rank}: {error}" for rank, error in enumerate(errors) if error is not None)


def _raise_collective_errors(stage: str, local_error: Optional[str]) -> None:
    errors = _all_rank_errors(local_error)
    if errors:
        raise RuntimeError(f"KT FSDP2 persistent-buffer repair failed during {stage}: {'; '.join(errors)}")


def raise_kt_distributed_save_errors(local_error: Optional[str], *, cause: Optional[Exception] = None) -> None:
    r"""Exchange adapter-save failures so non-main ranks never wait on a failed rank 0."""
    errors = _all_rank_errors(local_error)
    if not errors:
        return

    error = RuntimeError(f"KT FSDP2 adapter save failed: {'; '.join(errors)}")
    if cause is not None:
        raise error from cause
    raise error


def _buffer_parent(model: torch.nn.Module, name: str) -> tuple[torch.nn.Module, str]:
    if "." not in name:
        return model, name

    parent_name, local_name = name.rsplit(".", 1)
    return model.get_submodule(parent_name), local_name


def register_kt_fsdp2_persistent_buffer_hook(model: torch.nn.Module, accelerator: Any) -> Optional["RemovableHandle"]:
    r"""Repair persistent buffers left on meta by frozen accelerate-kt's rank-0 FSDP2 load.

    The hook is registered before ``Accelerator.prepare`` and removes itself after the first
    successful ``load_state_dict(assign=True)``. Every rank performs the same ordered set of
    collectives, and allocation errors are exchanged before entering each broadcast.
    """
    specs = _persistent_buffer_specs(model)
    if not specs:
        return None

    handle: Optional[RemovableHandle] = None

    def repair_buffers(loaded_model: torch.nn.Module, _incompatible_keys: Any) -> None:
        nonlocal handle
        distributed = dist.is_available() and dist.is_initialized()
        rank = dist.get_rank() if distributed else 0
        device = torch.device(getattr(accelerator, "device", "cpu"))

        if distributed:
            signature = tuple((spec.name, spec.shape, str(spec.dtype)) for spec in specs)
            signatures: list[Any] = [None] * dist.get_world_size()
            dist.all_gather_object(signatures, signature)
            if any(candidate != signature for candidate in signatures):
                raise RuntimeError(
                    "KT FSDP2 persistent-buffer repair requires identical ordered buffer metadata on every rank."
                )

        for spec in specs:
            tensor: Optional[torch.Tensor] = None
            local_error: Optional[str] = None
            try:
                current = loaded_model.get_buffer(spec.name)
                if tuple(current.shape) != spec.shape or current.dtype != spec.dtype:
                    raise RuntimeError(
                        f"{spec.name} changed from shape={spec.shape}, dtype={spec.dtype} "
                        f"to shape={tuple(current.shape)}, dtype={current.dtype}"
                    )
                if rank == 0:
                    if current.is_meta:
                        raise RuntimeError(f"rank 0 source buffer {spec.name} is on meta")
                    tensor = current.detach().to(device=device).contiguous()
                else:
                    tensor = torch.empty(spec.shape, dtype=spec.dtype, device=device)
            except Exception as exc:  # exchange the error before any rank enters broadcast
                local_error = f"preparing {spec.name}: {exc}"

            _raise_collective_errors("allocation", local_error)
            assert tensor is not None
            if distributed:
                dist.broadcast(tensor, src=0, group=dist.group.WORLD)

            local_error = None
            try:
                parent, local_name = _buffer_parent(loaded_model, spec.name)
                parent.register_buffer(local_name, tensor, persistent=True)
            except Exception as exc:
                local_error = f"installing {spec.name}: {exc}"

            _raise_collective_errors("installation", local_error)

        if handle is not None:
            handle.remove()
        logger.info_rank0(f"Restored {len(specs)} persistent FSDP2 buffers from rank 0.")

    handle = model.register_load_state_dict_post_hook(repair_buffers)
    return handle


def maybe_register_kt_fsdp2_persistent_buffer_hook(
    model: torch.nn.Module,
    accelerator: Any,
    use_kt: bool,
    *,
    accelerate_kt_version: Optional[str] = None,
) -> Optional["RemovableHandle"]:
    r"""Install the narrow compatibility hook only for the frozen accelerate-kt ABI."""
    if not use_kt or not _is_fsdp2(accelerator):
        return None

    plugin = getattr(getattr(accelerator, "state", None), "fsdp_plugin", None)
    if not bool(getattr(plugin, "cpu_ram_efficient_loading", False)):
        return None

    installed_version = accelerate_kt_version or _installed_accelerate_kt_version()
    if installed_version != _FROZEN_ACCELERATE_KT_VERSION:
        logger.warning_rank0(
            "Skipping the frozen accelerate-kt FSDP2 buffer workaround because accelerate-kt "
            f"{installed_version or 'is not installed'} (expected {_FROZEN_ACCELERATE_KT_VERSION})."
        )
        return None

    return register_kt_fsdp2_persistent_buffer_hook(model, accelerator)


def _get_kt_placeholders(
    model: torch.nn.Module,
    placeholder_getter: Optional[Callable[[torch.nn.Module], Mapping[str, torch.nn.Parameter]]],
) -> Mapping[str, torch.nn.Parameter]:
    if placeholder_getter is None:
        try:
            from kt_kernel.sft import get_kt_expert_placeholders
        except (ImportError, ModuleNotFoundError) as exc:
            raise RuntimeError(
                "The installed kt-kernel must expose get_kt_expert_placeholders() for KT FSDP2 adapter saving."
            ) from exc

        placeholder_getter = get_kt_expert_placeholders

    placeholders = placeholder_getter(model)
    if not isinstance(placeholders, Mapping):
        raise TypeError("get_kt_expert_placeholders() must return an FQN-to-Parameter mapping.")

    named_parameters = dict(model.named_parameters(remove_duplicate=False))
    for name, parameter in placeholders.items():
        if not isinstance(name, str) or not isinstance(parameter, torch.nn.Parameter):
            raise TypeError("KT expert placeholders must map string FQNs to torch.nn.Parameter objects.")
        if name not in named_parameters or named_parameters[name] is not parameter:
            raise RuntimeError(f"KT expert placeholder {name!r} does not match the current model parameter identity.")

    return placeholders


def get_kt_fsdp2_adapter_state_dict(
    model: torch.nn.Module,
    *,
    placeholder_getter: Optional[Callable[[torch.nn.Module], Mapping[str, torch.nn.Parameter]]] = None,
) -> dict[str, torch.Tensor]:
    r"""Collect a CPU full state dict containing trainable PEFT tensors only.

    ``placeholder_getter`` exists for isolated tests. Production callers must use the public
    kt-kernel identity API; no parameter name, stride, or storage heuristics are accepted.
    """
    from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict

    placeholders: Mapping[str, torch.nn.Parameter] = {}
    local_error: Optional[str] = None
    try:
        placeholders = _get_kt_placeholders(model, placeholder_getter)
    except Exception as exc:
        local_error = f"{type(exc).__name__}: {exc}"

    errors = _all_rank_errors(local_error)
    if errors:
        raise RuntimeError(f"KT FSDP2 adapter gather preflight failed: {'; '.join(errors)}")

    if dist.is_available() and dist.is_initialized():
        signature = tuple(
            (name, tuple(parameter.shape), str(parameter.dtype)) for name, parameter in placeholders.items()
        )
        signatures: list[Any] = [None] * dist.get_world_size()
        dist.all_gather_object(signatures, signature)
        if any(candidate != signature for candidate in signatures):
            raise RuntimeError("KT FSDP2 adapter gather requires identical placeholder metadata on every rank.")

    unique_placeholders = {id(parameter): parameter for parameter in placeholders.values()}
    original_requires_grad = {identity: parameter.requires_grad for identity, parameter in unique_placeholders.items()}
    try:
        # DCP's frozen-parameter filter expects every frozen parameter in model.state_dict().
        # KT deliberately omits expert placeholders, so exclude them from that filter while
        # the real trainable adapter tensors are gathered.
        for parameter in unique_placeholders.values():
            parameter.requires_grad_(True)

        state_dict = get_model_state_dict(
            model,
            options=StateDictOptions(full_state_dict=True, cpu_offload=True, ignore_frozen_params=True),
        )
        for name in placeholders:
            state_dict.pop(name, None)
        return state_dict
    finally:
        for identity, parameter in unique_placeholders.items():
            parameter.requires_grad_(original_requires_grad[identity])


def is_kt_fsdp2_peft(model: torch.nn.Module, accelerator: Any, use_kt: bool) -> bool:
    if not use_kt or not _is_fsdp2(accelerator):
        return False

    try:
        unwrapped_model = accelerator.unwrap_model(model, keep_torch_compile=False)
    except (AttributeError, TypeError):
        unwrapped_model = model
    return getattr(unwrapped_model, "peft_config", None) is not None


def validate_kt_distributed_checkpoint_policy(
    model_args: "ModelArguments", training_args: Any, *, world_size: Optional[int] = None
) -> None:
    r"""Reject unsupported multi-rank KT optimizer checkpoint and resume semantics."""
    if not model_args.use_kt:
        return

    if world_size is None:
        world_size = int(os.environ.get("WORLD_SIZE", getattr(training_args, "world_size", 1)))
    if world_size <= 1:
        return

    if getattr(training_args, "resume_from_checkpoint", None):
        raise ValueError(
            "Multi-rank KTransformers checkpoint resume is not supported. Start a fresh training run instead."
        )

    if getattr(training_args, "load_best_model_at_end", False):
        raise ValueError(
            "Multi-rank KTransformers does not support `load_best_model_at_end`; "
            "final and periodic saves contain adapters only."
        )

    save_strategy = getattr(training_args, "save_strategy", "no")
    save_strategy = getattr(save_strategy, "value", save_strategy)
    if str(save_strategy).lower() != "no" and not getattr(training_args, "save_only_model", False):
        raise ValueError(
            "Multi-rank KTransformers periodic optimizer checkpoints are not supported. "
            "Set `save_strategy: no` for a final adapter save, or set `save_only_model: true`."
        )
