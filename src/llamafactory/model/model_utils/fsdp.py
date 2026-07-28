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
import re
import threading
import warnings
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
            "frozen_params_to_ignore = set()",
            "if not param.requires_grad:",
            'fsdp2_kwargs["ignored_params"] = ignored | frozen_params_to_ignore',
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


def patch_fsdp2_kt_parameter_identity(model_args: "ModelArguments") -> None:
    if not model_args.use_kt or model_args.kt_expert_weight_format != "int8":
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

        kt_parameter_ids = {id(param) for param in explicit_kt_params}

        import torch.distributed.fsdp as torch_fsdp

        original_fully_shard = torch_fsdp.fully_shard
        if "ignored_params" not in inspect.signature(original_fully_shard).parameters:
            raise RuntimeError("This PyTorch fully_shard does not support ignored_params.")

        fully_shard_calls = 0

        @wraps(original_fully_shard)
        def selective_fully_shard(module, *args, **kwargs):
            nonlocal fully_shard_calls
            fully_shard_calls += 1
            kwargs["ignored_params"] = allowed_ignored_params
            return original_fully_shard(module, *args, **kwargs)

        with _FSDP2_PATCH_LOCK:
            torch_fsdp.fully_shard = selective_fully_shard
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message=r"Found \d+ frozen params .*Excluding them from FSDP2 sharding.*",
                    )
                    with _preserve_parameter_identity_on_conversion():
                        prepared_model = original_prepare(accelerator, model)
            finally:
                torch_fsdp.fully_shard = original_fully_shard

        if fully_shard_calls == 0:
            raise RuntimeError("Accelerate fsdp2_prepare_model did not call fully_shard.")

        prepared_parameter_ids = {id(param) for param in prepared_model.parameters()}
        if not kt_parameter_ids.issubset(prepared_parameter_ids):
            raise RuntimeError("FSDP2 replaced explicitly ignored KT expert Parameter objects.")

        return prepared_model

    prepare_model._llamafactory_kt_identity_patch = True
    accelerator_module.fsdp2_prepare_model = prepare_model
    accelerate_utils.fsdp2_prepare_model = prepare_model
    fsdp_utils.fsdp2_prepare_model = prepare_model
