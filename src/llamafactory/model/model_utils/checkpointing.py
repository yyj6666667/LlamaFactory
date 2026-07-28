# Copyright 2025 HuggingFace Inc., Daniel Han-Chen & the Unsloth team and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's Transformers and PEFT library,
# https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/modeling_utils.py
# https://github.com/huggingface/peft/blob/v0.10.0/src/peft/utils/other.py
# and the Unsloth library.
# https://github.com/unslothai/unsloth/blob/July-2024/unsloth/models/_utils.py
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
import os
from collections.abc import Callable
from functools import WRAPPER_ASSIGNMENTS, partial, wraps
from types import MethodType
from typing import TYPE_CHECKING, Any, Optional, Union

import torch

from ...extras import logging
from ...extras.constants import LAYERNORM_NAMES


if TYPE_CHECKING:
    from transformers import PreTrainedModel

    from ...hparams import ModelArguments


logger = logging.get_logger(__name__)


def validate_kt_deepseek_v3_native_config(config: Any, model_args: "ModelArguments") -> None:
    r"""Validate the production model-code contract before loading DeepSeek-V3 INT8 weights."""
    if not model_args.use_kt or model_args.kt_expert_weight_format != "int8":
        return

    if model_args.trust_remote_code:
        raise RuntimeError(
            "KTransformers DeepSeek-V3 INT8 LoRA must use the native Transformers implementation "
            "with `trust_remote_code: false`."
        )

    config_type = type(config)
    if (
        getattr(config, "model_type", None) != "deepseek_v3"
        or config_type.__name__ != "DeepseekV3Config"
        or not config_type.__module__.startswith("transformers.models.deepseek_v3.")
    ):
        raise RuntimeError(
            "KTransformers INT8 LoRA currently requires the native "
            "`transformers.models.deepseek_v3.DeepseekV3Config`; "
            f"got {config_type.__module__}.{config_type.__name__}."
        )

    expected = {
        "num_hidden_layers": 61,
        "first_k_dense_replace": 3,
        "n_routed_experts": 256,
    }
    mismatches = {
        name: (getattr(config, name, None), value)
        for name, value in expected.items()
        if getattr(config, name, None) != value
    }
    if mismatches:
        raise RuntimeError(f"DeepSeek-V3.1 INT8 model structure does not match the production contract: {mismatches}.")


def _validate_kt_deepseek_v3_checkpointing(model: "PreTrainedModel", model_args: "ModelArguments") -> None:
    if not model_args.use_kt or model_args.kt_expert_weight_format != "int8":
        return

    try:
        from transformers.modeling_layers import GradientCheckpointingLayer
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError(
            "KTransformers DeepSeek-V3 INT8 LoRA requires a Transformers build with `GradientCheckpointingLayer`."
        ) from exc

    model_type = type(model)
    if not model_type.__module__.startswith("transformers.models.deepseek_v3."):
        raise RuntimeError(
            f"DeepSeek-V3 INT8 loaded non-native model code: {model_type.__module__}.{model_type.__name__}."
        )

    base_model = getattr(model, "model", None)
    layers = getattr(base_model, "layers", None)
    if not isinstance(layers, torch.nn.ModuleList) or len(layers) != 61:
        raise RuntimeError("DeepSeek-V3.1 INT8 requires exactly 61 native decoder layers.")

    invalid_layer_types = [
        index for index, layer in enumerate(layers) if not isinstance(layer, GradientCheckpointingLayer)
    ]
    if invalid_layer_types:
        raise RuntimeError(
            "Every DeepSeek-V3.1 decoder layer must inherit `GradientCheckpointingLayer`; "
            f"invalid layers: {invalid_layer_types}."
        )

    disabled_layers = [index for index, layer in enumerate(layers) if not layer.gradient_checkpointing]
    if disabled_layers:
        raise RuntimeError(f"Gradient checkpointing is disabled on decoder layers: {disabled_layers}.")

    expected_context_fn = model_args.kt_activation_checkpoint_context_fn
    invalid_checkpoint_functions = []
    for index, layer in enumerate(layers):
        checkpoint_func = getattr(layer, "_gradient_checkpointing_func", None)
        checkpoint_contract = getattr(checkpoint_func, "_llamafactory_checkpoint_contract", None)
        if (
            not isinstance(checkpoint_contract, dict)
            or checkpoint_contract.get("use_reentrant") is not False
            or checkpoint_contract.get("context_fn") is not expected_context_fn
        ):
            invalid_checkpoint_functions.append(index)

    if invalid_checkpoint_functions:
        raise RuntimeError(
            "DeepSeek-V3.1 INT8 requires non-reentrant checkpointing with the exact KT context_fn; "
            f"invalid layers: {invalid_checkpoint_functions}."
        )

    if getattr(model.config, "use_cache", None) is not False:
        raise RuntimeError("DeepSeek-V3.1 INT8 checkpointing requires `model.config.use_cache = false`.")

    kt_wrappers = [module for module in model.modules() if getattr(module, "_is_kt_moe_wrapper", False)]
    wrapper_layers = [getattr(wrapper, "layer_idx", None) for wrapper in kt_wrappers]
    if not all(isinstance(layer_idx, int) for layer_idx in wrapper_layers):
        raise RuntimeError("Every DeepSeek-V3.1 KT MoE wrapper must declare an integer `layer_idx`.")

    wrapper_layers.sort()
    if wrapper_layers != list(range(3, 61)):
        raise RuntimeError(
            "DeepSeek-V3.1 INT8 requires one KT MoE wrapper on every routed layer 3..60; "
            f"got layer indices {wrapper_layers}."
        )


def get_unsloth_gradient_checkpointing_func() -> Callable:
    class UnslothGradientCheckpointing(torch.autograd.Function):
        r"""Saves VRAM by smartly offloading to RAM."""

        @staticmethod
        @torch.cuda.amp.custom_fwd
        def forward(
            ctx: "torch.autograd.Function",
            forward_function: "torch.Module",
            hidden_states: "torch.Tensor",
            *args: Union["torch.Tensor", Any],
        ) -> "torch.Tensor":
            saved_hidden_states = hidden_states.to("cpu", non_blocking=True)
            with torch.no_grad():
                outputs = forward_function(hidden_states, *args)

            ctx.save_for_backward(saved_hidden_states)
            ctx.forward_function = forward_function
            ctx.args = args
            return outputs

        @staticmethod
        @torch.cuda.amp.custom_bwd
        def backward(ctx: "torch.autograd.Function", grad_output: "torch.Tensor") -> "torch.Tensor":
            (hidden_states,) = ctx.saved_tensors
            hidden_states = hidden_states.to("cuda", non_blocking=True).detach()
            hidden_states.requires_grad_(True)
            with torch.enable_grad():
                outputs = ctx.forward_function(hidden_states, *ctx.args)
                output = outputs[0] if isinstance(outputs, tuple) else outputs

            torch.autograd.backward(output, grad_output)
            return (None, hidden_states.grad) + (None,) * len(ctx.args)

    return UnslothGradientCheckpointing.apply


def get_custom_gradient_checkpointing_func(gradient_checkpointing_func: Callable) -> Callable:
    r"""Only applies gradient checkpointing to trainable layers."""

    @wraps(gradient_checkpointing_func, assigned=WRAPPER_ASSIGNMENTS + ("__self__",))
    def custom_gradient_checkpointing_func(func: Callable, *args: Union["torch.Tensor", Any], **kwargs):
        if isinstance(func, partial):
            module: torch.nn.Module = func.func.__self__
        else:
            module: torch.nn.Module = func.__self__

        has_grad = False
        if any(param.requires_grad for param in module.parameters()):
            has_grad = True
            for arg in args:
                if torch.is_tensor(arg) and torch.is_floating_point(arg):
                    arg.requires_grad_(True)
                    break  # assume the first tensor is always the hidden states

        if has_grad:
            return gradient_checkpointing_func(func, *args, **kwargs)
        else:
            return func(*args, **kwargs)

    return custom_gradient_checkpointing_func


def _gradient_checkpointing_enable(
    self: "PreTrainedModel",
    gradient_checkpointing_kwargs: Optional[dict[str, Any]] = None,
    use_unsloth_gc: bool = False,
) -> None:
    r"""Activates gradient checkpointing for the current model.

    Modification of the original method to enable gradient checkpointing for block-wise optimizer.
    """
    from torch.utils.checkpoint import checkpoint

    if not self.supports_gradient_checkpointing:
        raise ValueError(f"{self.__class__.__name__} does not support gradient checkpointing.")

    if gradient_checkpointing_kwargs is None:
        gradient_checkpointing_kwargs = {"use_reentrant": True}

    if use_unsloth_gc:
        gradient_checkpointing_func = get_unsloth_gradient_checkpointing_func()
    else:
        gradient_checkpointing_func = partial(checkpoint, **gradient_checkpointing_kwargs)

    gradient_checkpointing_func = get_custom_gradient_checkpointing_func(gradient_checkpointing_func)
    gradient_checkpointing_func._llamafactory_checkpoint_contract = {
        "use_reentrant": gradient_checkpointing_kwargs.get("use_reentrant", True),
        "context_fn": gradient_checkpointing_kwargs.get("context_fn"),
    }
    if "value" in inspect.signature(self._set_gradient_checkpointing).parameters:  # old GC format
        self.apply(partial(self._set_gradient_checkpointing, value=True))
        self.enable_input_require_grads()
        logger.warning_rank0_once("You are using the old GC format, some features (e.g. BAdam) will be invalid.")
    else:  # have already enabled input require gradients
        self._set_gradient_checkpointing(enable=True, gradient_checkpointing_func=gradient_checkpointing_func)


def _fp32_forward_post_hook(
    module: "torch.nn.Module", args: tuple["torch.Tensor"], output: "torch.Tensor"
) -> "torch.Tensor":
    return output.to(torch.float32)


def prepare_model_for_training(model: "PreTrainedModel", model_args: "ModelArguments") -> None:
    r"""Prepare the model before training.

    Include:
    (1) cast the layernorm in fp32
    (2) make output embedding layer require grads
    (3) add the upcasting of the lm_head in fp32.
    """
    if model_args.upcast_layernorm:
        logger.info_rank0("Upcasting layernorm weights in float32.")
        for name, param in model.named_parameters():
            if param.ndim == 1 and any(ln_name in name for ln_name in LAYERNORM_NAMES):
                param.data = param.data.to(torch.float32)

    if (
        os.environ.get("ACCELERATE_USE_FSDP", "false").lower() == "true"
        and int(os.environ.get("FSDP_VERSION", "1")) == 2
    ):
        model_args.use_reentrant_gc = False
        logger.warning_rank0("You are using fsdp2, `use_reentrant_gc` has been set to False.")

    if not model_args.disable_gradient_checkpointing:
        if not getattr(model, "supports_gradient_checkpointing", False):
            logger.warning_rank0("Current model does not support gradient checkpointing.")
        else:
            # use_reentrant=False might increase VRAM usage (have not been empirically verified yet)
            # According to: https://github.com/huggingface/transformers/issues/28339
            gradient_checkpointing_enable = partial(
                _gradient_checkpointing_enable, use_unsloth_gc=model_args.use_unsloth_gc
            )
            model.gradient_checkpointing_enable = MethodType(gradient_checkpointing_enable, model)
            gradient_checkpointing_kwargs = {"use_reentrant": model_args.use_reentrant_gc}
            if model_args.kt_activation_checkpoint_context_fn is not None:
                gradient_checkpointing_kwargs["context_fn"] = model_args.kt_activation_checkpoint_context_fn

            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)
            setattr(model.config, "use_cache", False)  # turn off when gradient checkpointing is enabled
            _validate_kt_deepseek_v3_checkpointing(model, model_args)
            logger.info_rank0("Gradient checkpointing enabled.")

    if model_args.upcast_lmhead_output:
        output_layer = model.get_output_embeddings()
        if isinstance(output_layer, torch.nn.Linear) and output_layer.weight.dtype != torch.float32:
            logger.info_rank0("Upcasting lm_head outputs in float32.")
            output_layer.register_forward_hook(_fp32_forward_post_hook)
