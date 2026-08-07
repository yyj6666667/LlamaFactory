# Copyright 2025 HuggingFace Inc., the KVCache.AI team, Approaching AI, and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/examples/pytorch/language-modeling/run_clm.py
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

import json
import os
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Literal, Self

import torch
from omegaconf import OmegaConf
from transformers.training_args import _convert_str_dict

from ..extras.constants import AttentionFunction, EngineName, QuantizationMethod, RopeScaling
from ..extras.logging import get_logger


logger = get_logger(__name__)


@dataclass
class BaseModelArguments:
    r"""Arguments pertaining to the model."""

    model_name_or_path: str | None = field(
        default=None,
        metadata={
            "help": "Path to the model weight or identifier from huggingface.co/models or modelscope.cn/models."
        },
    )
    adapter_name_or_path: str | None = field(
        default=None,
        metadata={
            "help": (
                "Path to the adapter weight or identifier from huggingface.co/models. "
                "Use commas to separate multiple adapters."
            )
        },
    )
    adapter_folder: str | None = field(
        default=None,
        metadata={"help": "The folder containing the adapter weights to load."},
    )
    cache_dir: str | None = field(
        default=None,
        metadata={"help": "Where to store the pre-trained models downloaded from huggingface.co or modelscope.cn."},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether or not to use one of the fast tokenizer (backed by the tokenizers library)."},
    )
    resize_vocab: bool = field(
        default=False,
        metadata={"help": "Whether or not to resize the tokenizer vocab and the embedding layers."},
    )
    split_special_tokens: bool = field(
        default=False,
        metadata={"help": "Whether or not the special tokens should be split during the tokenization process."},
    )
    add_tokens: str | None = field(
        default=None,
        metadata={
            "help": "Non-special tokens to be added into the tokenizer. Use commas to separate multiple tokens."
        },
    )
    add_special_tokens: str | None = field(
        default=None,
        metadata={"help": "Special tokens to be added into the tokenizer. Use commas to separate multiple tokens."},
    )
    new_special_tokens_config: str | None = field(
        default=None,
        metadata={
            "help": (
                "Path to YAML config with special token descriptions for semantic initialization. "
                "If set, this takes precedence over add_special_tokens. "
                "YAML format: {'<token>': 'description text', ...}"
            )
        },
    )
    init_special_tokens: Literal["noise_init", "desc_init", "desc_init_w_noise"] = field(
        default="noise_init",
        metadata={
            "help": (
                "Initialization method for new special tokens: "
                "'noise_init' (default, random noise around mean), "
                "'desc_init' (semantic initialization from descriptions), "
                "'desc_init_w_noise' (semantic + random noise). "
                "Note: 'desc_init' methods require new_special_tokens_config."
            )
        },
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    low_cpu_mem_usage: bool = field(
        default=True,
        metadata={"help": "Whether or not to use memory-efficient model loading."},
    )
    rope_scaling: RopeScaling | None = field(
        default=None,
        metadata={"help": "Which scaling strategy should be adopted for the RoPE embeddings."},
    )
    flash_attn: AttentionFunction = field(
        default=AttentionFunction.AUTO,
        metadata={"help": "Enable FlashAttention for faster training and inference."},
    )
    shift_attn: bool = field(
        default=False,
        metadata={"help": "Enable shift short attention (S^2-Attn) proposed by LongLoRA."},
    )
    mixture_of_depths: Literal["convert", "load"] | None = field(
        default=None,
        metadata={"help": "Convert the model to mixture-of-depths (MoD) or load the MoD model."},
    )
    use_unsloth: bool = field(
        default=False,
        metadata={"help": "Whether or not to use unsloth's optimization for the LoRA training."},
    )
    use_unsloth_gc: bool = field(
        default=False,
        metadata={"help": "Whether or not to use unsloth's gradient checkpointing (no need to install unsloth)."},
    )
    enable_liger_kernel: bool = field(
        default=False,
        metadata={"help": "Whether or not to enable liger kernel for faster training."},
    )
    moe_aux_loss_coef: float | None = field(
        default=None,
        metadata={"help": "Coefficient of the auxiliary router loss in mixture-of-experts model."},
    )
    disable_gradient_checkpointing: bool = field(
        default=False,
        metadata={"help": "Whether or not to disable gradient checkpointing."},
    )
    use_reentrant_gc: bool = field(
        default=True,
        metadata={"help": "Whether or not to use reentrant gradient checkpointing."},
    )
    upcast_layernorm: bool = field(
        default=False,
        metadata={"help": "Whether or not to upcast the layernorm weights in fp32."},
    )
    upcast_lmhead_output: bool = field(
        default=False,
        metadata={"help": "Whether or not to upcast the output of lm_head in fp32."},
    )
    train_from_scratch: bool = field(
        default=False,
        metadata={"help": "Whether or not to randomly initialize the model weights."},
    )
    infer_backend: EngineName = field(
        default=EngineName.HF,
        metadata={"help": "Backend engine used at inference."},
    )
    offload_folder: str = field(
        default="offload",
        metadata={"help": "Path to offload model weights."},
    )
    use_kv_cache: bool = field(
        default=True,
        metadata={"help": "Whether or not to use KV cache in generation."},
    )
    use_v1_kernels: bool | None = field(
        default=False,
        metadata={"help": "Whether or not to use high-performance kernels in training."},
    )
    infer_dtype: Literal["auto", "float16", "bfloat16", "float32"] = field(
        default="auto",
        metadata={"help": "Data type for model weights and activations at inference."},
    )
    hf_hub_token: str | None = field(
        default=None,
        metadata={"help": "Auth token to log in with Hugging Face Hub."},
    )
    ms_hub_token: str | None = field(
        default=None,
        metadata={"help": "Auth token to log in with ModelScope Hub."},
    )
    om_hub_token: str | None = field(
        default=None,
        metadata={"help": "Auth token to log in with Modelers Hub."},
    )
    print_param_status: bool = field(
        default=False,
        metadata={"help": "For debugging purposes, print the status of the parameters in the model."},
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={"help": "Whether to trust the execution of code from datasets/models defined on the Hub or not."},
    )

    def __post_init__(self):
        if self.model_name_or_path is None:
            raise ValueError("Please provide `model_name_or_path`.")

        if self.adapter_name_or_path is not None:  # support merging multiple lora weights
            self.adapter_name_or_path = [path.strip() for path in self.adapter_name_or_path.split(",")]

        if self.add_tokens is not None:  # support multiple tokens
            self.add_tokens = [token.strip() for token in self.add_tokens.split(",")]

        # Process special tokens with priority: new_special_tokens_config > add_special_tokens
        if self.new_special_tokens_config is not None:
            # Priority 1: Load from YAML config (extracts both tokens and descriptions)
            try:
                cfg = OmegaConf.load(self.new_special_tokens_config)
                token_descriptions = OmegaConf.to_container(cfg)

                if not isinstance(token_descriptions, dict):
                    raise ValueError(
                        f"YAML config must be a dictionary mapping tokens to descriptions. "
                        f"Got: {type(token_descriptions)}"
                    )

                # Extract token list from config keys
                extracted_tokens = list(token_descriptions.keys())

                # Warn if both are set
                if self.add_special_tokens is not None:
                    logger.warning_rank0(
                        "Both 'new_special_tokens_config' and 'add_special_tokens' are set. "
                        f"Using tokens from config: {extracted_tokens}"
                    )

                # Override add_special_tokens with extracted tokens (as list)
                self.add_special_tokens = extracted_tokens

                # Store descriptions internally for later use (internal attribute)
                self._special_token_descriptions = token_descriptions

                logger.info_rank0(
                    f"Loaded {len(extracted_tokens)} special tokens with descriptions from: "
                    f"{self.new_special_tokens_config}"
                )

            except Exception as e:
                logger.error_rank0(
                    f"Failed to load special tokens config from '{self.new_special_tokens_config}': {e}"
                )
                raise

        elif self.add_special_tokens is not None:
            # Priority 2: Use simple comma-separated string (no descriptions)
            self.add_special_tokens = [token.strip() for token in self.add_special_tokens.split(",")]
            self._special_token_descriptions = None

        else:
            # No special tokens to add
            self._special_token_descriptions = None

        # Validate init method
        if self.init_special_tokens in ["desc_init", "desc_init_w_noise"]:
            if self._special_token_descriptions is None:
                logger.warning_rank0(
                    f"init_special_tokens='{self.init_special_tokens}' requires new_special_tokens_config. "
                    "Falling back to 'noise_init'"
                )
                self.init_special_tokens = "noise_init"


@dataclass
class QuantizationArguments:
    r"""Arguments pertaining to the quantization method."""

    quantization_method: QuantizationMethod = field(
        default=QuantizationMethod.BNB,
        metadata={"help": "Quantization method to use for on-the-fly quantization."},
    )
    quantization_bit: int | None = field(
        default=None,
        metadata={"help": "The number of bits to quantize the model using on-the-fly quantization."},
    )
    quantization_type: Literal["fp4", "nf4"] = field(
        default="nf4",
        metadata={"help": "Quantization data type to use in bitsandbytes int4 training."},
    )
    double_quantization: bool = field(
        default=True,
        metadata={"help": "Whether or not to use double quantization in bitsandbytes int4 training."},
    )
    quantization_device_map: Literal["auto"] | None = field(
        default=None,
        metadata={"help": "Device map used to infer the 4-bit quantized model, needs bitsandbytes>=0.43.0."},
    )


@dataclass
class ProcessorArguments:
    r"""Arguments pertaining to the image processor."""

    image_max_pixels: int = field(
        default=768 * 768,
        metadata={"help": "The maximum number of pixels of image inputs."},
    )
    image_min_pixels: int = field(
        default=32 * 32,
        metadata={"help": "The minimum number of pixels of image inputs."},
    )
    image_do_pan_and_scan: bool = field(
        default=False,
        metadata={"help": "Use pan and scan to process image for gemma3."},
    )
    crop_to_patches: bool = field(
        default=False,
        metadata={"help": "Whether to crop the image to patches for internvl."},
    )
    video_max_pixels: int = field(
        default=256 * 256,
        metadata={"help": "The maximum number of pixels of video inputs."},
    )
    video_min_pixels: int = field(
        default=16 * 16,
        metadata={"help": "The minimum number of pixels of video inputs."},
    )
    video_fps: float = field(
        default=2.0,
        metadata={"help": "The frames to sample per second for video inputs."},
    )
    video_maxlen: int = field(
        default=128,
        metadata={"help": "The maximum number of sampled frames for video inputs."},
    )
    use_audio_in_video: bool = field(
        default=False,
        metadata={"help": "Whether or not to use audio in video inputs."},
    )
    audio_sampling_rate: int = field(
        default=16000,
        metadata={"help": "The sampling rate of audio inputs."},
    )

    def __post_init__(self):
        if self.image_max_pixels < self.image_min_pixels:
            raise ValueError("`image_max_pixels` cannot be smaller than `image_min_pixels`.")

        if self.video_max_pixels < self.video_min_pixels:
            raise ValueError("`video_max_pixels` cannot be smaller than `video_min_pixels`.")


@dataclass
class ExportArguments:
    r"""Arguments pertaining to the model export."""

    export_dir: str | None = field(
        default=None,
        metadata={"help": "Path to the directory to save the exported model."},
    )
    export_size: int = field(
        default=5,
        metadata={"help": "The file shard size (in GB) of the exported model."},
    )
    export_device: Literal["cpu", "auto"] = field(
        default="cpu",
        metadata={"help": "The device used in model export, use `auto` to accelerate exporting."},
    )
    export_quantization_bit: int | None = field(
        default=None,
        metadata={"help": "The number of bits to quantize the exported model."},
    )
    export_quantization_dataset: str | None = field(
        default=None,
        metadata={"help": "Path to the dataset or dataset name to use in quantizing the exported model."},
    )
    export_quantization_nsamples: int = field(
        default=128,
        metadata={"help": "The number of samples used for quantization."},
    )
    export_quantization_maxlen: int = field(
        default=1024,
        metadata={"help": "The maximum length of the model inputs used for quantization."},
    )
    export_legacy_format: bool = field(
        default=False,
        metadata={"help": "Whether or not to save the `.bin` files instead of `.safetensors`."},
    )
    export_hub_model_id: str | None = field(
        default=None,
        metadata={"help": "The name of the repository if push the model to the Hugging Face hub."},
    )

    def __post_init__(self):
        if self.export_quantization_bit is not None and self.export_quantization_dataset is None:
            raise ValueError("Quantization dataset is necessary for exporting.")


@dataclass
class VllmArguments:
    r"""Arguments pertaining to the vLLM worker."""

    vllm_maxlen: int = field(
        default=4096,
        metadata={"help": "Maximum sequence (prompt + response) length of the vLLM engine."},
    )
    vllm_gpu_util: float = field(
        default=0.7,
        metadata={"help": "The fraction of GPU memory in (0,1) to be used for the vLLM engine."},
    )
    vllm_enforce_eager: bool = field(
        default=False,
        metadata={"help": "Whether or not to disable CUDA graph in the vLLM engine."},
    )
    vllm_max_lora_rank: int = field(
        default=32,
        metadata={"help": "Maximum rank of all LoRAs in the vLLM engine."},
    )
    vllm_config: dict | str | None = field(
        default=None,
        metadata={"help": "Config to initialize the vllm engine. Please use JSON strings."},
    )

    def __post_init__(self):
        if isinstance(self.vllm_config, str) and self.vllm_config.startswith("{"):
            self.vllm_config = _convert_str_dict(json.loads(self.vllm_config))


@dataclass
class SGLangArguments:
    r"""Arguments pertaining to the SGLang worker."""

    sglang_maxlen: int = field(
        default=4096,
        metadata={"help": "Maximum sequence (prompt + response) length of the SGLang engine."},
    )
    sglang_mem_fraction: float = field(
        default=0.7,
        metadata={"help": "The memory fraction (0-1) to be used for the SGLang engine."},
    )
    sglang_tp_size: int = field(
        default=-1,
        metadata={"help": "Tensor parallel size for the SGLang engine."},
    )
    sglang_config: dict | str | None = field(
        default=None,
        metadata={"help": "Config to initialize the SGLang engine. Please use JSON strings."},
    )
    sglang_lora_backend: Literal["triton", "flashinfer"] = field(
        default="triton",
        metadata={
            "help": "The backend of running GEMM kernels for Lora modules. Recommend using the Triton LoRA backend for better performance and stability."
        },
    )

    def __post_init__(self):
        if isinstance(self.sglang_config, str) and self.sglang_config.startswith("{"):
            self.sglang_config = _convert_str_dict(json.loads(self.sglang_config))


@dataclass
class KTransformersArguments:
    r"""Arguments pertaining to KTransformers AMX MoE SFT training.

    These fields are normalized into the transformers/accelerate KT config before training starts.
    """

    use_kt: bool = field(
        default=False,
        metadata={"help": "Whether to use KTransformers AMX MoE backend for SFT training."},
    )
    kt_cpu_activation: Literal["retain", "recompute"] | None = field(
        default=None,
        metadata={
            "help": (
                "Whether KTransformers retains CPU expert activations during GPU gradient checkpointing. "
                "Defaults to recompute while checkpointing is enabled and retain otherwise."
            )
        },
    )
    kt_weight_path: str | None = field(
        default=None,
        metadata={"help": "Path to pre-quantized INT8 expert weights (.kt files)."},
    )
    kt_non_expert_weight_path: str | None = field(
        default=None,
        metadata={
            "help": (
                "Path to a validated BF16 non-expert cache used with routed INT8 experts. "
                "This is a LLaMA-Factory loader setting, not a KT kernel option."
            )
        },
    )
    kt_expert_checkpoint_path: str | None = field(
        default=None,
        metadata={"help": "Path to expert checkpoint (safetensors) for online conversion."},
    )
    kt_use_lora_experts: bool | None = field(
        default=None,
        metadata={"help": "Whether to use GPU-side LoRA Experts."},
    )
    kt_lora_expert_num: int | None = field(
        default=None,
        metadata={"help": "Number of GPU-side LoRA Experts."},
    )
    kt_lora_expert_intermediate_size: int | None = field(
        default=None,
        metadata={"help": "Intermediate size for GPU-side LoRA Experts."},
    )
    kt_hf_config: Any = field(
        default=None,
        init=False,
        repr=False,
        metadata={"help": "Internal strong reference to the frozen Transformers KT config."},
    )
    _kt_resolved_config: dict[str, Any] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.kt_cpu_activation not in {None, "retain", "recompute"}:
            raise ValueError("`kt_cpu_activation` must be `retain` or `recompute`.")
        if not self.use_kt and self.kt_cpu_activation is not None:
            raise ValueError("`kt_cpu_activation` is only valid when `use_kt: true`.")

    def get_kt_activation_policy(self) -> dict[str, str]:
        r"""Resolve LF's GPU checkpoint switch and KT's CPU activation setting."""
        gpu_activation = "retain" if self.disable_gradient_checkpointing else "recompute"
        cpu_activation = self.kt_cpu_activation
        if cpu_activation is None:
            cpu_activation = "retain" if gpu_activation == "retain" else "recompute"

        if cpu_activation == "recompute" and gpu_activation == "retain":
            raise ValueError(
                "`kt_cpu_activation: recompute` requires GPU gradient checkpointing. "
                "Set `disable_gradient_checkpointing: false` or use `kt_cpu_activation: retain`."
            )

        return {"cpu": cpu_activation, "gpu": gpu_activation}

    _KT_ADVANCED_KEYS = frozenset(
        {
            "kt_backend",
            "kt_expert_weight_format",
            "kt_force_fused_expert_lora",
            "kt_max_cache_depth",
            "kt_model_max_length",
            "kt_num_gpu_experts",
            "kt_num_threads",
            "kt_share_backward_bb",
            "kt_threadpool_count",
            "kt_tp_enabled",
            "kt_weight_lifecycle",
        }
    )
    _KT_FORBIDDEN_DERIVED_KEYS = frozenset(
        {
            "enabled",
            "kt_activation_policy",
            "kt_expert_checkpoint_path",
            "kt_full_weight_grad",
            "kt_lora_alpha",
            "kt_lora_dropout",
            "kt_lora_expert_intermediate_size",
            "kt_lora_expert_num",
            "kt_lora_rank",
            "kt_non_expert_weight_path",
            "kt_skip_expert_loading",
            "kt_train_mode",
            "kt_use_lora_experts",
            "kt_weight_path",
        }
    )
    _KT_PLUGIN_KEYS = frozenset(
        {
            "allowed_distributed_types",
            "bypass_device_map_check",
            "kt_config",
            "require_single_process",
            "skip_device_placement",
        }
    )

    @staticmethod
    def _accelerator_kt_config(training_args: Any) -> Any:
        accelerator_config = getattr(training_args, "accelerator_config", None)
        if isinstance(accelerator_config, dict):
            return accelerator_config.get("kt_config")
        return getattr(accelerator_config, "kt_config", None)

    def _resolve_advanced_kt_config(self, training_args: Any) -> dict[str, Any]:
        raw_config = getattr(training_args, "kt_config", None)
        accelerator_config = self._accelerator_kt_config(training_args)
        reapplied_plugin = (
            bool(self._kt_resolved_config)
            and isinstance(accelerator_config, dict)
            and accelerator_config.get("enabled") is True
            and accelerator_config.get("kt_config") == self._kt_resolved_config
            and set(accelerator_config) == {"enabled", "kt_config"}
        )
        if reapplied_plugin:
            accelerator_config = None
        if raw_config is None and accelerator_config is not None:
            raise ValueError(
                "Put KTransformers settings in the LLaMA-Factory training YAML `kt_config`; "
                "remove `kt_config` from the Accelerate config."
            )
        if raw_config is None:
            raw_config = {}
        if not isinstance(raw_config, dict):
            raise TypeError("LLaMA-Factory `kt_config` must be a flat mapping, not a file path or nested config.")

        # Frozen Transformers injects these two defaults into the original dict
        # during TrainingArguments.__post_init__. They are LF-owned constants,
        # not user-facing runtime settings.
        config = dict(raw_config)
        for key in ("enabled", "kt_skip_expert_loading"):
            if key in config and config.pop(key) is not True:
                raise ValueError(f"`{key}` is derived by LLaMA-Factory and must not be overridden.")

        nested_or_plugin = sorted(set(config) & self._KT_PLUGIN_KEYS)
        if nested_or_plugin:
            raise ValueError(
                "LLaMA-Factory `kt_config` contains Accelerate plugin fields; "
                f"use a flat KT kernel mapping instead: {nested_or_plugin}."
            )
        derived = sorted(set(config) & self._KT_FORBIDDEN_DERIVED_KEYS)
        if derived:
            raise ValueError(
                f"These `kt_config` values are owned by LLaMA-Factory top-level or finetuning arguments: {derived}."
            )
        unknown = sorted(set(config) - self._KT_ADVANCED_KEYS)
        if unknown:
            raise ValueError(f"Unsupported LLaMA-Factory `kt_config` fields: {unknown}.")

        if (
            accelerator_config is not None
            and accelerator_config is not raw_config
            and accelerator_config != raw_config
        ):
            raise ValueError(
                "AcceleratorConfig.kt_config is not an independent KT configuration source; "
                "remove KT settings from the Accelerate config."
            )

        capacity = config.get("kt_model_max_length")
        if capacity is not None:
            if isinstance(capacity, bool) or not isinstance(capacity, (int, str)):
                raise ValueError("`kt_model_max_length` must be a positive integer.")
            try:
                capacity = int(capacity)
            except (TypeError, ValueError) as exc:
                raise ValueError("`kt_model_max_length` must be a positive integer.") from exc
            if capacity <= 0:
                raise ValueError("`kt_model_max_length` must be a positive integer.")
            config["kt_model_max_length"] = capacity

        return config

    def configure_kt_checkpointing(self, training_args: Any) -> None:
        r"""Keep LLaMA-Factory as the only gradient-checkpointing entry point for KT."""
        if not self.use_kt:
            return

        if self.use_unsloth:
            raise ValueError("KTransformers does not support `use_unsloth` checkpoint wrapping.")

        if self.use_unsloth_gc:
            raise ValueError("KTransformers does not support `use_unsloth_gc`.")

        if getattr(training_args, "gradient_checkpointing", False):
            raise ValueError(
                "KTransformers uses LLaMA-Factory's `disable_gradient_checkpointing` setting. "
                "Remove `gradient_checkpointing: true` to avoid installing a second checkpoint wrapper."
            )

        if getattr(training_args, "gradient_checkpointing_kwargs", None) is not None:
            raise ValueError(
                "KTransformers manages its checkpoint context internally; remove `gradient_checkpointing_kwargs`."
            )

        fsdp_config = getattr(training_args, "fsdp_config", None)
        if isinstance(fsdp_config, dict) and fsdp_config.get("activation_checkpointing"):
            raise ValueError(
                "KTransformers is incompatible with FSDP activation checkpointing. "
                "Use LLaMA-Factory's `disable_gradient_checkpointing` setting instead."
            )

        if os.environ.get("FSDP_ACTIVATION_CHECKPOINTING", "false").lower() in {"1", "true", "yes"}:
            raise ValueError(
                "KTransformers is incompatible with FSDP activation checkpointing. "
                "Disable `fsdp_activation_checkpointing` in the Accelerate config."
            )

        legacy_sources = [
            name
            for name in ("ACCELERATE_KT_ACTIVATION_POLICY", "KT_REUSE_CHECKPOINT_FORWARD")
            if os.environ.get(name) not in {None, ""}
        ]
        if legacy_sources:
            raise ValueError(
                "`kt_cpu_activation` is the KTransformers activation source in LLaMA-Factory; "
                f"remove legacy environment settings {legacy_sources}."
            )

        external_runtime_env = sorted(
            name for name, value in os.environ.items() if name.startswith("ACCELERATE_KT_") and value != ""
        )
        if external_runtime_env:
            raise ValueError(
                "The LLaMA-Factory training YAML is the only KTransformers configuration source; "
                f"remove external runtime settings {external_runtime_env}."
            )

        self.get_kt_activation_policy()  # validate the resolved combination
        if not self.disable_gradient_checkpointing:
            self.use_reentrant_gc = False

        training_args.gradient_checkpointing = False
        training_args.gradient_checkpointing_kwargs = None

    def get_kt_config_dict(
        self,
        finetuning_args: Any,
        model_max_length: int | None,
        advanced_config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        r"""Build KT config values from LLaMA-Factory model and LoRA arguments."""
        advanced_config = dict(advanced_config or {})
        configured_capacity = advanced_config.get("kt_model_max_length")
        kt_model_max_length = max(model_max_length or 0, configured_capacity or 0) or None
        finetuning_type = getattr(finetuning_args, "finetuning_type", "lora")
        train_mode = {"full": "full", "freeze": "hybrid", "lora": "lora"}.get(finetuning_type)
        if train_mode is None:
            raise ValueError(f"KTransformers does not support `finetuning_type: {finetuning_type}`.")

        kt_config = {
            **advanced_config,
            "kt_lora_rank": getattr(finetuning_args, "lora_rank", None),
            "kt_lora_alpha": getattr(finetuning_args, "lora_alpha", None),
            "kt_lora_dropout": getattr(finetuning_args, "lora_dropout", None),
            "kt_weight_path": self.kt_weight_path,
            "kt_expert_checkpoint_path": self.kt_expert_checkpoint_path,
            "kt_model_max_length": kt_model_max_length,
            "kt_use_lora_experts": self.kt_use_lora_experts,
            "kt_lora_expert_num": self.kt_lora_expert_num,
            "kt_lora_expert_intermediate_size": self.kt_lora_expert_intermediate_size,
            "kt_activation_policy": self.get_kt_activation_policy(),
            "kt_train_mode": train_mode,
            "kt_full_weight_grad": train_mode in {"full", "hybrid"},
        }
        if train_mode == "full":
            for key in ("kt_lora_rank", "kt_lora_alpha", "kt_lora_dropout"):
                kt_config.pop(key, None)

        weight_format = kt_config.get("kt_expert_weight_format")
        if weight_format is not None:
            weight_format = str(weight_format).strip().lower()
            if weight_format not in {"bf16", "int8", "fp8"}:
                raise ValueError("`kt_expert_weight_format` must be `bf16`, `int8`, or `fp8`.")
            kt_config["kt_expert_weight_format"] = weight_format
        if weight_format == "int8":
            if not self.kt_weight_path:
                raise ValueError("Routed INT8 training requires top-level `kt_weight_path`.")
            if not self.kt_non_expert_weight_path:
                raise ValueError("Routed INT8 training requires top-level `kt_non_expert_weight_path`.")
            if train_mode != "lora":
                raise ValueError("Routed INT8 training supports frozen-base LoRA only.")
            kt_config.setdefault("kt_backend", "auto")
            kt_config.setdefault("kt_weight_lifecycle", "persistent")
            if kt_config["kt_weight_lifecycle"] != "persistent":
                raise ValueError("Routed INT8 training requires `kt_weight_lifecycle: persistent`.")
        elif self.kt_non_expert_weight_path:
            raise ValueError("`kt_non_expert_weight_path` is only valid with `kt_expert_weight_format: int8`.")

        return {key: value for key, value in kt_config.items() if value is not None}

    def apply_kt_config(self, finetuning_args: Any, training_args: Any, model_max_length: int | None) -> None:
        r"""Apply LLaMA-Factory KT args to transformers/accelerate KT integration points."""
        if not self.use_kt:
            return

        self.configure_kt_checkpointing(training_args)
        advanced_config = self._resolve_advanced_kt_config(training_args)
        kt_config = self.get_kt_config_dict(finetuning_args, model_max_length, advanced_config)

        hf_kt = getattr(training_args, "hf_kt_config", None)
        if hf_kt is None or not hasattr(hf_kt, "_kt_config") or not isinstance(hf_kt._kt_config, dict):
            try:
                from transformers.integrations.kt import HfTrainerKTConfig
            except (ImportError, ModuleNotFoundError) as exc:
                raise RuntimeError(
                    "The installed Transformers build does not provide the KTransformers integration."
                ) from exc

            hf_kt = HfTrainerKTConfig(kt_config)
            training_args.hf_kt_config = hf_kt
        hf_kt._kt_config = kt_config
        hf_kt._llamafactory_authoritative = True
        self.kt_hf_config = hf_kt
        self._kt_resolved_config = dict(kt_config)
        training_args.kt_config = dict(advanced_config)
        os.environ["ACCELERATE_USE_KT"] = "true"

        accelerator_config = getattr(training_args, "accelerator_config", None)
        plugin_config = {"enabled": True, "kt_config": kt_config}
        if isinstance(accelerator_config, dict):
            accelerator_config["kt_config"] = plugin_config
        elif accelerator_config is not None:
            accelerator_config.kt_config = plugin_config


@dataclass
class ModelArguments(
    SGLangArguments,
    VllmArguments,
    KTransformersArguments,
    ExportArguments,
    ProcessorArguments,
    QuantizationArguments,
    BaseModelArguments,
):
    r"""Arguments pertaining to which model/config/tokenizer we are going to fine-tune or infer.

    The class on the most right will be displayed first.
    """

    compute_dtype: torch.dtype | None = field(
        default=None,
        init=False,
        metadata={"help": "Torch data type for computing model outputs, derived from `fp/bf16`. Do not specify it."},
    )
    device_map: str | dict[str, Any] | None = field(
        default=None,
        init=False,
        metadata={"help": "Device map for model placement, derived from training stage. Do not specify it."},
    )
    model_max_length: int | None = field(
        default=None,
        init=False,
        metadata={"help": "The maximum input length for model, derived from `cutoff_len`. Do not specify it."},
    )
    block_diag_attn: bool = field(
        default=False,
        init=False,
        metadata={"help": "Whether use block diag attention or not, derived from `neat_packing`. Do not specify it."},
    )

    def __post_init__(self):
        BaseModelArguments.__post_init__(self)
        ProcessorArguments.__post_init__(self)
        ExportArguments.__post_init__(self)
        VllmArguments.__post_init__(self)
        SGLangArguments.__post_init__(self)
        KTransformersArguments.__post_init__(self)

    @classmethod
    def copyfrom(cls, source: "Self", **kwargs) -> "Self":
        init_args, lazy_args = {}, {}
        for attr in fields(source):
            if attr.init:
                init_args[attr.name] = getattr(source, attr.name)
            else:
                lazy_args[attr.name] = getattr(source, attr.name)

        init_args.update(kwargs)
        result = cls(**init_args)
        for name, value in lazy_args.items():
            setattr(result, name, value)

        return result

    def to_dict(self) -> dict[str, Any]:
        args = asdict(self)
        args = {k: f"<{k.upper()}>" if k.endswith("token") else v for k, v in args.items()}
        return args
