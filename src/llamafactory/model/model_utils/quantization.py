# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's Transformers and Optimum library.
# https://github.com/huggingface/transformers/blob/v4.41.0/src/transformers/utils/quantization_config.py
# https://github.com/huggingface/optimum/blob/v1.20.0/optimum/gptq/data.py
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
import random
from functools import wraps
from inspect import Parameter, signature
from typing import TYPE_CHECKING, Any

import torch
from datasets import load_dataset
from packaging.version import Version
from transformers import BitsAndBytesConfig, EetqConfig, GPTQConfig, HqqConfig
from transformers.integrations import is_deepspeed_zero3_enabled
from transformers.modeling_utils import is_fsdp_enabled

from ...extras import logging
from ...extras.constants import FILEEXT2TYPE, QuantizationMethod
from ...extras.misc import check_version, get_current_device


if TYPE_CHECKING:
    from transformers import PretrainedConfig, PreTrainedTokenizer

    from ...hparams import ModelArguments


logger = logging.get_logger(__name__)


def _patch_deepseek_remote_code_compatibility() -> None:
    r"""Restore the Transformers 4.x availability helper used by DeepSeek remote code."""
    import transformers
    from transformers.utils import import_utils

    if hasattr(import_utils, "is_torch_fx_available"):
        return

    transformers_version = Version(transformers.__version__)
    if not (Version("5.0.0") <= transformers_version < Version("6.0.0")):
        raise RuntimeError(
            "DeepSeek-V3 remote-code compatibility requires Transformers 5.x when "
            "`is_torch_fx_available` is absent, got Transformers "
            f"{transformers.__version__}."
        )

    is_torch_available = getattr(import_utils, "is_torch_available", None)
    if not callable(is_torch_available):
        raise RuntimeError(
            "Cannot provide DeepSeek-V3 remote-code compatibility because Transformers "
            "does not expose `is_torch_available`."
        )

    import_utils.is_torch_fx_available = is_torch_available


def _patch_fp8_partial_block_dequantization() -> None:
    from transformers.integrations.finegrained_fp8 import Fp8Dequantize

    if getattr(Fp8Dequantize, "_llamafactory_partial_block_support", False):
        return

    original_convert = Fp8Dequantize.convert
    convert_signature = signature(original_convert)
    expected_parameters = {"self", "input_dict", "full_layer_name"}
    if not expected_parameters.issubset(convert_signature.parameters) or not any(
        parameter.kind == Parameter.VAR_KEYWORD for parameter in convert_signature.parameters.values()
    ):
        raise RuntimeError(
            "Unsupported Transformers `Fp8Dequantize.convert` signature for partial-block dequantization: "
            f"{convert_signature}."
        )

    @wraps(original_convert)
    def convert(self, input_dict, full_layer_name=None, **kwargs):
        output_dtype = getattr(
            self.hf_quantizer.quantization_config,
            "_llamafactory_dequantization_dtype",
            None,
        )

        def cast_output(converted):
            if output_dtype is None:
                return converted

            def cast_tensor(tensor):
                if isinstance(tensor, list):
                    return [item.to(output_dtype) for item in tensor]
                if isinstance(tensor, tuple):
                    return tuple(item.to(output_dtype) for item in tensor)

                return tensor.to(output_dtype)

            return {name: cast_tensor(tensor) for name, tensor in converted.items()}

        quantized_values = input_dict.get("weight$")
        scale_values = input_dict.get("weight_scale_inv")
        if quantized_values is None or scale_values is None:
            return cast_output(original_convert(self, input_dict, full_layer_name=full_layer_name, **kwargs))

        quantized = quantized_values[0] if isinstance(quantized_values, list) else quantized_values
        scales = scale_values[0] if isinstance(scale_values, list) else scale_values
        rows, cols = quantized.shape[-2:]
        block_size = self.hf_quantizer.quantization_config.weight_block_size
        if block_size is None:
            return cast_output(original_convert(self, input_dict, full_layer_name=full_layer_name, **kwargs))

        block_m, block_n = block_size
        scale_shape = ((rows + block_m - 1) // block_m, (cols + block_n - 1) // block_n)
        if rows % block_m == 0 and cols % block_n == 0:
            return cast_output(original_convert(self, input_dict, full_layer_name=full_layer_name, **kwargs))

        if scales.shape[-2:] != scale_shape:
            raise ValueError(
                f"FP8 scale dimensions {tuple(scales.shape[-2:])} do not match the expected partial-block "
                f"shape {scale_shape} for matrix dimensions ({rows}, {cols}) and block sizes ({block_m}, {block_n})."
            )

        expanded_scales = scales.repeat_interleave(block_m, dim=-2)
        expanded_scales = expanded_scales.repeat_interleave(block_n, dim=-1)[..., :rows, :cols]
        dequantized = quantized.to(scales.dtype) * expanded_scales
        if output_dtype is not None:
            dequantized = dequantized.to(output_dtype)

        return {full_layer_name: dequantized}

    Fp8Dequantize.convert = convert
    Fp8Dequantize._llamafactory_partial_block_support = True


def _get_quantization_dataset(tokenizer: "PreTrainedTokenizer", model_args: "ModelArguments") -> list[dict[str, Any]]:
    r"""Prepare the tokenized dataset to perform AutoGPTQ. Do not use tensor output for JSON serialization."""
    if os.path.isfile(model_args.export_quantization_dataset):
        data_path = FILEEXT2TYPE.get(model_args.export_quantization_dataset.split(".")[-1], None)
        data_files = model_args.export_quantization_dataset
    else:
        data_path = model_args.export_quantization_dataset
        data_files = None

    dataset = load_dataset(
        path=data_path,
        data_files=data_files,
        split="train",
        cache_dir=model_args.cache_dir,
        token=model_args.hf_hub_token,
    )

    samples = []
    maxlen = model_args.export_quantization_maxlen
    for _ in range(model_args.export_quantization_nsamples):
        n_try = 0
        while True:
            if n_try > 100:
                raise ValueError("Cannot find satisfying example, considering decrease `export_quantization_maxlen`.")

            sample_idx = random.randint(0, len(dataset) - 1)
            sample: dict[str, torch.Tensor] = tokenizer(dataset[sample_idx]["text"], return_tensors="pt")
            n_try += 1
            if sample["input_ids"].size(1) > maxlen:
                break  # TODO: fix large maxlen

        word_idx = random.randint(0, sample["input_ids"].size(1) - maxlen - 1)
        input_ids = sample["input_ids"][:, word_idx : word_idx + maxlen]
        attention_mask = sample["attention_mask"][:, word_idx : word_idx + maxlen]
        samples.append({"input_ids": input_ids.tolist(), "attention_mask": attention_mask.tolist()})

    return samples


def configure_quantization(
    config: "PretrainedConfig",
    tokenizer: "PreTrainedTokenizer",
    model_args: "ModelArguments",
    is_trainable: bool,
    init_kwargs: dict[str, Any],
) -> None:
    r"""Priority: PTQ-quantized (train/infer) > AutoGPTQ (export) > On-the-fly quantization (train/infer)."""
    if getattr(config, "quantization_config", None):  # ptq
        if model_args.quantization_bit is not None:
            logger.warning_rank0("`quantization_bit` will not affect on the PTQ-quantized models.")

        quantization_config: dict[str, Any] = getattr(config, "quantization_config", None)
        quant_method = quantization_config.get("quant_method", "")

        kt_backend = os.environ.get("ACCELERATE_KT_BACKEND", "")
        kt_kgroup_path = (
            getattr(model_args, "use_kt", False)
            and getattr(model_args, "kt_weight_path", None)
            and quant_method == "compressed-tensors"
            and "KGroup" in kt_backend
        )
        if (
            quant_method not in (QuantizationMethod.MXFP4, QuantizationMethod.FP8)
            and (is_deepspeed_zero3_enabled() or is_fsdp_enabled())
            and not kt_kgroup_path
        ):
            # mxfp4 will dequant the model weights
            raise ValueError("DeepSpeed ZeRO-3 or FSDP is incompatible with PTQ-quantized models.")
        if kt_kgroup_path:
            logger.info_rank0("Allowing compressed-tensors PTQ config under FSDP for KTransformers KGroup SFT.")
            if getattr(model_args, "kt_text_only_sft", False) and getattr(config, "model_type", None) == "kimi_k25":
                ignore = quantization_config.setdefault("ignore", [])
                for pattern in ["re:.*vision_tower.*", "re:.*mm_projector.*"]:
                    if pattern not in ignore:
                        ignore.append(pattern)

        if quant_method == QuantizationMethod.MXFP4:
            from transformers import Mxfp4Config

            quant_config = Mxfp4Config(dequantize=True)
            init_kwargs["quantization_config"] = quant_config
            init_kwargs["ignore_mismatched_sizes"] = True

        if quant_method == QuantizationMethod.FP8:
            from transformers import FineGrainedFP8Config

            is_kt_int8_deepseek = (
                getattr(model_args, "use_kt", False)
                and getattr(model_args, "kt_expert_weight_format", None) == "int8"
                and getattr(config, "model_type", None) == "deepseek_v3"
            )
            if is_kt_int8_deepseek:
                _patch_deepseek_remote_code_compatibility()
                _patch_fp8_partial_block_dequantization()

            quant_config = FineGrainedFP8Config(dequantize=True)
            if is_kt_int8_deepseek:
                quant_config._llamafactory_dequantization_dtype = torch.bfloat16

            init_kwargs["quantization_config"] = quant_config
            init_kwargs["ignore_mismatched_sizes"] = True

        if quant_method == QuantizationMethod.GPTQ:
            check_version("gptqmodel>=2.0.0", mandatory=True)
            quantization_config.pop("disable_exllama", None)  # remove deprecated args
            quantization_config["use_exllama"] = False  # disable exllama

        if quant_method == QuantizationMethod.AWQ:
            check_version("autoawq", mandatory=True)

        if quant_method == QuantizationMethod.AQLM:
            check_version("aqlm>=1.1.0", mandatory=True)
            quantization_config["bits"] = 2

        quant_bits = quantization_config.get("bits", "?")
        logger.info_rank0(f"Loading {quant_bits}-bit {quant_method.upper()}-quantized model.")

    elif model_args.export_quantization_bit is not None:  # gptqmodel
        if model_args.export_quantization_bit not in [8, 4, 3, 2]:
            raise ValueError("AutoGPTQ only accepts 2/3/4/8-bit quantization.")

        check_version("optimum>=1.24.0", mandatory=True)
        check_version("gptqmodel>=2.0.0", mandatory=True)
        from accelerate.utils import get_max_memory

        if getattr(config, "model_type", None) == "chatglm":
            raise ValueError("ChatGLM model is not supported yet.")

        try:
            from optimum.gptq import utils as gq_utils

            if "language_model.model.layers" not in gq_utils.BLOCK_PATTERNS:
                gq_utils.BLOCK_PATTERNS.insert(0, "language_model.model.layers")
        except ImportError:
            pass

        block_name_to_quantize = None
        if getattr(config, "model_type", None) in ["gemma3", "paligemma"]:
            block_name_to_quantize = "language_model.model.layers"

        init_kwargs["quantization_config"] = GPTQConfig(
            bits=model_args.export_quantization_bit,
            tokenizer=tokenizer,
            dataset=_get_quantization_dataset(tokenizer, model_args),
            block_name_to_quantize=block_name_to_quantize,
        )
        init_kwargs["device_map"] = "auto"
        init_kwargs["max_memory"] = get_max_memory()
        model_args.compute_dtype = torch.float16  # force fp16 for gptqmodel
        logger.info_rank0(f"Quantizing model to {model_args.export_quantization_bit} bit with GPTQModel.")

    elif model_args.quantization_bit is not None:  # on-the-fly
        if model_args.quantization_method == QuantizationMethod.BNB:
            if model_args.quantization_bit == 8:
                check_version("bitsandbytes>=0.37.0", mandatory=True)
                init_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            elif model_args.quantization_bit == 4:
                check_version("bitsandbytes>=0.39.0", mandatory=True)
                init_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=model_args.compute_dtype,
                    bnb_4bit_use_double_quant=model_args.double_quantization,
                    bnb_4bit_quant_type=model_args.quantization_type,
                    bnb_4bit_quant_storage=model_args.compute_dtype,  # crucial for fsdp+qlora
                )
            else:
                raise ValueError("Bitsandbytes only accepts 4-bit or 8-bit quantization.")

            # Do not assign device map if:
            # 1. deepspeed zero3 or fsdp (train)
            # 2. auto quantization device map (inference)
            if is_deepspeed_zero3_enabled() or is_fsdp_enabled() or model_args.quantization_device_map == "auto":
                if model_args.quantization_bit != 4:
                    raise ValueError("Only 4-bit quantized model can use fsdp+qlora or auto device map.")

                check_version("bitsandbytes>=0.43.0", mandatory=True)
            else:
                init_kwargs["device_map"] = {"": get_current_device()}  # change auto device map for inference

            logger.info_rank0(f"Quantizing model to {model_args.quantization_bit} bit with bitsandbytes.")
        elif model_args.quantization_method == QuantizationMethod.HQQ:
            if model_args.quantization_bit not in [8, 6, 5, 4, 3, 2, 1]:
                raise ValueError("HQQ only accepts 1/2/3/4/5/6/8-bit quantization.")

            if is_deepspeed_zero3_enabled() or is_fsdp_enabled():
                raise ValueError("HQQ quantization is incompatible with DeepSpeed ZeRO-3 or FSDP.")

            check_version("hqq", mandatory=True)
            init_kwargs["quantization_config"] = HqqConfig(
                nbits=model_args.quantization_bit, quant_zero=False, quant_scale=False, axis=0
            )  # use ATEN kernel (axis=0) for performance
            logger.info_rank0(f"Quantizing model to {model_args.quantization_bit} bit with HQQ.")
        elif model_args.quantization_method == QuantizationMethod.EETQ:
            if model_args.quantization_bit != 8:
                raise ValueError("EETQ only accepts 8-bit quantization.")

            if is_deepspeed_zero3_enabled() or is_fsdp_enabled():
                raise ValueError("EETQ quantization is incompatible with DeepSpeed ZeRO-3 or FSDP.")

            check_version("eetq", mandatory=True)
            init_kwargs["quantization_config"] = EetqConfig()
            logger.info_rank0(f"Quantizing model to {model_args.quantization_bit} bit with EETQ.")
