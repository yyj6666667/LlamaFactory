# Copyright 2026 the LlamaFactory team.
# Licensed under the Apache License, Version 2.0.

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


SOURCE = Path(__file__).resolve().parents[2] / "src/llamafactory/model/model_utils/deepseek_v4.py"
SPEC = importlib.util.spec_from_file_location("v4_recipe", SOURCE)
recipe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(recipe)


def arguments():
    return {
        "config": SimpleNamespace(model_type="deepseek_v4"),
        "model_args": SimpleNamespace(use_kt=True, flash_attn="disabled", kt_non_expert_weight_path="/cache/v4"),
        "data_args": SimpleNamespace(
            template="deepseek4_nothink",
            packing=False,
            neat_packing=False,
            enable_thinking=False,
            tool_format=None,
            cutoff_len=1024,
        ),
        "training_args": SimpleNamespace(
            bf16=True,
            bf16_full_eval=False,
            predict_with_generate=False,
            kt_config={"kt_expert_weight_format": "mxfp4"},
        ),
        "finetuning_args": SimpleNamespace(
            stage="sft",
            finetuning_type="lora",
            pure_bf16=False,
            lora_rank=8,
            lora_alpha=16,
            lora_dropout=0,
            lora_target=sorted(recipe.V4_LORA_TARGETS),
            additional_target=None,
        ),
    }


def test_supported_recipe_and_explicit_target_subset():
    args = arguments()
    recipe.validate_kt_deepseek_v4_training(**args)
    args["finetuning_args"].lora_target = ["self_attn.q_a_proj"]
    recipe.validate_kt_deepseek_v4_training(**args)


@pytest.mark.parametrize(
    "group,key,value,match",
    [
        ("data_args", "template", "deepseek3", "deepseek4_nothink"),
        ("data_args", "packing", True, "packing"),
        ("data_args", "neat_packing", True, "packing"),
        ("data_args", "enable_thinking", True, "non-thinking"),
        ("data_args", "cutoff_len", 4096, "1024"),
        ("training_args", "bf16", False, "autocast"),
        ("training_args", "bf16_full_eval", True, "autocast"),
        ("training_args", "predict_with_generate", True, "SGLang"),
        ("finetuning_args", "pure_bf16", True, "autocast"),
        ("finetuning_args", "lora_rank", 16, "lora_rank"),
        ("finetuning_args", "lora_alpha", 8, "lora_alpha"),
        ("finetuning_args", "lora_dropout", 0.1, "lora_dropout"),
        ("finetuning_args", "lora_target", ["all"], "explicit LoRA"),
        ("finetuning_args", "lora_target", ["self_attn.o_a_proj"], "explicit LoRA"),
        ("finetuning_args", "additional_target", ["lm_head"], "additional_target"),
        ("model_args", "kt_non_expert_weight_path", None, "kt_non_expert_weight_path"),
        ("model_args", "flash_attn", "fa2", "eager"),
        ("training_args", "kt_config", {}, "mxfp4"),
    ],
)
def test_unsupported_recipe_fails_before_model_loading(group, key, value, match):
    args = arguments()
    setattr(args[group], key, value)
    with pytest.raises(ValueError, match=match):
        recipe.validate_kt_deepseek_v4_training(**args)


@pytest.mark.parametrize("model_type", ["qwen3_5_moe", "kimi_k25", "deepseek_v3"])
def test_other_models_keep_their_existing_recipe(model_type):
    args = arguments()
    args["config"].model_type = model_type
    args["data_args"].packing = True
    args["finetuning_args"].lora_target = ["all"]
    recipe.validate_kt_deepseek_v4_training(**args)
