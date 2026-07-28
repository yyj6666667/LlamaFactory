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

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from llamafactory.hparams import model_args as model_args_module
from llamafactory.hparams.model_args import ModelArguments
from llamafactory.hparams.parser import _validate_kt_activation_policy_source


def _finetuning_args(**kwargs):
    values = {
        "stage": "sft",
        "finetuning_type": "lora",
        "lora_rank": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.0,
        "lora_target": ["all"],
        "pure_bf16": True,
    }
    values.update(kwargs)
    return SimpleNamespace(**values)


def test_deepseek_int8_example_is_two_gpu_acceptance_config():
    example_path = (
        Path(__file__).parents[2] / "examples" / "ktransformers" / "train_lora" / "deepseek_v3_lora_sft_kt.yaml"
    )
    config = yaml.safe_load(example_path.read_text())

    assert config["model_name_or_path"] == "/mnt/data/models/DeepSeek-V3.1"
    assert config["kt_expert_weight_format"] == "int8"
    assert config["kt_weight_path"] == "/mnt/data/models/kt-int8-dsv31-20260728-int8"
    assert config["kt_weight_lifecycle"] == "persistent"
    assert config["kt_backend"] == "AMXINT8"
    assert config["kt_tp_enabled"] is True
    assert config["kt_threadpool_count"] == 2
    assert config["kt_num_gpu_experts"] == 0
    assert config["kt_share_backward_bb"] is True
    assert config["kt_force_fused_expert_lora"] is True
    assert config["activation_policy"] == {"cpu": "retain", "gpu": "recompute"}
    assert (config["lora_rank"], config["lora_alpha"], config["lora_dropout"]) == (8, 16, 0)
    assert config["pure_bf16"] is True
    assert (config["cutoff_len"], config["per_device_train_batch_size"]) == (1024, 1)
    assert (config["gradient_accumulation_steps"], config["max_steps"]) == (1, 3)


def test_int8_config_is_forwarded_from_single_yaml_entry(monkeypatch: pytest.MonkeyPatch):
    checkpoint_context = object()
    monkeypatch.setattr(model_args_module.os, "environ", {})
    monkeypatch.setattr(
        model_args_module,
        "_get_kt_activation_checkpoint_context_fn",
        lambda: checkpoint_context,
    )
    model_args = ModelArguments(
        model_name_or_path="dummy",
        use_kt=True,
        kt_expert_weight_format="int8",
        kt_weight_path="/tmp/int8-weights",
        kt_weight_lifecycle="persistent",
        activation_policy={"cpu": "retain", "gpu": "recompute"},
    )
    finetuning_args = _finetuning_args()
    training_args = SimpleNamespace(
        bf16=True,
        gradient_checkpointing=False,
        gradient_checkpointing_kwargs=None,
        hf_kt_config=SimpleNamespace(_kt_config={}),
    )

    model_args.apply_kt_config(finetuning_args, training_args, model_max_length=1024)

    assert training_args.hf_kt_config._kt_config["kt_expert_weight_format"] == "int8"
    assert training_args.hf_kt_config._kt_config["kt_weight_path"] == "/tmp/int8-weights"
    assert training_args.hf_kt_config._kt_config["kt_weight_lifecycle"] == "persistent"
    assert training_args.hf_kt_config._kt_config["kt_backend"] == "AMXINT8"
    assert training_args.hf_kt_config._kt_config["kt_tp_enabled"] is True
    assert training_args.hf_kt_config._kt_config["kt_threadpool_count"] == 2
    assert training_args.hf_kt_config._kt_config["kt_num_gpu_experts"] == 0
    assert training_args.hf_kt_config._kt_config["kt_share_backward_bb"] is True
    assert training_args.hf_kt_config._kt_config["kt_force_fused_expert_lora"] is True
    assert model_args_module.os.environ["ACCELERATE_KT_BACKEND"] == "AMXINT8"
    assert model_args_module.os.environ["ACCELERATE_KT_TP_ENABLED"] == "True"
    assert model_args_module.os.environ["ACCELERATE_KT_THREADPOOL_COUNT"] == "2"
    assert model_args_module.os.environ["ACCELERATE_KT_NUM_GPU_EXPERTS"] == "0"
    assert model_args_module.os.environ["ACCELERATE_KT_SHARE_BACKWARD_BB"] == "True"
    assert model_args_module.os.environ["ACCELERATE_KT_FORCE_FUSED_EXPERT_LORA"] == "True"
    assert training_args.hf_kt_config._kt_config["kt_activation_policy"] == {
        "cpu": "retain",
        "gpu": "recompute",
    }
    assert model_args.kt_activation_checkpoint_context_fn is checkpoint_context
    assert training_args.gradient_checkpointing is False
    assert training_args.gradient_checkpointing_kwargs is None


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("kt_backend", "AMXBF16"),
        ("kt_tp_enabled", False),
        ("kt_threadpool_count", 1),
        ("kt_num_gpu_experts", 1),
        ("kt_share_backward_bb", False),
        ("kt_force_fused_expert_lora", False),
    ],
)
def test_int8_rejects_conflicting_accelerate_runtime(name: str, value):
    model_args = ModelArguments(
        model_name_or_path="dummy",
        use_kt=True,
        kt_expert_weight_format="int8",
        kt_weight_path="/tmp/int8-weights",
        activation_policy={"cpu": "retain", "gpu": "recompute"},
    )
    training_args = SimpleNamespace(
        bf16=True,
        gradient_checkpointing=False,
        gradient_checkpointing_kwargs=None,
        hf_kt_config=SimpleNamespace(_kt_config={name: value}),
    )

    with pytest.raises(ValueError, match=f"Accelerate `{name}"):
        model_args.apply_kt_config(_finetuning_args(), training_args, model_max_length=1024)


def test_int8_requires_bf16_training():
    model_args = ModelArguments(
        model_name_or_path="dummy",
        use_kt=True,
        kt_expert_weight_format="int8",
        kt_weight_path="/tmp/int8-weights",
        activation_policy={"cpu": "retain", "gpu": "recompute"},
    )
    training_args = SimpleNamespace(
        bf16=False,
        gradient_checkpointing=False,
        gradient_checkpointing_kwargs=None,
        hf_kt_config=SimpleNamespace(_kt_config={}),
    )

    with pytest.raises(ValueError, match="requires `bf16: true`"):
        model_args.apply_kt_config(_finetuning_args(), training_args, model_max_length=1024)


@pytest.mark.parametrize(
    ("model_kwargs", "finetuning_kwargs", "match"),
    [
        ({"kt_weight_path": None}, {}, "requires `kt_weight_path`"),
        (
            {"kt_expert_checkpoint_path": "/tmp/checkpoint"},
            {},
            "pre-quantized weights only",
        ),
        ({}, {"stage": "dpo"}, "only `stage: sft`"),
        ({}, {"finetuning_type": "full"}, "frozen-base"),
        ({"kt_weight_lifecycle": "ephemeral"}, {}, "requires `kt_weight_lifecycle: persistent`"),
        ({"activation_policy": {"cpu": "recompute", "gpu": "recompute"}}, {}, "requires `activation_policy"),
        ({"kt_backend": "AMXBF16"}, {}, "requires `kt_backend: amxint8`"),
        ({"kt_tp_enabled": False}, {}, "requires `kt_tp_enabled: true`"),
        ({"kt_threadpool_count": 1}, {}, "requires `kt_threadpool_count: 2`"),
        ({"kt_num_gpu_experts": 1}, {}, "requires `kt_num_gpu_experts: 0`"),
        ({"kt_share_backward_bb": False}, {}, "requires `kt_share_backward_bb: true`"),
        ({"kt_force_fused_expert_lora": False}, {}, "requires `kt_force_fused_expert_lora: true`"),
        ({"kt_use_lora_experts": True}, {}, "GPU-side LoRA experts"),
        ({"kt_lora_expert_num": 1}, {}, "GPU-side LoRA experts"),
        ({"kt_lora_expert_intermediate_size": 256}, {}, "GPU-side LoRA experts"),
        ({"kt_skip_expert_lora_adaptation": True}, {}, "skipping expert LoRA"),
        ({}, {"lora_dropout": 0.1}, "requires `lora_dropout: 0`"),
        ({}, {"lora_rank": 4}, "requires `lora_rank: 8`"),
        ({}, {"lora_alpha": 8}, "requires `lora_alpha: 16`"),
        ({}, {"lora_target": ["q_proj"]}, "requires `lora_target: all`"),
        ({}, {"use_rslora": True}, "does not support `use_rslora`"),
        ({}, {"pure_bf16": False}, "requires `pure_bf16: true`"),
        ({}, {"additional_target": ["embed_tokens"]}, "does not support `additional_target`"),
    ],
)
def test_int8_contract_fails_fast(model_kwargs: dict, finetuning_kwargs: dict, match: str):
    kwargs = {
        "model_name_or_path": "dummy",
        "use_kt": True,
        "kt_expert_weight_format": "int8",
        "kt_weight_path": "/tmp/int8-weights",
        "activation_policy": {"cpu": "retain", "gpu": "recompute"},
    }
    kwargs.update(model_kwargs)
    model_args = ModelArguments(**kwargs)

    with pytest.raises(ValueError, match=match):
        model_args.validate_kt_finetuning(_finetuning_args(**finetuning_kwargs))


def test_ephemeral_lifecycle_requires_explicit_weight_format():
    model_args = ModelArguments(
        model_name_or_path="dummy",
        use_kt=True,
        kt_weight_path="/tmp/weights",
        kt_weight_lifecycle="ephemeral",
    )

    with pytest.raises(ValueError, match="requires an explicit `kt_expert_weight_format`"):
        model_args.validate_kt_finetuning(_finetuning_args())


def test_int8_lifecycle_must_be_persistent():
    model_args = ModelArguments(
        model_name_or_path="dummy",
        use_kt=True,
        kt_expert_weight_format="int8",
        kt_weight_path="/tmp/weights",
        kt_weight_lifecycle="ephemeral",
        activation_policy={"cpu": "retain", "gpu": "recompute"},
    )

    with pytest.raises(ValueError, match="requires `kt_weight_lifecycle: persistent`"):
        model_args.validate_kt_finetuning(_finetuning_args())


@pytest.mark.parametrize(
    "arg_name",
    [
        "kt_backend",
        "kt_expert_weight_format",
        "kt_force_fused_expert_lora",
        "kt_num_gpu_experts",
        "kt_share_backward_bb",
        "kt_threadpool_count",
        "kt_tp_enabled",
        "kt_weight_lifecycle",
    ],
)
def test_int8_args_require_kt(arg_name: str):
    model_args = ModelArguments(model_name_or_path="dummy")

    with pytest.raises(ValueError, match="only valid when `use_kt: true`"):
        _validate_kt_activation_policy_source(model_args, {arg_name})


def test_default_lifecycle_preserves_existing_kt_configs():
    model_args = ModelArguments(model_name_or_path="dummy", use_kt=True)
    config = model_args.get_kt_config_dict(_finetuning_args(), model_max_length=1024)

    assert model_args.kt_expert_weight_format is None
    assert config["kt_weight_lifecycle"] == "persistent"
