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
import sys
import types
from types import SimpleNamespace

import pytest

from llamafactory.hparams import ModelArguments


@pytest.fixture(autouse=True)
def isolate_kt_environment():
    names = (
        "ACCELERATE_KT_ACTIVATION_POLICY",
        "ACCELERATE_KT_BACKEND",
        "ACCELERATE_KT_EXPERT_CHECKPOINT_PATH",
        "ACCELERATE_KT_LORA_ALPHA",
        "ACCELERATE_KT_LORA_EXPERT_INTERMEDIATE_SIZE",
        "ACCELERATE_KT_LORA_EXPERT_NUM",
        "ACCELERATE_KT_LORA_RANK",
        "ACCELERATE_KT_MODEL_MAX_LENGTH",
        "ACCELERATE_KT_USE_LORA_EXPERTS",
        "ACCELERATE_KT_WEIGHT_PATH",
        "ACCELERATE_USE_KT",
        "FSDP_ACTIVATION_CHECKPOINTING",
        "KT_REUSE_CHECKPOINT_FORWARD",
    )
    original = {name: os.environ[name] for name in names if name in os.environ}
    for name in names:
        os.environ.pop(name, None)

    yield

    for name in names:
        os.environ.pop(name, None)
    os.environ.update(original)


def _training_args(**overrides):
    values = {
        "gradient_checkpointing": False,
        "gradient_checkpointing_kwargs": None,
        "fsdp_config": {},
        "hf_kt_config": SimpleNamespace(_kt_config={}),
        "accelerator_config": SimpleNamespace(kt_config=None),
        "kt_config": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _finetuning_args(**overrides):
    values = {"stage": "sft", "finetuning_type": "lora", "lora_rank": 8, "lora_alpha": 16, "lora_dropout": 0.0}
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.parametrize(
    ("disable_gradient_checkpointing", "kt_cpu_activation", "expected"),
    [
        (False, None, {"cpu": "recompute", "gpu": "recompute"}),
        (False, "recompute", {"cpu": "recompute", "gpu": "recompute"}),
        (False, "retain", {"cpu": "retain", "gpu": "recompute"}),
        (True, None, {"cpu": "retain", "gpu": "retain"}),
        (True, "retain", {"cpu": "retain", "gpu": "retain"}),
    ],
)
def test_kt_activation_policy_resolution(disable_gradient_checkpointing, kt_cpu_activation, expected):
    model_args = ModelArguments(
        model_name_or_path="dummy",
        use_kt=True,
        disable_gradient_checkpointing=disable_gradient_checkpointing,
        kt_cpu_activation=kt_cpu_activation,
    )
    assert model_args.get_kt_activation_policy() == expected


def test_kt_cpu_recompute_requires_gpu_checkpointing():
    model_args = ModelArguments(
        model_name_or_path="dummy",
        use_kt=True,
        disable_gradient_checkpointing=True,
        kt_cpu_activation="recompute",
    )
    with pytest.raises(ValueError, match="requires GPU gradient checkpointing"):
        model_args.get_kt_activation_policy()


@pytest.mark.parametrize("kt_cpu_activation", ["invalid", "RETAIN"])
def test_kt_cpu_activation_rejects_invalid_value(kt_cpu_activation):
    with pytest.raises(ValueError, match="must be `retain` or `recompute`"):
        ModelArguments(
            model_name_or_path="dummy",
            use_kt=True,
            kt_cpu_activation=kt_cpu_activation,
        )


def test_kt_cpu_activation_requires_kt():
    with pytest.raises(ValueError, match="only valid when `use_kt: true`"):
        ModelArguments(model_name_or_path="dummy", kt_cpu_activation="retain")


def test_apply_kt_config_publishes_policy_without_cache_pool_override():
    model_args = ModelArguments(
        model_name_or_path="dummy",
        use_kt=True,
        kt_cpu_activation="retain",
    )
    training_args = _training_args()

    model_args.apply_kt_config(_finetuning_args(), training_args, model_max_length=1024)

    assert model_args.use_reentrant_gc is False
    assert training_args.gradient_checkpointing is False
    assert training_args.gradient_checkpointing_kwargs is None
    assert training_args.hf_kt_config._kt_config["kt_activation_policy"] == {
        "cpu": "retain",
        "gpu": "recompute",
    }
    assert "kt_share_cache_pool" not in training_args.hf_kt_config._kt_config
    assert training_args.accelerator_config.kt_config == {
        "enabled": True,
        "kt_config": training_args.hf_kt_config._kt_config,
    }
    assert training_args.hf_kt_config._kt_config["kt_train_mode"] == "lora"
    assert training_args.hf_kt_config._kt_config["kt_full_weight_grad"] is False


def test_apply_kt_config_uses_flat_training_yaml_as_only_advanced_source():
    model_args = ModelArguments(model_name_or_path="dummy", use_kt=True, kt_cpu_activation="retain")
    training_args = _training_args(
        kt_config={"kt_backend": "AMXBF16", "kt_model_max_length": 1152},
        hf_kt_config=SimpleNamespace(
            _kt_config={
                "enabled": True,
                "kt_skip_expert_loading": True,
                "kt_backend": "AMXBF16",
                "kt_model_max_length": 1152,
            }
        ),
    )

    model_args.apply_kt_config(_finetuning_args(), training_args, model_max_length=1024)

    plugin_config = training_args.accelerator_config.kt_config
    assert plugin_config["enabled"] is True
    assert plugin_config["kt_config"] is training_args.hf_kt_config._kt_config
    assert plugin_config["kt_config"]["kt_backend"] == "AMXBF16"
    assert plugin_config["kt_config"]["kt_model_max_length"] == 1152
    assert "kt_config" not in plugin_config["kt_config"]


def test_apply_kt_config_keeps_transformers_only_values_out_of_kernel_plugin():
    model_args = ModelArguments(
        model_name_or_path="dummy",
        use_kt=True,
        kt_cpu_activation="retain",
        kt_weight_path="/tmp/int8-experts",
        kt_non_expert_weight_path="/tmp/nonexpert-cache",
    )
    training_args = _training_args(
        kt_config={"kt_expert_weight_format": "int8"},
    )

    model_args.apply_kt_config(_finetuning_args(), training_args, model_max_length=1024)

    plugin_kernel_config = training_args.accelerator_config.kt_config["kt_config"]
    assert plugin_kernel_config["kt_expert_weight_format"] == "int8"
    assert plugin_kernel_config["kt_backend"] == "auto"
    assert plugin_kernel_config["kt_weight_lifecycle"] == "persistent"
    assert "kt_non_expert_weight_path" not in plugin_kernel_config
    assert "kt_non_expert_weight_path" not in training_args.hf_kt_config._kt_config


def test_apply_kt_config_is_idempotent():
    model_args = ModelArguments(model_name_or_path="dummy", use_kt=True, kt_cpu_activation="retain")
    training_args = _training_args()

    model_args.apply_kt_config(_finetuning_args(), training_args, model_max_length=1024)
    first_plugin_config = training_args.accelerator_config.kt_config
    model_args.apply_kt_config(_finetuning_args(), training_args, model_max_length=1024)

    assert training_args.accelerator_config.kt_config == first_plugin_config
    assert "kt_config" not in training_args.hf_kt_config._kt_config


@pytest.mark.parametrize("configured_capacity", [True, 1152.5, -1])
def test_apply_kt_config_rejects_invalid_plugin_capacity(configured_capacity):
    model_args = ModelArguments(model_name_or_path="dummy", use_kt=True)
    training_args = _training_args(kt_config={"kt_model_max_length": configured_capacity})

    with pytest.raises(ValueError, match="must be a positive integer"):
        model_args.apply_kt_config(_finetuning_args(), training_args, model_max_length=1024)


@pytest.mark.parametrize(
    ("training_overrides", "message"),
    [
        ({"gradient_checkpointing": True}, "second checkpoint wrapper"),
        ({"gradient_checkpointing_kwargs": {}}, "checkpoint context internally"),
        ({"gradient_checkpointing_kwargs": {"use_reentrant": False}}, "checkpoint context internally"),
        ({"fsdp_config": {"activation_checkpointing": True}}, "FSDP activation checkpointing"),
    ],
)
def test_kt_rejects_duplicate_checkpoint_entrypoints(training_overrides, message):
    model_args = ModelArguments(model_name_or_path="dummy", use_kt=True)
    with pytest.raises(ValueError, match=message):
        model_args.apply_kt_config(
            _finetuning_args(),
            _training_args(**training_overrides),
            model_max_length=1024,
        )


def test_kt_rejects_unsloth_checkpointing():
    model_args = ModelArguments(model_name_or_path="dummy", use_kt=True, use_unsloth_gc=True)
    with pytest.raises(ValueError, match="does not support `use_unsloth_gc`"):
        model_args.apply_kt_config(_finetuning_args(), _training_args(), model_max_length=1024)


def test_kt_rejects_unsloth_model_wrapper():
    model_args = ModelArguments(model_name_or_path="dummy", use_kt=True, use_unsloth=True)
    with pytest.raises(ValueError, match="does not support `use_unsloth` checkpoint wrapping"):
        model_args.apply_kt_config(_finetuning_args(), _training_args(), model_max_length=1024)


def test_kt_rejects_fsdp_activation_checkpointing_env(monkeypatch):
    monkeypatch.setenv("FSDP_ACTIVATION_CHECKPOINTING", "true")
    model_args = ModelArguments(model_name_or_path="dummy", use_kt=True)
    with pytest.raises(ValueError, match="FSDP activation checkpointing"):
        model_args.apply_kt_config(_finetuning_args(), _training_args(), model_max_length=1024)


@pytest.mark.parametrize("env_name", ["ACCELERATE_KT_ACTIVATION_POLICY", "KT_REUSE_CHECKPOINT_FORWARD"])
def test_kt_rejects_legacy_activation_env(monkeypatch, env_name):
    monkeypatch.setenv(env_name, "retain,recompute")
    model_args = ModelArguments(model_name_or_path="dummy", use_kt=True)
    with pytest.raises(ValueError, match="legacy environment settings"):
        model_args.apply_kt_config(_finetuning_args(), _training_args(), model_max_length=1024)


def test_kt_rejects_duplicate_activation_policy_config():
    model_args = ModelArguments(model_name_or_path="dummy", use_kt=True)
    training_args = _training_args(kt_config={"kt_activation_policy": {"cpu": "retain", "gpu": "recompute"}})
    with pytest.raises(ValueError, match="owned by LLaMA-Factory"):
        model_args.apply_kt_config(_finetuning_args(), training_args, model_max_length=1024)


def test_kt_rejects_accelerate_yaml_as_config_source():
    model_args = ModelArguments(model_name_or_path="dummy", use_kt=True)
    training_args = _training_args(accelerator_config=SimpleNamespace(kt_config={"kt_backend": "AMXBF16"}))
    with pytest.raises(ValueError, match="remove `kt_config` from the Accelerate config"):
        model_args.apply_kt_config(_finetuning_args(), training_args, model_max_length=1024)


@pytest.mark.parametrize("forbidden", ["kt_lora_rank", "kt_weight_path", "enabled", "kt_train_mode"])
def test_kt_rejects_lf_derived_fields_in_flat_config(forbidden):
    model_args = ModelArguments(model_name_or_path="dummy", use_kt=True)
    value = True if forbidden == "enabled" else 4
    if forbidden == "enabled":
        value = False  # True is the frozen Transformers injected default.
    with pytest.raises(ValueError, match="derived|owned"):
        model_args.apply_kt_config(
            _finetuning_args(),
            _training_args(kt_config={forbidden: value}),
            model_max_length=1024,
        )


def test_kt_rejects_external_runtime_environment(monkeypatch):
    monkeypatch.setenv("ACCELERATE_KT_BACKEND", "AMXBF16")
    model_args = ModelArguments(model_name_or_path="dummy", use_kt=True)
    with pytest.raises(ValueError, match="only KTransformers configuration source"):
        model_args.apply_kt_config(_finetuning_args(), _training_args(), model_max_length=1024)


def test_frozen_transformers_and_accelerate_accept_lf_plugin_shape(monkeypatch):
    kt_integration = pytest.importorskip("transformers.integrations.kt")
    accelerate_utils = pytest.importorskip("accelerate.utils")
    plugin_class = getattr(accelerate_utils, "KTransformersPlugin", None)
    if plugin_class is None:
        pytest.skip("installed Accelerate does not provide the historical KT plugin")

    kernel_config = {
        "kt_backend": "AMXBF16",
        "kt_activation_policy": {"cpu": "retain", "gpu": "recompute"},
        "kt_lora_rank": 8,
        "kt_lora_alpha": 16,
        "kt_lora_dropout": 0.0,
        "kt_model_max_length": 1024,
        "kt_train_mode": "lora",
        "kt_full_weight_grad": False,
    }
    hf_config = kt_integration.HfTrainerKTConfig(kernel_config)
    fake_sft = types.ModuleType("kt_kernel.sft")

    class FakeKTConfig:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    fake_sft.KTConfig = FakeKTConfig
    fake_package = types.ModuleType("kt_kernel")
    fake_package.sft = fake_sft
    monkeypatch.setitem(sys.modules, "kt_kernel", fake_package)
    monkeypatch.setitem(sys.modules, "kt_kernel.sft", fake_sft)
    try:
        plugin = plugin_class(enabled=True, kt_config=kernel_config)
        assert hf_config.kt_activation_policy == {"cpu": "retain", "gpu": "recompute"}
        assert plugin.kt_config.kt_model_max_length == 1024
        assert plugin.kt_config.kt_train_mode == "lora"
    finally:
        kt_integration.unset_kt_config()
