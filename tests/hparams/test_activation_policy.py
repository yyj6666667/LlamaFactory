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

from types import SimpleNamespace

import pytest
import torch

from llamafactory.hparams import model_args as model_args_module
from llamafactory.hparams.model_args import ModelArguments
from llamafactory.hparams.parser import _validate_kt_activation_policy_source
from llamafactory.model.model_utils import checkpointing


@pytest.mark.parametrize(
    "policy",
    [
        {"cpu": "retain", "gpu": "retain"},
        {"cpu": "retain", "gpu": "recompute"},
        {"cpu": "recompute", "gpu": "recompute"},
    ],
)
def test_supported_activation_policies(policy: dict[str, str]):
    model_args = ModelArguments(model_name_or_path="dummy", use_kt=True, activation_policy=policy)
    assert model_args.activation_policy == policy


def test_default_activation_policy():
    model_args = ModelArguments(model_name_or_path="dummy", use_kt=True)
    assert model_args.activation_policy == {"cpu": "recompute", "gpu": "recompute"}


@pytest.mark.parametrize(
    ("policy", "error", "match"),
    [
        ({"cpu": "retain"}, ValueError, "missing keys"),
        ({"cpu": "retain", "gpu": "recompute", "other": "retain"}, ValueError, "unknown keys"),
        ({"cpu": "drop", "gpu": "recompute"}, ValueError, "must be `retain` or `recompute`"),
        (
            {"cpu": "recompute", "gpu": "retain"},
            NotImplementedError,
            "is not supported yet",
        ),
    ],
)
def test_invalid_activation_policies(policy: dict[str, str], error: type[Exception], match: str):
    with pytest.raises(error, match=match):
        ModelArguments(model_name_or_path="dummy", use_kt=True, activation_policy=policy)


@pytest.mark.parametrize(
    "legacy_arg",
    [
        "disable_gradient_checkpointing",
        "gradient_checkpointing",
        "gradient_checkpointing_kwargs",
        "use_reentrant_gc",
        "use_unsloth_gc",
    ],
)
def test_kt_rejects_legacy_checkpointing_sources(legacy_arg: str):
    model_args = ModelArguments(model_name_or_path="dummy", use_kt=True)
    with pytest.raises(ValueError, match="only checkpointing configuration"):
        _validate_kt_activation_policy_source(model_args, {legacy_arg})


def test_policy_requires_kt():
    model_args = ModelArguments(model_name_or_path="dummy")
    with pytest.raises(ValueError, match="only valid when `use_kt: true`"):
        _validate_kt_activation_policy_source(model_args, {"activation_policy"})


def test_recompute_policy_uses_only_llamafactory_checkpointing(monkeypatch: pytest.MonkeyPatch):
    context_fn = object()
    monkeypatch.setattr(model_args_module, "_get_kt_activation_checkpoint_context_fn", lambda: context_fn)
    model_args = ModelArguments(
        model_name_or_path="dummy",
        use_kt=True,
        activation_policy={"cpu": "retain", "gpu": "recompute"},
    )
    training_args = SimpleNamespace(
        gradient_checkpointing=False,
        gradient_checkpointing_kwargs=None,
        hf_kt_config=SimpleNamespace(_kt_config={}),
    )
    finetuning_args = SimpleNamespace(lora_rank=None, lora_alpha=None)

    model_args.apply_kt_config(finetuning_args, training_args, model_max_length=None)

    assert model_args.disable_gradient_checkpointing is False
    assert model_args.use_reentrant_gc is False
    assert model_args.kt_activation_checkpoint_context_fn is context_fn
    assert training_args.gradient_checkpointing is False
    assert training_args.gradient_checkpointing_kwargs is None
    assert training_args.hf_kt_config._kt_config["kt_activation_policy"] == {
        "cpu": "retain",
        "gpu": "recompute",
    }


def test_retain_policy_disables_both_checkpointing_paths(monkeypatch: pytest.MonkeyPatch):
    def fail_context_import():
        raise AssertionError("GPU retain must not import the KT checkpoint context")

    monkeypatch.setattr(model_args_module, "_get_kt_activation_checkpoint_context_fn", fail_context_import)
    model_args = ModelArguments(
        model_name_or_path="dummy",
        use_kt=True,
        activation_policy={"cpu": "retain", "gpu": "retain"},
    )
    training_args = SimpleNamespace(
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": True},
        hf_kt_config=SimpleNamespace(_kt_config={}),
    )
    finetuning_args = SimpleNamespace(lora_rank=None, lora_alpha=None)

    model_args.apply_kt_config(finetuning_args, training_args, model_max_length=None)

    assert model_args.disable_gradient_checkpointing is True
    assert model_args.kt_activation_checkpoint_context_fn is None
    assert training_args.gradient_checkpointing is False
    assert training_args.gradient_checkpointing_kwargs is None


def test_llamafactory_checkpointing_uses_kt_context(monkeypatch: pytest.MonkeyPatch):
    context_fn = object()
    captured_kwargs = None

    def capture_checkpointing_kwargs(
        self,
        gradient_checkpointing_kwargs=None,
        use_unsloth_gc=False,  # noqa: ARG001
    ):
        nonlocal captured_kwargs
        captured_kwargs = gradient_checkpointing_kwargs

    monkeypatch.setattr(checkpointing, "_gradient_checkpointing_enable", capture_checkpointing_kwargs)
    model_args = ModelArguments(model_name_or_path="dummy", use_kt=True)
    model_args.disable_gradient_checkpointing = False
    model_args.use_reentrant_gc = False
    model_args.kt_activation_checkpoint_context_fn = context_fn
    model = SimpleNamespace(
        supports_gradient_checkpointing=True,
        config=SimpleNamespace(use_cache=True),
        named_parameters=lambda: [],
    )

    checkpointing.prepare_model_for_training(model, model_args)

    assert captured_kwargs == {"use_reentrant": False, "context_fn": context_fn}
    assert model.config.use_cache is False


def test_deepseek_int8_native_config_contract():
    native_config_type = type(
        "DeepseekV3Config",
        (),
        {"__module__": "transformers.models.deepseek_v3.configuration_deepseek_v3"},
    )
    config = native_config_type()
    config.model_type = "deepseek_v3"
    config.num_hidden_layers = 61
    config.first_k_dense_replace = 3
    config.n_routed_experts = 256
    model_args = ModelArguments(
        model_name_or_path="dummy",
        use_kt=True,
        kt_expert_weight_format="int8",
        kt_weight_path="/tmp/int8",
        activation_policy={"cpu": "retain", "gpu": "recompute"},
    )

    checkpointing.validate_kt_deepseek_v3_native_config(config, model_args)


def test_deepseek_int8_rejects_remote_config():
    remote_config_type = type("DeepseekV3Config", (), {"__module__": "transformers_modules.deepseek.modeling"})
    config = remote_config_type()
    config.model_type = "deepseek_v3"
    config.num_hidden_layers = 61
    config.first_k_dense_replace = 3
    config.n_routed_experts = 256
    model_args = ModelArguments(
        model_name_or_path="dummy",
        use_kt=True,
        kt_expert_weight_format="int8",
        kt_weight_path="/tmp/int8",
        activation_policy={"cpu": "retain", "gpu": "recompute"},
    )

    with pytest.raises(RuntimeError, match="native"):
        checkpointing.validate_kt_deepseek_v3_native_config(config, model_args)


def test_deepseek_int8_checkpoint_contract_covers_all_native_layers():
    modeling_layers = pytest.importorskip("transformers.modeling_layers")
    gradient_layer_type = modeling_layers.GradientCheckpointingLayer
    context_fn = object()

    class DeepseekV3DecoderLayer(gradient_layer_type):
        def __init__(self, layer_idx):
            super().__init__()
            self.proj = torch.nn.Linear(2, 2)
            self.gradient_checkpointing = True
            self._gradient_checkpointing_func = lambda *args, **kwargs: None
            self._gradient_checkpointing_func._llamafactory_checkpoint_contract = {
                "use_reentrant": False,
                "context_fn": context_fn,
            }
            if layer_idx >= 3:
                self.mlp = torch.nn.Module()
                self.mlp._is_kt_moe_wrapper = True
                self.mlp.layer_idx = layer_idx

    class Backbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([DeepseekV3DecoderLayer(layer_idx) for layer_idx in range(61)])

    class DeepseekV3ForCausalLM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Backbone()
            self.config = SimpleNamespace(use_cache=False)

    DeepseekV3ForCausalLM.__module__ = "transformers.models.deepseek_v3.modeling_deepseek_v3"
    model_args = ModelArguments(
        model_name_or_path="dummy",
        use_kt=True,
        kt_expert_weight_format="int8",
        kt_weight_path="/tmp/int8",
        activation_policy={"cpu": "retain", "gpu": "recompute"},
    )
    model_args.kt_activation_checkpoint_context_fn = context_fn

    checkpointing._validate_kt_deepseek_v3_checkpointing(DeepseekV3ForCausalLM(), model_args)
