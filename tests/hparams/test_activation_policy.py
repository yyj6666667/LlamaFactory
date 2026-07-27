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
