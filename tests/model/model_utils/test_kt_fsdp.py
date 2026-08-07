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

import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from llamafactory.model.model_utils.kt_fsdp import (
    get_kt_fsdp2_adapter_state_dict,
    maybe_register_kt_fsdp2_persistent_buffer_hook,
    raise_kt_distributed_save_errors,
    register_kt_fsdp2_persistent_buffer_hook,
    validate_kt_distributed_checkpoint_policy,
)


class _BufferModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("persistent", torch.arange(4, dtype=torch.float32))
        self.register_buffer("non_persistent", torch.ones(2), persistent=False)


class _AdapterAndPlaceholderModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.adapter = torch.nn.Parameter(torch.ones(2, 2))
        self.experts = torch.nn.Module()
        self.experts.register_parameter("placeholder", torch.nn.Parameter(torch.empty(1), requires_grad=False))

        def omit_placeholder(_module, state_dict, _prefix, _metadata):
            state_dict.pop("experts.placeholder")

        self._register_state_dict_hook(omit_placeholder)


def _persistent_buffer_worker(rank: int, init_file: str) -> None:
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=2)
    try:
        model = _BufferModel()
        accelerator = SimpleNamespace(device=torch.device("cpu"))
        handle = register_kt_fsdp2_persistent_buffer_hook(model, accelerator)
        assert handle is not None

        source = torch.arange(4, dtype=torch.float32) if rank == 0 else torch.empty(4, device="meta")
        model.load_state_dict({"persistent": source}, assign=True)

        assert not model.persistent.is_meta
        torch.testing.assert_close(model.persistent, torch.arange(4, dtype=torch.float32))
        assert model.non_persistent.device.type == "cpu"
        assert handle.id not in model._load_state_dict_post_hooks
    finally:
        dist.destroy_process_group()


def _tiny_peft_model():
    from peft import LoraConfig, get_peft_model
    from transformers import LlamaConfig, LlamaForCausalLM

    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            tie_word_embeddings=False,
        )
    )
    return get_peft_model(model, LoraConfig(r=2, lora_alpha=4, target_modules=["q_proj"], bias="none"))


def _adapter_gather_worker(rank: int, init_file: str) -> None:
    from torch.distributed.device_mesh import DeviceMesh
    from torch.distributed.fsdp import fully_shard

    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=2)
    try:
        model = _tiny_peft_model()
        fully_shard(model, mesh=DeviceMesh("cpu", [0, 1]), reshard_after_forward=True)
        state_dict = get_kt_fsdp2_adapter_state_dict(model, placeholder_getter=lambda _model: {})
        if rank == 0:
            assert set(state_dict) == {
                "base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight",
                "base_model.model.model.layers.0.self_attn.q_proj.lora_B.default.weight",
            }
            assert all(tensor.device.type == "cpu" for tensor in state_dict.values())
        else:
            assert state_dict == {}
    finally:
        dist.destroy_process_group()


def _adapter_preflight_error_worker(rank: int, init_file: str) -> None:
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=2)
    try:
        model = _AdapterAndPlaceholderModel()

        def placeholder_getter(_model):
            if rank == 0:
                raise RuntimeError("rank-zero placeholder failure")
            return {"experts.placeholder": model.experts.placeholder}

        with pytest.raises(RuntimeError, match="rank-zero placeholder failure"):
            get_kt_fsdp2_adapter_state_dict(model, placeholder_getter=placeholder_getter)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available() or os.name == "nt", reason="requires distributed PyTorch")
def test_persistent_buffer_hook_broadcasts_and_removes_itself(tmp_path: Path):
    mp.spawn(_persistent_buffer_worker, args=(str(tmp_path / "buffer-init"),), nprocs=2, join=True)


@pytest.mark.skipif(
    not dist.is_available() or os.name == "nt" or importlib.util.find_spec("peft") is None,
    reason="requires distributed PyTorch and PEFT",
)
def test_fsdp2_adapter_gather_is_trainable_only(tmp_path: Path):
    mp.spawn(_adapter_gather_worker, args=(str(tmp_path / "adapter-init"),), nprocs=2, join=True)


@pytest.mark.skipif(not dist.is_available() or os.name == "nt", reason="requires distributed PyTorch")
def test_fsdp2_adapter_gather_exchanges_preflight_errors(tmp_path: Path):
    mp.spawn(_adapter_preflight_error_worker, args=(str(tmp_path / "error-init"),), nprocs=2, join=True)


def test_adapter_gather_uses_explicit_placeholder_identities_and_restores_grad():
    model = _AdapterAndPlaceholderModel()
    placeholder = model.experts.placeholder
    gathered = {
        "adapter": model.adapter.detach().clone(),
        "experts.placeholder": placeholder.detach().clone(),
    }

    def gather(_model, *, options):
        assert placeholder.requires_grad
        assert options.full_state_dict
        assert options.cpu_offload
        assert options.ignore_frozen_params
        return gathered

    with patch("torch.distributed.checkpoint.state_dict.get_model_state_dict", side_effect=gather):
        state_dict = get_kt_fsdp2_adapter_state_dict(
            model, placeholder_getter=lambda _model: {"experts.placeholder": placeholder}
        )

    assert state_dict is gathered
    assert set(state_dict) == {"adapter"}
    assert not placeholder.requires_grad


def test_adapter_gather_rejects_stale_placeholder_identity():
    model = _AdapterAndPlaceholderModel()
    stale = torch.nn.Parameter(torch.empty(1), requires_grad=False)
    with pytest.raises(RuntimeError, match="does not match the current model parameter identity"):
        get_kt_fsdp2_adapter_state_dict(model, placeholder_getter=lambda _model: {"experts.placeholder": stale})


def test_frozen_accelerate_buffer_hook_is_version_gated():
    model = _BufferModel()
    accelerator = SimpleNamespace(
        device=torch.device("cpu"),
        is_fsdp2=True,
        state=SimpleNamespace(
            fsdp_plugin=SimpleNamespace(fsdp_version=2, cpu_ram_efficient_loading=True)
        ),
    )

    assert (
        maybe_register_kt_fsdp2_persistent_buffer_hook(model, accelerator, True, accelerate_kt_version="1.14.0.post2")
        is None
    )
    handle = maybe_register_kt_fsdp2_persistent_buffer_hook(
        model, accelerator, True, accelerate_kt_version="1.14.0.post1"
    )
    assert handle is not None
    handle.remove()


def test_buffer_hook_is_not_installed_without_ram_efficient_loading():
    model = _BufferModel()
    accelerator = SimpleNamespace(
        device=torch.device("cpu"),
        is_fsdp2=True,
        state=SimpleNamespace(
            fsdp_plugin=SimpleNamespace(fsdp_version=2, cpu_ram_efficient_loading=False)
        ),
    )

    assert (
        maybe_register_kt_fsdp2_persistent_buffer_hook(
            model, accelerator, True, accelerate_kt_version="1.14.0.post1"
        )
        is None
    )


def test_non_main_rank_observes_rank_zero_save_error():
    with (
        patch(
            "llamafactory.model.model_utils.kt_fsdp._all_rank_errors",
            return_value=("rank 0: OSError: disk full",),
        ),
        pytest.raises(RuntimeError, match="rank 0: OSError: disk full"),
    ):
        raise_kt_distributed_save_errors(None)


@pytest.mark.parametrize(
    ("save_strategy", "save_only_model"),
    [("no", False), ("no", True), ("steps", True)],
)
def test_supported_multi_rank_kt_save_policies(save_strategy: str, save_only_model: bool):
    model_args = SimpleNamespace(use_kt=True)
    training_args = SimpleNamespace(
        resume_from_checkpoint=None,
        save_strategy=save_strategy,
        save_only_model=save_only_model,
    )
    validate_kt_distributed_checkpoint_policy(model_args, training_args, world_size=2)


def test_multi_rank_kt_optimizer_checkpoint_is_rejected():
    model_args = SimpleNamespace(use_kt=True)
    training_args = SimpleNamespace(
        resume_from_checkpoint=None,
        save_strategy="steps",
        save_only_model=False,
    )
    with pytest.raises(ValueError, match="periodic optimizer checkpoints"):
        validate_kt_distributed_checkpoint_policy(model_args, training_args, world_size=2)


def test_multi_rank_kt_resume_is_rejected():
    model_args = SimpleNamespace(use_kt=True)
    training_args = SimpleNamespace(
        resume_from_checkpoint="checkpoint-10",
        save_strategy="no",
        save_only_model=True,
    )
    with pytest.raises(ValueError, match="checkpoint resume"):
        validate_kt_distributed_checkpoint_policy(model_args, training_args, world_size=2)


def test_multi_rank_kt_load_best_model_is_rejected():
    model_args = SimpleNamespace(use_kt=True)
    training_args = SimpleNamespace(
        resume_from_checkpoint=None,
        load_best_model_at_end=True,
        save_strategy="steps",
        save_only_model=True,
    )
    with pytest.raises(ValueError, match="load_best_model_at_end"):
        validate_kt_distributed_checkpoint_policy(model_args, training_args, world_size=2)


def test_non_kt_checkpoint_policy_is_unchanged():
    model_args = SimpleNamespace(use_kt=False)
    training_args = SimpleNamespace(
        resume_from_checkpoint="checkpoint-10",
        save_strategy="steps",
        save_only_model=False,
    )
    validate_kt_distributed_checkpoint_policy(model_args, training_args, world_size=8)
