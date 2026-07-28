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

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from llamafactory.model.model_utils.fsdp import (
    _collect_kt_fsdp_ignored_params,
    _kt_fsdp2_streaming_load_full_state_dict,
    _preserve_parameter_identity_on_conversion,
    patch_fsdp2_kt_parameter_identity,
)


def _accelerate_kt_prepare_model(accelerator, model):
    from torch.distributed.fsdp import FSDPModule, MixedPrecisionPolicy, fully_shard

    del FSDPModule, MixedPrecisionPolicy
    frozen_params_to_ignore = set()
    for _, param in model.named_parameters():
        if not param.requires_grad:
            frozen_params_to_ignore.add(param)

    ignored = set()
    fsdp2_kwargs = {}
    fsdp2_kwargs["ignored_params"] = ignored | frozen_params_to_ignore
    model = model.to(torch.device("meta"))
    fully_shard(model, **fsdp2_kwargs)
    return model


def _accelerate_kt_load_full_state_dict(accelerator, model, full_sd, cpu_offload=False):
    del accelerator, full_sd, cpu_offload

    def _is_dtensor(param):
        return hasattr(param, "device_mesh")

    sharded_sd = {}
    for name, sharded_param in model.state_dict().items():
        if not _is_dtensor(sharded_param):
            sharded_sd[name] = sharded_param

    model.load_state_dict(sharded_sd, assign=True)
    return model


def _accelerate_kt_switch_optimizer_parameters(optimizer, mapping):
    for param_group in optimizer.param_groups:
        new_params = []
        for p in param_group["params"]:
            ptr = p.data_ptr()
            if ptr in mapping:
                new_params.append(mapping[ptr])
            else:
                new_params.append(p)

        param_group["params"] = new_params


def _install_fake_accelerate(monkeypatch, *, version="1.14.0", prepare_model=_accelerate_kt_prepare_model):
    accelerate_module = ModuleType("accelerate")
    accelerator_module = ModuleType("accelerate.accelerator")
    accelerate_utils = ModuleType("accelerate.utils")
    fsdp_utils = ModuleType("accelerate.utils.fsdp_utils")

    accelerate_module.__version__ = version
    accelerator_module.fsdp2_prepare_model = prepare_model
    accelerate_utils.fsdp2_prepare_model = prepare_model
    accelerate_utils.fsdp2_load_full_state_dict = _accelerate_kt_load_full_state_dict
    accelerate_utils.fsdp_utils = fsdp_utils
    fsdp_utils.fsdp2_prepare_model = prepare_model
    fsdp_utils.fsdp2_load_full_state_dict = _accelerate_kt_load_full_state_dict
    fsdp_utils.fsdp2_switch_optimizer_parameters = _accelerate_kt_switch_optimizer_parameters

    accelerate_module.accelerator = accelerator_module
    accelerate_module.utils = accelerate_utils

    monkeypatch.setitem(sys.modules, "accelerate", accelerate_module)
    monkeypatch.setitem(sys.modules, "accelerate.accelerator", accelerator_module)
    monkeypatch.setitem(sys.modules, "accelerate.utils", accelerate_utils)
    monkeypatch.setitem(sys.modules, "accelerate.utils.fsdp_utils", fsdp_utils)
    return accelerator_module, accelerate_utils, fsdp_utils


class _KTExpertWrapper(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._is_kt_moe_wrapper = True
        self._experts_attr = "experts"
        self.gate = torch.nn.Linear(4, 2, bias=False)
        self.shared_expert = torch.nn.Linear(4, 4, bias=False)
        self.experts = torch.nn.ModuleDict(
            {
                "base": torch.nn.Linear(4, 4, bias=False),
                "lora": torch.nn.Linear(4, 2, bias=False),
            }
        )

        self.gate.requires_grad_(False)
        self.shared_expert.requires_grad_(False)
        self.experts["base"].requires_grad_(False)
        self.experts["base"].weight._kt_zero_storage = True


class _KTModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(8, 4)
        self.embedding.requires_grad_(False)
        self.user_ignored = torch.nn.Linear(4, 4, bias=False)
        self.user_ignored.requires_grad_(False)
        self.moe = _KTExpertWrapper()


def _run_streaming_loader_worker(rank: int, init_file: str):
    import torch.distributed as dist
    from torch.distributed.device_mesh import DeviceMesh
    from torch.distributed.tensor import DTensor, Shard

    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=2,
    )
    try:
        mesh = DeviceMesh("cpu", [0, 1])
        model = torch.nn.Module()
        for name, shape in (("even", (4, 3)), ("uneven", (5, 3))):
            chunk_size = (shape[0] + 1) // 2
            start = min(rank * chunk_size, shape[0])
            local_shape = (min(chunk_size, shape[0] - start), shape[1])
            local = torch.empty(local_shape, dtype=torch.bfloat16)
            value = DTensor.from_local(
                local,
                device_mesh=mesh,
                placements=(Shard(0),),
                run_check=False,
                shape=torch.Size(shape),
                stride=(shape[1], 1),
            )
            model.register_parameter(name, torch.nn.Parameter(value))

        if rank == 0:
            model.register_parameter("rank_zero_only", torch.nn.Parameter(torch.empty(2, dtype=torch.bfloat16)))
            full_state = {
                "even": torch.arange(12, dtype=torch.bfloat16).reshape(4, 3),
                "uneven": torch.arange(15, dtype=torch.bfloat16).reshape(5, 3),
                "rank_zero_only": torch.tensor([7.0, 11.0], dtype=torch.bfloat16),
                "persistent": torch.tensor([2, 3, 5]),
            }
        else:
            full_state = {}
        model.register_buffer("persistent", torch.empty(3, dtype=torch.int64))

        accelerator = SimpleNamespace(is_main_process=rank == 0, device=torch.device("cpu"))
        original_requires_grad = model.even.requires_grad
        _kt_fsdp2_streaming_load_full_state_dict(
            accelerator,
            model,
            full_state,
            rank_zero_only_names=frozenset({"rank_zero_only"}),
        )

        expected_even = torch.arange(12, dtype=torch.bfloat16).reshape(4, 3).chunk(2)[rank]
        expected_uneven = torch.arange(15, dtype=torch.bfloat16).reshape(5, 3).chunk(2)[rank]
        torch.testing.assert_close(model.even.to_local(), expected_even)
        torch.testing.assert_close(model.uneven.to_local(), expected_uneven)
        torch.testing.assert_close(model.persistent, torch.tensor([2, 3, 5]))
        assert model.even.requires_grad is original_requires_grad
        if rank == 0:
            torch.testing.assert_close(
                model.rank_zero_only,
                torch.tensor([7.0, 11.0], dtype=torch.bfloat16),
            )
        else:
            assert not hasattr(model, "rank_zero_only")
    finally:
        dist.destroy_process_group()


def test_parameter_identity_is_preserved_during_meta_conversion():
    module = torch.nn.Linear(4, 4)
    state_dict = module.state_dict()
    weight_id = id(module.weight)
    previous = torch.__future__.get_swap_module_params_on_conversion()

    with _preserve_parameter_identity_on_conversion():
        module.to(torch.device("meta"))

    assert id(module.weight) == weight_id
    assert module.weight.device.type == "meta"
    assert state_dict["weight"].device.type == "cpu"
    assert torch.__future__.get_swap_module_params_on_conversion() is previous


def test_only_explicit_kt_expert_subtree_is_ignored():
    model = _KTModel()

    ignored = _collect_kt_fsdp_ignored_params(model)

    assert ignored == set(model.moe.experts.parameters())
    assert model.embedding.weight not in ignored
    assert model.moe.gate.weight not in ignored
    assert model.moe.shared_expert.weight not in ignored


def test_streaming_loader_shards_even_and_uneven_tensors_and_preserves_rank_ownership(tmp_path):
    torch.multiprocessing.spawn(
        _run_streaming_loader_worker,
        args=(str(tmp_path / "gloo_init"),),
        nprocs=2,
        join=True,
    )


def test_fsdp2_patch_filters_broad_frozen_ignore_and_preserves_identity(monkeypatch):
    accelerator_module, accelerate_utils, fsdp_utils = _install_fake_accelerate(monkeypatch)
    model_args = SimpleNamespace(use_kt=True, kt_expert_weight_format="int8")
    model = _KTModel()
    explicit_kt_params = set(model.moe.experts.parameters())
    kt_parameter_ids = {id(param) for param in explicit_kt_params}
    captured_ignored_params = []

    import torch.distributed.fsdp as torch_fsdp

    original_fully_shard = torch_fsdp.fully_shard

    def fake_fully_shard(module, *, ignored_params=None, **kwargs):
        del module, kwargs
        captured_ignored_params.append(set(ignored_params or set()))

    monkeypatch.setattr(torch_fsdp, "fully_shard", fake_fully_shard)

    patch_fsdp2_kt_parameter_identity(model_args)

    assert accelerator_module.fsdp2_prepare_model is accelerate_utils.fsdp2_prepare_model
    assert accelerator_module.fsdp2_prepare_model is fsdp_utils.fsdp2_prepare_model
    assert accelerator_module.fsdp2_prepare_model is not _accelerate_kt_prepare_model

    accelerator = SimpleNamespace(
        state=SimpleNamespace(fsdp_plugin=SimpleNamespace(ignored_modules=[model.user_ignored]))
    )
    prepared_model = accelerator_module.fsdp2_prepare_model(accelerator, model)

    expected_ignored = explicit_kt_params | set(model.user_ignored.parameters())
    assert captured_ignored_params == [expected_ignored]
    assert model.embedding.weight not in captured_ignored_params[0]
    assert model.moe.gate.weight not in captured_ignored_params[0]
    assert model.moe.shared_expert.weight not in captured_ignored_params[0]
    assert kt_parameter_ids.issubset({id(param) for param in prepared_model.parameters()})
    assert torch_fsdp.fully_shard is fake_fully_shard
    assert original_fully_shard is not fake_fully_shard
    assert fsdp_utils.fsdp2_load_full_state_dict is _accelerate_kt_load_full_state_dict
    assert accelerate_utils.fsdp2_load_full_state_dict is _accelerate_kt_load_full_state_dict


def test_fsdp2_patch_rejects_unknown_accelerate_version(monkeypatch):
    _install_fake_accelerate(monkeypatch, version="1.15.0")
    model_args = SimpleNamespace(use_kt=True, kt_expert_weight_format="int8")

    with pytest.raises(RuntimeError, match="Accelerate KT 1.14.x"):
        patch_fsdp2_kt_parameter_identity(model_args)


def test_fsdp2_patch_rejects_nonfloating_frozen_param_outside_kt(monkeypatch):
    accelerator_module, _, _ = _install_fake_accelerate(monkeypatch)
    model_args = SimpleNamespace(use_kt=True, kt_expert_weight_format="int8")
    model = _KTModel()
    model.register_parameter(
        "packed_weight",
        torch.nn.Parameter(torch.ones(4, dtype=torch.int8), requires_grad=False),
    )
    accelerator = SimpleNamespace(state=SimpleNamespace(fsdp_plugin=SimpleNamespace(ignored_modules=None)))

    patch_fsdp2_kt_parameter_identity(model_args)

    with pytest.raises(RuntimeError, match="non-floating frozen parameters"):
        accelerator_module.fsdp2_prepare_model(accelerator, model)
