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

import re
from types import SimpleNamespace

import torch

from llamafactory.model.adapter import _get_kt_fused_expert_exclude_pattern


class _Experts(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = torch.nn.Linear(4, 4, bias=False)
        self.up_proj = torch.nn.Linear(4, 4, bias=False)
        self.down_proj = torch.nn.Linear(4, 4, bias=False)


class _KTWrapper(torch.nn.Module):
    def __init__(self, *, forced: bool):
        super().__init__()
        self._is_kt_moe_wrapper = True
        self._force_fused_expert_lora = forced
        self._experts_attr = "experts"
        self.moe_config = SimpleNamespace(weight_names=("gate_proj", "up_proj", "down_proj"))
        self.experts = torch.nn.ModuleList([_Experts(), _Experts()])
        self.shared_experts = _Experts()


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([_KTWrapper(forced=True), _KTWrapper(forced=True)])


def test_fused_expert_exclusion_matches_only_routed_expert_projections():
    pattern = _get_kt_fused_expert_exclude_pattern(_Model())

    assert re.fullmatch(pattern, "layers.0.experts.0.gate_proj")
    assert re.fullmatch(pattern, "layers.1.experts.1.down_proj")
    assert not re.fullmatch(pattern, "layers.0.shared_experts.gate_proj")
    assert not re.fullmatch(pattern, "layers.0.experts.0.router")


def test_fused_expert_exclusion_requires_forced_wrappers():
    model = torch.nn.Module()
    model.wrapper = _KTWrapper(forced=False)

    try:
        _get_kt_fused_expert_exclude_pattern(model)
    except RuntimeError as exc:
        assert "no forced-fused KT wrappers" in str(exc)
    else:
        raise AssertionError("missing forced-fused KT wrappers must fail")
