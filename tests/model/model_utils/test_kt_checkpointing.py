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
from types import ModuleType

from llamafactory.hparams import ModelArguments
from llamafactory.model.model_utils.checkpointing import _get_gradient_checkpointing_kwargs


def test_non_kt_checkpointing_kwargs_are_unchanged():
    model_args = ModelArguments(model_name_or_path="dummy", use_reentrant_gc=True)
    assert _get_gradient_checkpointing_kwargs(model_args) == {"use_reentrant": True}


def test_kt_checkpointing_uses_non_reentrant_context(monkeypatch):
    context_fn = object()
    kt_kernel = ModuleType("kt_kernel")
    kt_kernel.__path__ = []
    sft = ModuleType("kt_kernel.sft")
    sft.get_activation_checkpoint_context_fn = lambda: context_fn
    monkeypatch.setitem(sys.modules, "kt_kernel", kt_kernel)
    monkeypatch.setitem(sys.modules, "kt_kernel.sft", sft)

    model_args = ModelArguments(model_name_or_path="dummy", use_kt=True, kt_cpu_activation="retain")
    assert _get_gradient_checkpointing_kwargs(model_args) == {
        "use_reentrant": False,
        "context_fn": context_fn,
    }
