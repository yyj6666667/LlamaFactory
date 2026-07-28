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

from llamafactory.model.adapter import _load_kt_adapter_with_manifest


def test_kt_adapter_load_is_manifest_gated(monkeypatch):
    events = []
    kt_sft = ModuleType("kt_kernel.sft")
    kt_sft.kt_adapt_peft_lora = lambda model: events.append(("adapt", model))

    def raw_load(model, path):
        events.append(("raw_load", model, path))

    kt_sft.load_kt_moe_from_adapter = raw_load
    artifacts = ModuleType("transformers.integrations.kt_artifacts")

    def validated_load(model, path, callback):
        events.append(("validate", model, path, callback))
        callback(model, path)

    artifacts.load_kt_adapter_artifacts = validated_load
    kt_kernel = ModuleType("kt_kernel")
    kt_kernel.sft = kt_sft
    monkeypatch.setitem(sys.modules, "kt_kernel", kt_kernel)
    monkeypatch.setitem(sys.modules, "kt_kernel.sft", kt_sft)
    monkeypatch.setitem(sys.modules, "transformers.integrations.kt_artifacts", artifacts)

    model = SimpleNamespace()
    _load_kt_adapter_with_manifest(model, "/tmp/adapter")

    assert events == [
        ("adapt", model),
        ("validate", model, "/tmp/adapter", raw_load),
        ("raw_load", model, "/tmp/adapter"),
    ]
    assert model._kt_adapter_loaded is True
