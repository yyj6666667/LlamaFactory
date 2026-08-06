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

from llamafactory.hparams import DataArguments, ModelArguments
from llamafactory.hparams.parser import _get_kt_runtime_capacity


@pytest.mark.parametrize(
    ("stage", "do_train", "do_eval", "packing", "cutoff_len", "train_batch", "eval_batch", "expected"),
    [
        ("sft", True, False, False, 1023, 1, 1, 1024),
        ("sft", True, False, False, 1024, 2, 1, 2048),
        ("sft", True, False, True, 1024, 1, 1, 1032),
        ("sft", True, False, True, 1024, 2, 1, 2064),
        ("sft", False, True, True, 1024, 1, 2, 2050),
        ("sft", True, True, False, 1024, 1, 4, 4096),
        ("pt", True, False, True, 1024, 2, 1, 2048),
    ],
)
def test_kt_runtime_capacity(stage, do_train, do_eval, packing, cutoff_len, train_batch, eval_batch, expected):
    data_args = SimpleNamespace(cutoff_len=cutoff_len, packing=packing)
    training_args = SimpleNamespace(
        do_train=do_train,
        do_eval=do_eval,
        do_predict=False,
        per_device_train_batch_size=train_batch,
        per_device_eval_batch_size=eval_batch,
    )
    finetuning_args = SimpleNamespace(stage=stage)
    assert _get_kt_runtime_capacity(data_args, training_args, finetuning_args) == expected


def test_kt_runtime_capacity_uses_internal_packed_cutoff():
    data_args = DataArguments(cutoff_len=1024, packing=True)
    training_args = SimpleNamespace(
        do_train=True,
        do_eval=False,
        do_predict=False,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
    )
    assert data_args.cutoff_len == 1023
    assert _get_kt_runtime_capacity(data_args, training_args, SimpleNamespace(stage="sft")) == 1024


@pytest.mark.parametrize(("computed", "configured", "expected"), [(1024, "1152", 1152), (2048, "1024", 2048)])
def test_kt_runtime_capacity_preserves_configured_headroom(monkeypatch, computed, configured, expected):
    monkeypatch.setenv("ACCELERATE_KT_MODEL_MAX_LENGTH", configured)
    model_args = ModelArguments(model_name_or_path="dummy", use_kt=True)
    config = model_args.get_kt_config_dict(SimpleNamespace(lora_rank=8, lora_alpha=16), computed)
    assert config["kt_model_max_length"] == expected


@pytest.mark.parametrize("configured", ["invalid", "0", "-1"])
def test_kt_runtime_capacity_rejects_invalid_headroom(monkeypatch, configured):
    monkeypatch.setenv("ACCELERATE_KT_MODEL_MAX_LENGTH", configured)
    model_args = ModelArguments(model_name_or_path="dummy", use_kt=True)
    with pytest.raises(ValueError, match="must be a positive integer"):
        model_args.get_kt_config_dict(SimpleNamespace(lora_rank=8, lora_alpha=16), 1024)
