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
from unittest.mock import Mock, patch

import torch
from transformers import Seq2SeqTrainer

from llamafactory.train.sft.trainer import CustomSeq2SeqTrainer


def _bare_trainer(*, is_main_process: bool) -> CustomSeq2SeqTrainer:
    trainer = object.__new__(CustomSeq2SeqTrainer)
    trainer.model = torch.nn.Linear(2, 2)
    trainer.model_args = SimpleNamespace(_kt_resolved_config={})
    trainer._lf_use_kt = True
    trainer.args = SimpleNamespace(
        output_dir="/tmp/kt-adapter",
        should_save=is_main_process,
        push_to_hub=False,
        hub_revision=None,
    )
    trainer.accelerator = SimpleNamespace(
        is_main_process=is_main_process,
        unwrap_model=lambda model, keep_torch_compile=False: model,
        wait_for_everyone=Mock(),
    )
    return trainer


def test_kt_fsdp2_save_gathers_on_all_ranks_and_saves_only_on_rank_zero():
    adapter_state = {"base_model.model.lora_A.default.weight": torch.ones(2, 2)}
    for is_main_process in (False, True):
        trainer = _bare_trainer(is_main_process=is_main_process)
        with (
            patch("llamafactory.train.sft.trainer.is_kt_fsdp2_peft", return_value=True),
            patch(
                "llamafactory.train.sft.trainer.get_kt_fsdp2_adapter_state_dict", return_value=adapter_state
            ) as gather,
            patch("llamafactory.train.sft.trainer.publish_kt_int8_adapter_manifest") as publish_manifest,
            patch.object(Seq2SeqTrainer, "_save") as parent_save,
        ):
            trainer.save_model()

        gather.assert_called_once_with(trainer.model)
        trainer.accelerator.wait_for_everyone.assert_called_once_with()
        if is_main_process:
            parent_save.assert_called_once_with(output_dir="/tmp/kt-adapter", state_dict=adapter_state)
            publish_manifest.assert_called_once_with(trainer.model, "/tmp/kt-adapter", trainer.model_args)
        else:
            parent_save.assert_not_called()
            publish_manifest.assert_not_called()


def test_kt_fsdp2_rank_zero_save_failure_is_exchanged_before_barrier():
    trainer = _bare_trainer(is_main_process=True)
    with (
        patch("llamafactory.train.sft.trainer.is_kt_fsdp2_peft", return_value=True),
        patch("llamafactory.train.sft.trainer.get_kt_fsdp2_adapter_state_dict", return_value={}),
        patch("llamafactory.train.sft.trainer.publish_kt_int8_adapter_manifest"),
        patch.object(Seq2SeqTrainer, "_save", side_effect=OSError("disk full")),
    ):
        try:
            trainer.save_model()
        except RuntimeError as exc:
            assert "OSError: disk full" in str(exc)
            assert isinstance(exc.__cause__, OSError)
        else:
            raise AssertionError("expected the synchronized adapter-save error")

    trainer.accelerator.wait_for_everyone.assert_not_called()
