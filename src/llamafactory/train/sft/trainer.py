# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/trainer_seq2seq.py
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

import json
import os
import time
from functools import partial
from types import MethodType
from typing import TYPE_CHECKING, Any, Optional, Union

import numpy as np
import torch
from transformers import Seq2SeqTrainer
from transformers.training_args import OptimizerNames
from transformers.utils import is_sagemaker_mp_enabled
from typing_extensions import override

from accelerate.utils import DistributedType
from accelerate.utils.memory import clear_device_cache

from ...extras import logging
from ...extras.constants import IGNORE_INDEX
from ..callbacks import SaveProcessorCallback
from ..fp8_utils import configure_fp8_environment, patch_accelerator_for_fp8, verify_fp8_status
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler


if TYPE_CHECKING:
    from torch.utils.data import Dataset
    from transformers import ProcessorMixin
    from transformers.trainer import PredictionOutput

    from ...hparams import FinetuningArguments, ModelArguments, TrainingArguments


logger = logging.get_logger(__name__)


class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    r"""Inherits Seq2SeqTrainer to compute generative metrics such as BLEU and ROUGE."""

    def __init__(
        self,
        finetuning_args: "FinetuningArguments",
        processor: Optional["ProcessorMixin"],
        model_args: Optional["ModelArguments"] = None,
        gen_kwargs: Optional[dict[str, Any]] = None,
        ref_model: Optional["torch.nn.Module"] = None,
        **kwargs,
    ) -> None:
        kwargs["processing_class"] = kwargs.pop("tokenizer")
        # Configure FP8 environment if enabled
        training_args: TrainingArguments = kwargs.get("args")
        if training_args.fp8:
            configure_fp8_environment(training_args)
            if getattr(training_args, "fp8_backend", "auto") == "te":
                patch_accelerator_for_fp8()

        super().__init__(**kwargs)
        if processor is not None:
            # avoid wrong loss under gradient accumulation
            # https://github.com/huggingface/transformers/pull/36044#issuecomment-2746657112
            self.model_accepts_loss_kwargs = False

        self.finetuning_args = finetuning_args
        if gen_kwargs is not None:
            # https://github.com/huggingface/transformers/blob/v4.45.0/src/transformers/trainer_seq2seq.py#L287
            self._gen_kwargs = gen_kwargs

        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))

        if finetuning_args.use_badam:
            from badam import BAdamCallback, clip_grad_norm_old_version  # type: ignore

            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_old_version, self.accelerator)
            self.add_callback(BAdamCallback)

        self.ref_model = ref_model

        if ref_model is not None:
            from trl.models.utils import prepare_deepspeed, prepare_fsdp

            if getattr(self.accelerator.state, "deepspeed_plugin", None) is not None:
                if not (
                    getattr(ref_model, "is_loaded_in_8bit", False) or getattr(ref_model, "is_loaded_in_4bit", False)
                ):  # quantized models are already set on the correct device
                    self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            elif getattr(self.accelerator.state, "fsdp_plugin", None) is not None:
                if self.accelerator.is_fsdp2:
                    from accelerate.utils.fsdp_utils import fsdp2_prepare_model

                    self.ref_model = fsdp2_prepare_model(self.accelerator, self.ref_model)
                else:
                    self.ref_model = prepare_fsdp(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)
                self.ref_model.eval()

        if finetuning_args.use_dft_loss:
            from ..trainer_utils import dft_loss_func

            self.compute_loss_func = dft_loss_func

        elif finetuning_args.use_eaft_loss:
            from ..trainer_utils import eaft_loss_func

            self.compute_loss_func = lambda outputs, labels, num_items_in_batch=None: eaft_loss_func(
                outputs, labels, num_items_in_batch, finetuning_args.eaft_alpha
            )
        elif finetuning_args.use_asft_loss:
            from ..trainer_utils import asft_loss_func

            self.compute_loss_func = partial(
                asft_loss_func,
                asft_alpha=finetuning_args.asft_alpha,
            )

        if training_args.fp8 and hasattr(self, "accelerator"):  # verify FP8 status after trainer initialization
            verify_fp8_status(self.accelerator, training_args)

    def _kt_e2e_timing_enabled(self) -> bool:
        return bool(os.environ.get("KT_E2E_TIMING_JSONL"))

    def _kt_e2e_sync_time(self) -> float:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return time.perf_counter()

    def _kt_e2e_reduce_max_ms(self, value_ms: float) -> float:
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            return value_ms

        device = self.args.device if self.args.device.type != "cpu" else torch.device("cpu")
        value = torch.tensor([value_ms], dtype=torch.float64, device=device)
        torch.distributed.all_reduce(value, op=torch.distributed.ReduceOp.MAX)
        return float(value.item())

    def _kt_e2e_reduce_sum_int(self, value: int) -> int:
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            return value

        device = self.args.device if self.args.device.type != "cpu" else torch.device("cpu")
        tensor = torch.tensor([value], dtype=torch.long, device=device)
        torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
        return int(tensor.item())

    def _kt_e2e_reduce_max_int(self, value: int) -> int:
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            return value

        device = self.args.device if self.args.device.type != "cpu" else torch.device("cpu")
        tensor = torch.tensor([value], dtype=torch.long, device=device)
        torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.MAX)
        return int(tensor.item())

    def _kt_e2e_input_stats(self, inputs: dict[str, Union["torch.Tensor", Any]]) -> dict[str, Any]:
        stats = {
            "batch_size_local": 0,
            "seq_len_local": 0,
            "input_tokens_local": 0,
            "attention_tokens_local": 0,
            "label_tokens_local": 0,
        }
        input_ids = inputs.get("input_ids")
        attention_mask = inputs.get("attention_mask")
        labels = inputs.get("labels")

        if isinstance(input_ids, torch.Tensor):
            stats["input_shape"] = list(input_ids.shape)
            if input_ids.ndim >= 1:
                stats["batch_size_local"] = int(input_ids.shape[0])
            if input_ids.ndim >= 2:
                stats["seq_len_local"] = int(input_ids.shape[1])
            stats["input_tokens_local"] = int(input_ids.numel())

        if isinstance(attention_mask, torch.Tensor):
            try:
                stats["attention_tokens_local"] = int(attention_mask.detach().sum().item())
            except Exception:
                pass

        if isinstance(labels, torch.Tensor):
            try:
                stats["label_tokens_local"] = int((labels.detach() != IGNORE_INDEX).sum().item())
            except Exception:
                pass

        for key in ("input_tokens", "attention_tokens", "label_tokens"):
            local_value = int(stats[f"{key}_local"])
            stats[f"{key}_global"] = self._kt_e2e_reduce_sum_int(local_value)

        for key in ("batch_size", "seq_len"):
            local_value = int(stats[f"{key}_local"])
            stats[f"{key}_max"] = self._kt_e2e_reduce_max_int(local_value)

        return stats

    def _kt_e2e_write_timing(self, record: dict[str, Any]) -> None:
        if not self.is_world_process_zero():
            return

        path = os.environ.get("KT_E2E_TIMING_JSONL")
        if not path:
            return

        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    @override
    def training_step(
        self,
        model: "torch.nn.Module",
        inputs: dict[str, Union["torch.Tensor", Any]],
        num_items_in_batch: Optional[Union["torch.Tensor", int]] = None,
    ) -> "torch.Tensor":
        if not self._kt_e2e_timing_enabled() or is_sagemaker_mp_enabled():
            return super().training_step(model, inputs, num_items_in_batch=num_items_in_batch)

        cp_context, inputs = self._prepare_context_parallel_inputs(model, inputs)

        with cp_context():
            model.train()
            if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
                self.optimizer.train()

            inputs = self._prepare_inputs(inputs)
            input_stats = self._kt_e2e_input_stats(inputs)
            forward_start = self._kt_e2e_sync_time()
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)
            forward_ms_local = (self._kt_e2e_sync_time() - forward_start) * 1000.0

            del inputs
            if (
                self.args.torch_empty_cache_steps is not None
                and self.state.global_step % self.args.torch_empty_cache_steps == 0
            ):
                clear_device_cache()

            kwargs = {}
            if self.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
                kwargs["learning_rate"] = self._get_learning_rate()

            if self.args.n_gpu > 1:
                loss = loss.mean()

            if (not self.model_accepts_loss_kwargs or num_items_in_batch is None) and self.compute_loss_func is None:
                loss = loss / self.current_gradient_accumulation_steps

            if self.accelerator.distributed_type == DistributedType.DEEPSPEED:
                kwargs["scale_wrt_gas"] = False

            backward_start = self._kt_e2e_sync_time()
            self.accelerator.backward(loss, **kwargs)
            backward_ms_local = (self._kt_e2e_sync_time() - backward_start) * 1000.0

            forward_ms = self._kt_e2e_reduce_max_ms(forward_ms_local)
            backward_ms = self._kt_e2e_reduce_max_ms(backward_ms_local)
            self._kt_e2e_write_timing(
                {
                    "record_type": "training_step_timing",
                    "global_step_before": int(self.state.global_step),
                    "sync_gradients": bool(getattr(self.accelerator, "sync_gradients", False)),
                    "gradient_accumulation_steps": int(self.current_gradient_accumulation_steps),
                    "forward_ms": forward_ms,
                    "backward_ms": backward_ms,
                    "backward_over_forward": backward_ms / forward_ms if forward_ms else None,
                    "loss": float(loss.detach().float().item()),
                    **input_stats,
                }
            )

            return loss.detach()

    @override
    def create_optimizer(self) -> "torch.optim.Optimizer":
        if self.optimizer is None:
            self.optimizer = create_custom_optimizer(self.model, self.args, self.finetuning_args)
        return super().create_optimizer()

    @override
    def create_scheduler(
        self, num_training_steps: int, optimizer: Optional["torch.optim.Optimizer"] = None
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        return super().create_scheduler(num_training_steps, optimizer)

    @override
    def _get_train_sampler(self, *args, **kwargs) -> Optional["torch.utils.data.Sampler"]:
        if self.finetuning_args.disable_shuffling:
            return torch.utils.data.SequentialSampler(self.train_dataset)

        return super()._get_train_sampler(*args, **kwargs)

    @override
    def compute_loss(self, model, inputs, *args, **kwargs):
        if self.finetuning_args.use_asft_loss:
            with torch.no_grad():
                ref_outputs = self.ref_model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask", None),
                )
                ref_logits = ref_outputs.logits
            outputs = model(**inputs)
            return self.compute_loss_func(outputs, inputs["labels"], ref_logits)
        else:
            return super().compute_loss(model, inputs, *args, **kwargs)

    @override
    def prediction_step(
        self,
        model: "torch.nn.Module",
        inputs: dict[str, Union["torch.Tensor", Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[list[str]] = None,
        **gen_kwargs,
    ) -> tuple[Optional[float], Optional["torch.Tensor"], Optional["torch.Tensor"]]:
        r"""Remove the prompt part in the generated tokens.

        Subclass and override to inject custom behavior.
        """
        if self.args.predict_with_generate:  # do not pass labels to model when generate
            labels = inputs.pop("labels", None)
        else:
            labels = inputs.get("labels")

        loss, generated_tokens, _ = super().prediction_step(
            model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys, **gen_kwargs
        )
        if generated_tokens is not None and self.args.predict_with_generate:
            generated_tokens[:, : inputs["input_ids"].size(-1)] = self.processing_class.pad_token_id
            generated_tokens = generated_tokens.contiguous()

        return loss, generated_tokens, labels

    def save_predictions(
        self, dataset: "Dataset", predict_results: "PredictionOutput", skip_special_tokens: bool = True
    ) -> None:
        r"""Save model predictions to `output_dir`.

        A custom behavior that not contained in Seq2SeqTrainer.
        """
        if not self.is_world_process_zero():
            return

        output_prediction_file = os.path.join(self.args.output_dir, "generated_predictions.jsonl")
        logger.info_rank0(f"Saving prediction results to {output_prediction_file}")

        labels = np.where(
            predict_results.label_ids != IGNORE_INDEX, predict_results.label_ids, self.processing_class.pad_token_id
        )
        preds = np.where(
            predict_results.predictions != IGNORE_INDEX,
            predict_results.predictions,
            self.processing_class.pad_token_id,
        )

        for i in range(len(preds)):
            pad_len = np.nonzero(preds[i] != self.processing_class.pad_token_id)[0]
            if len(pad_len):  # move pad token to last
                preds[i] = np.concatenate((preds[i][pad_len[0] :], preds[i][: pad_len[0]]), axis=-1)

        input_ids_column = dataset["input_ids"]
        try:
            input_ids_list = input_ids_column.to_pylist()
        except AttributeError:
            input_ids_list = list(input_ids_column)

        decoded_inputs = self.processing_class.batch_decode(input_ids_list, skip_special_tokens=False)
        decoded_preds = self.processing_class.batch_decode(preds, skip_special_tokens=skip_special_tokens)
        decoded_labels = self.processing_class.batch_decode(labels, skip_special_tokens=skip_special_tokens)

        with open(output_prediction_file, "w", encoding="utf-8") as f:
            for text, pred, label in zip(decoded_inputs, decoded_preds, decoded_labels):
                f.write(json.dumps({"prompt": text, "predict": pred, "label": label}, ensure_ascii=False) + "\n")
