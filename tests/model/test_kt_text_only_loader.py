# Copyright 2026 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import json
from types import SimpleNamespace

import pytest

from llamafactory.model.loader import _is_kt_text_only_qwen35_sft


def _model(**overrides):
    config = {
        "model_type": "qwen3_5_moe",
        "image_token_id": None,
        "video_token_id": None,
    }
    config.update(overrides)
    return SimpleNamespace(config=SimpleNamespace(**config))


def _arguments(model_path, **overrides):
    values = {"use_kt": True, "stage": "sft", "finetuning_type": "lora", "model_name_or_path": str(model_path)}
    values.update(overrides)
    return (
        SimpleNamespace(use_kt=values.pop("use_kt"), model_name_or_path=values.pop("model_name_or_path")),
        SimpleNamespace(**values),
    )


def _write_config(model_path, **overrides):
    model_path.mkdir()
    config = {"model_type": "qwen3_5_moe"}
    config.update(overrides)
    (model_path / "config.json").write_text(json.dumps(config), encoding="utf-8")


def test_detects_kt_text_only_qwen35_lora(tmp_path):
    _write_config(tmp_path / "model")
    model_args, finetuning_args = _arguments(tmp_path / "model")
    assert _is_kt_text_only_qwen35_sft(_model(), model_args, finetuning_args)


@pytest.mark.parametrize(
    ("model_overrides", "argument_overrides", "raw_config_overrides"),
    [
        ({"model_type": "deepseek_v3"}, {}, {}),
        ({}, {}, {"image_token_id": 42}),
        ({}, {}, {"video_token_id": 43}),
        ({}, {"use_kt": False}, {}),
        ({}, {"stage": "pt"}, {}),
        ({}, {"finetuning_type": "full"}, {}),
    ],
)
def test_rejects_non_text_only_or_non_kt_lora(tmp_path, model_overrides, argument_overrides, raw_config_overrides):
    _write_config(tmp_path / "model", **raw_config_overrides)
    model_args, finetuning_args = _arguments(tmp_path / "model", **argument_overrides)
    assert not _is_kt_text_only_qwen35_sft(_model(**model_overrides), model_args, finetuning_args)
