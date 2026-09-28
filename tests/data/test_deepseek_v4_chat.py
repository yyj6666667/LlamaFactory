# Copyright 2026 the LlamaFactory team.
# Licensed under the Apache License, Version 2.0.

import json
import os
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
from transformers import AutoTokenizer

from llamafactory.data.converter import AlpacaDatasetConverter, SharegptDatasetConverter
from llamafactory.data.parser import DatasetAttr
from llamafactory.data.processor.supervised import SupervisedDatasetProcessor
from llamafactory.data.template import TEMPLATES


REFERENCE = json.loads((Path(__file__).parent / "fixtures/deepseek_v4_chat.json").read_text())


@pytest.fixture(scope="module")
def tokenizer():
    path = os.environ.get("DSV4_TOKENIZER_PATH", REFERENCE["model"])
    return AutoTokenizer.from_pretrained(path, revision=REFERENCE["revision"])


@pytest.mark.parametrize("case", REFERENCE["cases"])
def test_training_and_saved_template_match_official_chat_tokens_and_labels(tokenizer, case):
    template = deepcopy(TEMPLATES["deepseek4_nothink"])
    template.fix_special_tokens(tokenizer)
    template.fix_jinja_template(tokenizer)
    data_args = SimpleNamespace(cutoff_len=1024, mask_history=False, train_on_prompt=False)
    processor = SupervisedDatasetProcessor(template=template, tokenizer=tokenizer, processor=None, data_args=data_args)
    ids, labels = processor._encode_data_example(
        case["messages"][:-1], case["messages"][-1:], case["system"], None, [], [], []
    )
    assert ids == case["input_ids"]
    assert labels == case["labels"]
    messages = ([{"role": "system", "content": case["system"]}] if case["system"] else []) + case["messages"]
    assert tokenizer.apply_chat_template(messages, tokenize=False) == case["rendered"]
    assert tokenizer.apply_chat_template(messages, tokenize=True, return_dict=False) == ids
    prompt, answer = template.encode_oneturn(tokenizer, case["messages"], case["system"])
    assert (
        tokenizer.apply_chat_template(messages[:-1], tokenize=True, return_dict=False, add_generation_prompt=True)
        == prompt
    )
    assert prompt + answer == ids
    assert tokenizer.eos_token_id == 1
    assert ids[0] == tokenizer.bos_token_id == 0


@pytest.mark.parametrize("tools,role", [("[]", "user"), (None, "function"), (None, "observation")])
def test_ordinary_chat_template_rejects_tool_paths(tokenizer, tools, role):
    template = TEMPLATES["deepseek4_nothink"]
    with pytest.raises(ValueError, match="without tools"):
        template.encode_oneturn(tokenizer, [{"role": role, "content": "x"}], tools=tools)


@pytest.mark.parametrize("formatting", ["alpaca", "sharegpt"])
def test_raw_multiturn_dataset_entry_matches_official_labels(tokenizer, formatting):
    case = REFERENCE["cases"][1]
    data_args = SimpleNamespace(cutoff_len=1024, mask_history=False, train_on_prompt=False, media_dir=".")
    attr = DatasetAttr(load_from="file", dataset_name="example", formatting=formatting, system="system")
    if formatting == "alpaca":
        attr.history = "history"
        raw = {
            "system": case["system"],
            "history": [["1+1？", "2"]],
            "instruction": "再加一呢？",
            "input": "",
            "output": "3",
        }
        normalized = AlpacaDatasetConverter(attr, data_args)(raw)
    else:
        raw = {
            "system": case["system"],
            "conversations": [
                {"from": "human" if m["role"] == "user" else "gpt", "value": m["content"]} for m in case["messages"]
            ],
        }
        normalized = SharegptDatasetConverter(attr, data_args)(raw)
    template = deepcopy(TEMPLATES["deepseek4_nothink"])
    template.fix_special_tokens(tokenizer)
    processor = SupervisedDatasetProcessor(template=template, tokenizer=tokenizer, processor=None, data_args=data_args)
    ids, labels = processor._encode_data_example(
        normalized["_prompt"], normalized["_response"], normalized["_system"], normalized["_tools"], [], [], []
    )
    assert ids == case["input_ids"]
    assert labels == case["labels"]
