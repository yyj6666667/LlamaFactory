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

import importlib.metadata
from pathlib import Path

import pytest
import yaml


_ROOT = Path(__file__).parents[2]
_KT_EXAMPLES = _ROOT / "examples" / "ktransformers"
_ADVANCED_KEYS = {
    "kt_backend",
    "kt_expert_weight_format",
    "kt_force_fused_expert_lora",
    "kt_max_cache_depth",
    "kt_model_max_length",
    "kt_num_gpu_experts",
    "kt_num_threads",
    "kt_share_backward_bb",
    "kt_threadpool_count",
    "kt_tp_enabled",
    "kt_weight_lifecycle",
}


def _load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)

    assert isinstance(config, dict), path
    return config


def test_kt_accelerate_examples_are_generic_fsdp2_configs():
    paths = sorted((_KT_EXAMPLES / "accelerate").glob("*.yaml"))
    assert [path.name for path in paths] == ["fsdp2_kt.yaml"]
    for path in paths:
        config = _load_yaml(path)
        assert "kt_config" not in config, path
        assert config["distributed_type"] == "FSDP", path
        assert config["fsdp_config"]["fsdp_version"] == 2, path
        assert not config["fsdp_config"].get("fsdp_activation_checkpointing", False), path


def test_kt_training_examples_own_one_flat_kernel_config():
    paths = sorted((_KT_EXAMPLES / "train_lora").glob("*.yaml"))
    assert [path.name for path in paths] == [
        "deepseek_v3_1_int8_lora_sft_kt.yaml",
        "qwen3_5moe_lora_sft_kt.yaml",
    ]
    for path in paths:
        config = _load_yaml(path)
        kernel_config = config.get("kt_config")
        assert config.get("use_kt") is True, path
        assert config.get("disable_gradient_checkpointing") is False, path
        assert "gradient_checkpointing" not in config, path
        assert "gradient_checkpointing_kwargs" not in config, path
        assert config.get("save_only_model") is True, path
        assert str(config.get("save_strategy", "no")).lower() == "no", path
        assert not config.get("resume_from_checkpoint"), path
        assert not config.get("adapter_name_or_path"), path
        assert not config.get("load_best_model_at_end", False), path
        assert isinstance(kernel_config, dict) and kernel_config, path
        assert set(kernel_config) <= _ADVANCED_KEYS, path
        assert all(not isinstance(value, (dict, list)) for value in kernel_config.values()), path
        assert kernel_config["kt_model_max_length"] >= config["cutoff_len"], path


def test_routed_int8_example_declares_required_weight_paths_and_format():
    config = _load_yaml(_KT_EXAMPLES / "train_lora" / "deepseek_v3_1_int8_lora_sft_kt.yaml")
    assert config["kt_weight_path"].startswith("/")
    assert config["kt_non_expert_weight_path"].startswith("/")
    assert config["kt_config"]["kt_expert_weight_format"] == "int8"
    assert config["kt_config"]["kt_weight_lifecycle"] == "persistent"


def test_qwen35_example_requires_release_text_only_model_directory():
    config = _load_yaml(_KT_EXAMPLES / "train_lora" / "qwen3_5moe_lora_sft_kt.yaml")
    model_path = config["model_name_or_path"]
    assert model_path.startswith("/")
    assert model_path.endswith("-TEXTONLY")
    assert not model_path.startswith("Qwen/")


@pytest.mark.parametrize("path", sorted((_KT_EXAMPLES / "train_lora").glob("*.yaml")), ids=lambda path: path.stem)
def test_kt_training_examples_parse_with_frozen_frontend(path: Path, monkeypatch: pytest.MonkeyPatch):
    try:
        transformers_version = importlib.metadata.version("transformers-kt")
        accelerate_version = importlib.metadata.version("accelerate-kt")
    except importlib.metadata.PackageNotFoundError:
        pytest.skip("frozen KT frontend distributions are not installed")

    if (transformers_version, accelerate_version) != ("5.6.0.post1", "1.14.0.post1"):
        pytest.skip("tests require the reviewed frozen KT frontend distributions")

    pytest.importorskip("kt_kernel")
    from llamafactory.hparams import get_train_args

    monkeypatch.delenv("ACCELERATE_USE_KT", raising=False)
    config = _load_yaml(path)
    model_args, _, training_args, _, _ = get_train_args(config)
    resolved = model_args._kt_resolved_config
    expected_gpu = "retain" if config["disable_gradient_checkpointing"] else "recompute"
    expected_cpu = config.get("kt_cpu_activation", expected_gpu)
    assert resolved["kt_activation_policy"] == {"cpu": expected_cpu, "gpu": expected_gpu}
    assert resolved["kt_backend"] == config["kt_config"]["kt_backend"]
    assert training_args.accelerator_config.kt_config == {"enabled": True, "kt_config": resolved}
