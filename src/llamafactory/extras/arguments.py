# Copyright 2026 the LlamaFactory team.
# Licensed under the Apache License, Version 2.0.

"""Configuration reading shared by the launcher and argument parser."""

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf


def read_args(args: dict[str, Any] | list[str] | None = None) -> dict[str, Any] | list[str]:
    r"""Get arguments from the command line or a config file."""
    if args is not None:
        return args

    if len(sys.argv) > 1 and (sys.argv[1].endswith(".yaml") or sys.argv[1].endswith(".yml")):
        override_config = OmegaConf.from_cli(sys.argv[2:])
        dict_config = OmegaConf.load(Path(sys.argv[1]).absolute())
        return OmegaConf.to_container(OmegaConf.merge(dict_config, override_config))
    elif len(sys.argv) > 1 and sys.argv[1].endswith(".json"):
        override_config = OmegaConf.from_cli(sys.argv[2:])
        dict_config = OmegaConf.create(json.loads(Path(sys.argv[1]).read_text(encoding="utf-8")))
        return OmegaConf.to_container(OmegaConf.merge(dict_config, override_config))
    else:
        return sys.argv[1:]


@dataclass
class _BackendArguments:
    use_kt: bool = False
    kt_config: str | None = None
    accelerator_config: str | None = None


def _check_disabled_kt(args: dict[str, Any]) -> None:
    r"""Reject implicit KT activation before TrainingArguments can initialize it."""
    if args.get("kt_config") is not None:
        raise ValueError("`kt_config` requires `use_kt: true`; remove it for non-KT execution.")
    if os.getenv("ACCELERATE_USE_KT", "false").lower() in {"1", "true", "yes", "on", "y", "t"}:
        raise ValueError("`ACCELERATE_USE_KT` conflicts with `use_kt: false`; unset it for non-KT execution.")

    config = args.get("accelerator_config")
    if isinstance(config, str):
        config = json.loads(config if config.startswith("{") else Path(config).read_text(encoding="utf-8"))
    kt_config = config.get("kt_config") if isinstance(config, dict) else getattr(config, "kt_config", None)
    if kt_config is not None:
        raise ValueError("`accelerator_config.kt_config` requires `use_kt: true`; remove it for non-KT execution.")


def get_use_kt(args: dict[str, Any] | list[str]) -> bool:
    r"""Read the backend choice without constructing training arguments."""
    if isinstance(args, dict):
        enabled = args.get("use_kt", False)
        if not isinstance(enabled, bool):
            raise ValueError("`use_kt` must be a YAML/JSON boolean.")
        if not enabled:
            _check_disabled_kt(args)
        return enabled

    from transformers import HfArgumentParser

    parser = HfArgumentParser(_BackendArguments, add_help=False)
    backend, _ = parser.parse_args_into_dataclasses(args=args, return_remaining_strings=True)
    if not backend.use_kt:
        _check_disabled_kt(vars(backend))
    return backend.use_kt
