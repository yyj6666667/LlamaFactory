# Copyright 2026 the LlamaFactory team.
# Licensed under the Apache License, Version 2.0.

"""Configuration reading shared by the launcher and argument parser."""

import json
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


def get_use_kt(args: dict[str, Any] | list[str]) -> bool:
    r"""Read the backend choice without constructing training arguments."""
    if isinstance(args, dict):
        enabled = args.get("use_kt", False)
        if not isinstance(enabled, bool):
            raise ValueError("`use_kt` must be a YAML/JSON boolean.")
        return enabled

    from transformers import HfArgumentParser

    parser = HfArgumentParser(_BackendArguments, add_help=False)
    backend, _ = parser.parse_args_into_dataclasses(args=args, return_remaining_strings=True)
    return backend.use_kt
