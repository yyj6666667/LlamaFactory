# Copyright 2026 the LlamaFactory team.
# Licensed under the Apache License, Version 2.0.

"""Opt-in packaging profile; never imported by the training runtime."""

import ast
import os
from pathlib import Path

from hatchling.metadata.plugin.interface import MetadataHookInterface
from packaging.requirements import Requirement


def get_profile():
    profile = os.environ.get("LLAMAFACTORY_BUILD_PROFILE", "standard")
    if profile not in {"standard", "kt"}:
        raise ValueError("LLAMAFACTORY_BUILD_PROFILE must be 'standard' or 'kt'.")
    return profile


def get_version():
    tree = ast.parse((Path(__file__).parent / "src/llamafactory/extras/env.py").read_text())
    version = next(
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "VERSION" for t in node.targets)
    )
    return f"{version}+kt.20260912" if get_profile() == "kt" else version


class CustomMetadataHook(MetadataHookInterface):
    def update(self, metadata):
        dependencies = list(self.config["dependencies"])
        if get_profile() == "kt":
            replacements = self.config["kt-dependencies"]
            original_names = {Requirement(item).name for item in dependencies}
            if not replacements.keys() <= original_names:
                raise ValueError("KT dependency overrides must name existing default dependencies.")
            dependencies = [replacements.get(Requirement(item).name, item) for item in dependencies]
        metadata["dependencies"] = dependencies
