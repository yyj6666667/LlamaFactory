# Copyright 2026 the LlamaFactory team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

import ast
import importlib.metadata
import tomllib
from pathlib import Path
from unittest.mock import patch

import pytest
from packaging import version
from packaging.requirements import Requirement


ROOT = Path(__file__).resolve().parents[2]
EXPECTED = {
    "transformers-kt": "5.6.0.post5",
    "accelerate-kt": "1.14.0.post3",
    "peft": "0.18.1+kt.20260912",
    "trl": "0.24.0+kt.20260912",
}


def source_function(filename, name, namespace):
    tree = ast.parse((ROOT / "src/llamafactory/extras" / filename).read_bytes())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
    exec(compile(ast.Module(body=[function], type_ignores=[]), filename, "exec"), namespace)
    return namespace[name]


def test_release_metadata_and_runtime_checks_agree():
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text())
    dependencies = {item.name: item for item in map(Requirement, metadata["project"]["dependencies"])}
    assert "transformers" not in dependencies
    assert "accelerate" not in dependencies
    checks = []
    check = source_function("misc.py", "check_dependencies", {"check_version": checks.append})
    check()
    runtime = {item.name: item for item in map(Requirement, checks)}
    for name, expected in EXPECTED.items():
        assert str(dependencies[name].specifier) == str(runtime[name].specifier) == f"=={expected}"
        assert dependencies[name].specifier.contains(expected)
        assert not dependencies[name].specifier.contains("999.0.0")
    for name in ("peft", "trl"):
        assert not dependencies[name].specifier.contains(EXPECTED[name].split("+")[0])


def test_release_has_distinct_version_without_build_profile():
    tree = ast.parse((ROOT / "src/llamafactory/extras/env.py").read_bytes())
    value = next(
        node.value.value for node in tree.body if isinstance(node, ast.Assign) and node.targets[0].id == "VERSION"
    )
    assert value == "0.9.6.dev0+kt.20260912"
    config = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert config["build-system"]["build-backend"] == "hatchling.build"
    assert config["project"]["dynamic"] == ["version"]


@pytest.mark.parametrize("name,distribution", [("transformers", "transformers-kt"), ("torch", "torch")])
def test_version_detection_uses_installed_distribution(name, distribution):
    get_version = source_function("packages.py", "_get_package_version", {"version": version, "importlib": importlib})
    with patch("importlib.metadata.version", return_value="5.6.0.post5") as lookup:
        assert get_version(name) == version.parse("5.6.0.post5")
        lookup.assert_called_once_with(distribution)


def test_missing_dependency_is_not_reported_as_supported():
    get_version = source_function("packages.py", "_get_package_version", {"version": version, "importlib": importlib})
    with patch("importlib.metadata.version", side_effect=importlib.metadata.PackageNotFoundError):
        assert get_version("transformers") == version.parse("0.0.0")
