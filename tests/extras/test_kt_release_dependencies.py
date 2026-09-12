# Copyright 2026 the LlamaFactory team.
# Licensed under the Apache License, Version 2.0.

import ast
import importlib.metadata
import importlib.util
import os
import sys
import tomllib
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from packaging import version
from packaging.requirements import Requirement


ROOT = Path(__file__).resolve().parents[2]
STANDARD_CHECKS = [
    "transformers>=4.55.0,<=5.8.0,!=4.57.0,!=5.6.0",
    "datasets>=2.16.0,<=4.0.0",
    "accelerate>=1.3.0,<=1.15.0",
    "peft>=0.18.0,<=0.20.0",
    "trl>=0.18.0,<=0.24.0",
]
KT_CHECKS = [
    "transformers-kt==5.6.0.post5",
    "datasets>=2.16.0,<=4.0.0",
    "accelerate-kt==1.14.0.post3",
    "peft==0.18.1+kt.20260912",
    "trl==0.24.0+kt.20260912",
]


def source_function(filename, name, namespace):
    """Load dependency helpers without importing the full training stack."""
    tree = ast.parse((ROOT / "src/llamafactory/extras" / filename).read_bytes())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
    exec(compile(ast.Module(body=[function], type_ignores=[]), filename, "exec"), namespace)
    return namespace[name]


def load_module(filename, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("profile", ["standard", "kt"])
def test_build_profile_changes_only_selected_metadata(monkeypatch, profile):
    monkeypatch.setenv("LLAMAFACTORY_BUILD_PROFILE", profile)
    hook_module = load_module("hatch_build.py", "_lf_build_hook")
    config = tomllib.loads((ROOT / "pyproject.toml").read_text())["tool"]["hatch"]["metadata"]["hooks"]["custom"]
    metadata = {}
    hook_module.CustomMetadataHook(str(ROOT), config).update(metadata)
    expected = [
        config["kt-dependencies"].get(Requirement(item).name, item) if profile == "kt" else item
        for item in config["dependencies"]
    ]
    assert metadata == {"dependencies": expected}
    expected_version = "0.9.6.dev0+kt.20260912" if profile == "kt" else "0.9.6.dev0"
    assert hook_module.get_version() == expected_version
    names = {Requirement(item).name for item in metadata["dependencies"]}
    assert ("transformers-kt" in names) == (profile == "kt")
    assert ("transformers" in names) == (profile == "standard")


def test_default_profile_does_not_use_runtime_environment(monkeypatch):
    monkeypatch.delenv("LLAMAFACTORY_BUILD_PROFILE", raising=False)
    monkeypatch.setenv("USE_KT", "1")
    hook_module = load_module("hatch_build.py", "_lf_build_hook")
    assert hook_module.get_profile() == "standard"
    monkeypatch.setenv("LLAMAFACTORY_BUILD_PROFILE", "typo")
    with pytest.raises(ValueError, match="LLAMAFACTORY_BUILD_PROFILE"):
        hook_module.get_profile()


@pytest.mark.parametrize("enabled,installed", [(False, False), (True, False), (False, True), (True, True)])
def test_checks_match_provider_but_do_not_enable_backend(enabled, installed):
    checks = []
    namespace = {
        "check_version": checks.append,
        "is_transformers_kt_available": lambda: installed,
        "importlib": importlib,
    }
    check = source_function("misc.py", "check_dependencies", namespace)
    with patch("importlib.metadata.version", side_effect=importlib.metadata.PackageNotFoundError):
        check(use_kt=enabled)
    assert checks == (KT_CHECKS if enabled or installed else STANDARD_CHECKS)


@pytest.mark.parametrize("upstream", ["transformers", "accelerate"])
def test_kt_rejects_overlapping_distributions(upstream):
    def distribution(name):
        if name == upstream:
            return "5.6.0"
        raise importlib.metadata.PackageNotFoundError(name)

    namespace = {"check_version": lambda _: None, "is_transformers_kt_available": lambda: True, "importlib": importlib}
    check = source_function("misc.py", "check_dependencies", namespace)
    with patch("importlib.metadata.version", side_effect=distribution), pytest.raises(ImportError, match=upstream):
        check()


def test_version_query_uses_actual_distribution_not_environment(monkeypatch):
    monkeypatch.setenv("USE_KT", "1")
    get_version = source_function("packages.py", "_get_package_version", {"version": version, "importlib": importlib})
    with patch("importlib.metadata.version", return_value="5.8.0") as lookup:
        assert get_version("transformers") == version.parse("5.8.0")
        lookup.assert_called_once_with("transformers")
    with patch("importlib.metadata.version", side_effect=[importlib.metadata.PackageNotFoundError(), "5.6.0.post5"]):
        assert get_version("transformers") == version.parse("5.6.0.post5")
    with patch("importlib.metadata.version", side_effect=importlib.metadata.PackageNotFoundError) as lookup:
        assert get_version("torch") == version.parse("0.0.0")
        lookup.assert_called_once_with("torch")


@pytest.mark.parametrize("enabled", [False, True])
def test_dependency_check_failures_propagate(enabled):
    def reject(requirement):
        raise importlib.metadata.PackageNotFoundError(requirement)

    namespace = {"check_version": reject, "is_transformers_kt_available": lambda: False, "importlib": importlib}
    check = source_function("misc.py", "check_dependencies", namespace)
    with patch("importlib.metadata.version", side_effect=importlib.metadata.PackageNotFoundError):
        with pytest.raises(importlib.metadata.PackageNotFoundError):
            check(use_kt=enabled)


@pytest.fixture
def arguments():
    return load_module("src/llamafactory/extras/arguments.py", "_lf_arguments")


@pytest.mark.parametrize("extension", ["yaml", "json"])
@pytest.mark.parametrize("override", [True, False])
def test_yaml_json_cli_override_and_environment_independence(arguments, tmp_path, monkeypatch, extension, override):
    path = tmp_path / f"train.{extension}"
    path.write_text('{"use_kt": true, "cutoff_len": 512}')
    monkeypatch.setenv("USE_KT", "0" if override else "1")
    monkeypatch.setattr(sys, "argv", ["lf", str(path), f"use_kt={str(override).lower()}", "cutoff_len=1024"])
    args = arguments.read_args()
    assert arguments.get_use_kt(args) is override
    assert args["cutoff_len"] == 1024


@pytest.mark.parametrize("args", [{}, {"use_kt": False}, {"use_kt": True}])
def test_dict_backend_is_explicit(arguments, args):
    assert arguments.read_args(args) is args
    assert arguments.get_use_kt(args) is args.get("use_kt", False)


@pytest.mark.parametrize("value", ["false", "true", None, 1])
def test_invalid_yaml_boolean_fails(arguments, value):
    with pytest.raises(ValueError, match="boolean"):
        arguments.get_use_kt({"use_kt": value})


@pytest.mark.parametrize(
    "args,expected",
    [
        ([], False),
        (["--help"], False),
        (["--use_kt"], True),
        (["--use-kt", "false"], False),
        (["--use_kt=true"], True),
        (["--use_kt", "true", "--use_kt", "false"], False),
        (["--cutoff_len", "512", "--use_kt", "true"], True),
    ],
)
def test_cli_uses_hf_boolean_semantics(arguments, args, expected):
    pytest.importorskip("transformers")
    assert arguments.get_use_kt(args) is expected


@pytest.mark.parametrize("config", [{"kt_config": {}}, {"accelerator_config": {"kt_config": {}}}])
@pytest.mark.parametrize("enabled", [False, True])
def test_implicit_kt_config_cannot_override_yaml(arguments, config, enabled):
    args = config | {"use_kt": enabled}
    if enabled:
        assert arguments.get_use_kt(args)
    else:
        with pytest.raises(ValueError, match="requires `use_kt: true`"):
            arguments.get_use_kt(args)
    assert args == config | {"use_kt": enabled}


@pytest.mark.parametrize("flag", ["--kt_config", "--kt-config", "--accelerator_config", "--accelerator-config"])
def test_implicit_cli_config_is_rejected(arguments, flag):
    pytest.importorskip("transformers")
    value = '{"kt_config": {}}' if "accelerator" in flag else "{}"
    with pytest.raises(ValueError, match="requires `use_kt: true`"):
        arguments.get_use_kt([flag, value, "--use_kt", "false"])


def test_accelerator_config_file_and_ordinary_config(arguments, tmp_path):
    pytest.importorskip("transformers")
    path = tmp_path / "accelerator.json"
    path.write_text('{"kt_config": {}}')
    with pytest.raises(ValueError, match="accelerator_config.kt_config"):
        arguments.get_use_kt({"accelerator_config": str(path)})
    for config in ({"split_batches": False}, '{"split_batches": false}', None):
        assert arguments.get_use_kt({"accelerator_config": config}) is False
    path.write_text('{"split_batches": false}')
    assert arguments.get_use_kt(["--accelerator_config", str(path)]) is False


@pytest.mark.parametrize("value", ["true", "1", "yes", "false", "0"])
def test_accelerate_environment_cannot_silently_enable_kt(arguments, monkeypatch, value):
    monkeypatch.setenv("ACCELERATE_USE_KT", value)
    if value in {"true", "1", "yes"}:
        with pytest.raises(ValueError, match="ACCELERATE_USE_KT"):
            arguments.get_use_kt({"use_kt": False})
    else:
        assert arguments.get_use_kt({"use_kt": False}) is False
    assert arguments.get_use_kt({"use_kt": True}) is True
    assert os.environ["ACCELERATE_USE_KT"] == value


@pytest.mark.parametrize("enabled", [False, True])
def test_launcher_dispatch_uses_yaml_override_not_environment(arguments, tmp_path, monkeypatch, enabled):
    config = tmp_path / "train.yaml"
    config.write_text("use_kt: true\n")
    monkeypatch.setenv("USE_KT", "0" if enabled else "1")
    monkeypatch.setattr(sys, "argv", ["lf", "train", str(config), f"use_kt={str(enabled).lower()}"])
    dependency_check = Mock()
    run_exp = Mock()
    misc = SimpleNamespace(
        check_dependencies=dependency_check,
        find_available_port=lambda: 12345,
        get_device_count=lambda: 2,
        is_env_enabled=lambda *args: False,
        use_ray=lambda: False,
    )
    modules = {
        "llamafactory": ModuleType("llamafactory"),
        "llamafactory.extras": SimpleNamespace(logging=SimpleNamespace(get_logger=lambda _: Mock())),
        "llamafactory.extras.arguments": arguments,
        "llamafactory.extras.env": SimpleNamespace(VERSION="0.9.6.dev0", print_env=Mock()),
        "llamafactory.extras.misc": misc,
        "llamafactory.train.tuner": SimpleNamespace(run_exp=run_exp),
    }
    with (
        patch.dict(sys.modules, modules),
        patch("subprocess.run", return_value=SimpleNamespace(returncode=0)) as spawn,
    ):
        launcher = load_module("src/llamafactory/launcher.py", "llamafactory.launcher")
        if enabled:
            launcher.launch()
            run_exp.assert_called_once_with()
            spawn.assert_not_called()
        else:
            with pytest.raises(SystemExit, match="0"):
                launcher.launch()
            assert spawn.call_args.args[0][0] == "torchrun"
            run_exp.assert_not_called()
    dependency_check.assert_called_once_with(use_kt=enabled)
