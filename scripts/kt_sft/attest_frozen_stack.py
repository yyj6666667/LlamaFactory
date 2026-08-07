#!/usr/bin/env python3
"""Fail closed unless a KT SFT run uses the reviewed source stack."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


# Source checkouts are evidence inputs. Importing them must not create pyc files
# after their clean-tree check has passed.
sys.dont_write_bytecode = True


TRANSFORMERS_HEAD = "f7cf9750c33ca946f12203c1e8d15f8ae24b8628"
ACCELERATE_HEAD = "300eebe175c44d41366a892f6be60f8e0cd43ef1"
TRANSFORMERS_DIST_VERSION = "5.6.0.post1"
ACCELERATE_DIST_VERSION = "1.14.0.post1"


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def _attest_checkout(name: str, root: Path, expected_head: str) -> dict[str, str]:
    root = root.resolve()
    actual_head = _git(root, "rev-parse", "HEAD")
    if actual_head != expected_head:
        raise RuntimeError(f"{name} HEAD {actual_head} != frozen {expected_head}")

    status = _git(root, "status", "--porcelain", "--untracked-files=all")
    if status:
        raise RuntimeError(f"{name} checkout is dirty:\n{status}")

    return {"root": str(root), "head": actual_head, "status": "clean"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _python_tree_sha256(root: Path) -> tuple[str, tuple[str, ...]]:
    files = tuple(sorted(str(path.relative_to(root)) for path in root.rglob("*.py") if path.is_file()))
    digest = hashlib.sha256()
    for relative_name in files:
        digest.update(relative_name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256((root / relative_name).read_bytes()).digest())
    return digest.hexdigest(), files


def _attest_distribution(name: str, expected_version: str) -> str:
    actual_version = importlib.metadata.version(name)
    if actual_version != expected_version:
        raise RuntimeError(f"{name} {actual_version} != frozen {expected_version}")
    return actual_version


def _attest_module(name: str, expected_root: Path) -> dict[str, str | None]:
    module = importlib.import_module(name)
    module_path = Path(module.__file__).resolve()
    expected_root = expected_root.resolve()
    if not module_path.is_relative_to(expected_root):
        raise RuntimeError(f"{name} imported from {module_path}, expected under {expected_root}")

    return {
        "file": str(module_path),
        "version": getattr(module, "__version__", None),
    }


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llama-factory-root", type=Path, required=True)
    parser.add_argument("--ktransformers-root", type=Path, required=True)
    parser.add_argument("--transformers-root", type=Path, required=True)
    parser.add_argument("--accelerate-root", type=Path, required=True)
    parser.add_argument("--kt-runtime-root", type=Path, required=True)
    parser.add_argument("--expected-llama-factory-head", required=True)
    parser.add_argument("--expected-ktransformers-head", required=True)
    parser.add_argument("--expected-kt-extension-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkouts = {
        "llama_factory": _attest_checkout(
            "LLaMA-Factory", args.llama_factory_root, args.expected_llama_factory_head
        ),
        "ktransformers": _attest_checkout(
            "KTransformers", args.ktransformers_root, args.expected_ktransformers_head
        ),
        "transformers": _attest_checkout("Transformers-KT", args.transformers_root, TRANSFORMERS_HEAD),
        "accelerate": _attest_checkout("Accelerate-KT", args.accelerate_root, ACCELERATE_HEAD),
    }

    distributions = {
        "transformers-kt": _attest_distribution("transformers-kt", TRANSFORMERS_DIST_VERSION),
        "accelerate-kt": _attest_distribution("accelerate-kt", ACCELERATE_DIST_VERSION),
    }
    modules = {
        "llamafactory": _attest_module("llamafactory", args.llama_factory_root / "src"),
        "transformers": _attest_module("transformers", args.transformers_root / "src"),
        "accelerate": _attest_module("accelerate", args.accelerate_root / "src"),
        "kt_kernel": _attest_module("kt_kernel", args.kt_runtime_root),
    }

    for name, record in checkouts.items():
        status = _git(Path(record["root"]), "status", "--porcelain", "--untracked-files=all")
        if status:
            raise RuntimeError(f"{name} checkout became dirty during attestation:\n{status}")

    source_python_sha, source_python_files = _python_tree_sha256(args.ktransformers_root / "kt-kernel" / "python")
    runtime_python_sha, runtime_python_files = _python_tree_sha256(args.kt_runtime_root / "kt_kernel")
    if source_python_files != runtime_python_files or source_python_sha != runtime_python_sha:
        raise RuntimeError("KT runtime Python package does not match the frozen KTransformers checkout")

    extensions = sorted(args.kt_runtime_root.resolve().glob("kt_kernel/kt_kernel_ext*.so"))
    if len(extensions) != 1:
        raise RuntimeError(f"expected exactly one KT extension, found: {extensions}")
    extension_sha256 = _sha256(extensions[0])
    if extension_sha256 != args.expected_kt_extension_sha256:
        raise RuntimeError(
            f"KT extension SHA256 {extension_sha256} != reviewed {args.expected_kt_extension_sha256}"
        )

    payload = {
        "status": "PASS",
        "created_at": datetime.now(UTC).isoformat(),
        "python": str(Path(sys.executable).resolve()),
        "checkouts": checkouts,
        "distributions": distributions,
        "modules": modules,
        "kt_extension": {
            "file": str(extensions[0]),
            "sha256": extension_sha256,
        },
        "kt_runtime_python": {
            "file_count": len(runtime_python_files),
            "sha256": runtime_python_sha,
        },
    }
    _write_json_atomic(args.output.resolve(), payload)
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
