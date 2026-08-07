import importlib.util
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "kt_sft" / "attest_frozen_stack.py"
SPEC = importlib.util.spec_from_file_location("attest_frozen_stack", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
ATTEST = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ATTEST)


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def _init_repo(root: Path) -> str:
    root.mkdir()
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    _git(root, "config", "user.name", "KT test")
    _git(root, "config", "user.email", "kt-test@example.invalid")
    (root / "tracked.txt").write_text("frozen\n", encoding="utf-8")
    _git(root, "add", "tracked.txt")
    _git(root, "commit", "-qm", "frozen")
    return _git(root, "rev-parse", "HEAD")


def test_checkout_attestation_requires_exact_clean_head(tmp_path):
    root = tmp_path / "repo"
    head = _init_repo(root)

    assert ATTEST._attest_checkout("dependency", root, head)["status"] == "clean"

    with pytest.raises(RuntimeError, match="HEAD"):
        ATTEST._attest_checkout("dependency", root, "0" * 40)

    (root / "untracked.txt").write_text("dirty\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="dirty"):
        ATTEST._attest_checkout("dependency", root, head)


def test_attestation_disables_bytecode_writes():
    assert ATTEST.sys.dont_write_bytecode


def test_distribution_and_module_attestation_fail_closed(monkeypatch, tmp_path):
    monkeypatch.setattr(ATTEST.importlib.metadata, "version", lambda _: "1.2.3")
    assert ATTEST._attest_distribution("dependency", "1.2.3") == "1.2.3"
    with pytest.raises(RuntimeError, match="frozen"):
        ATTEST._attest_distribution("dependency", "1.2.4")

    module_root = tmp_path / "expected"
    module_root.mkdir()
    module_file = module_root / "module.py"
    module_file.write_text("", encoding="utf-8")
    monkeypatch.setattr(
        ATTEST.importlib,
        "import_module",
        lambda _: SimpleNamespace(__file__=str(module_file), __version__="test"),
    )
    assert ATTEST._attest_module("module", module_root)["version"] == "test"
    with pytest.raises(RuntimeError, match="expected under"):
        ATTEST._attest_module("module", tmp_path / "elsewhere")


def test_attestation_output_is_atomically_replaced(tmp_path):
    output = tmp_path / "evidence" / "stack.json"
    ATTEST._write_json_atomic(output, {"status": "PASS"})
    assert json.loads(output.read_text(encoding="utf-8")) == {"status": "PASS"}
    assert not list(output.parent.glob(f".{output.name}.*.tmp"))


def test_python_tree_digest_covers_relative_names_and_contents(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "a.py").write_text("value = 1\n", encoding="utf-8")
    (source / "nested").mkdir()
    (source / "nested" / "b.py").write_text("value = 2\n", encoding="utf-8")

    first_sha, files = ATTEST._python_tree_sha256(source)
    assert files == ("a.py", "nested/b.py")

    (source / "nested" / "b.py").write_text("value = 3\n", encoding="utf-8")
    second_sha, second_files = ATTEST._python_tree_sha256(source)
    assert second_files == files
    assert second_sha != first_sha


def test_frozen_dependency_constants():
    assert ATTEST.TRANSFORMERS_HEAD == "f7cf9750c33ca946f12203c1e8d15f8ae24b8628"
    assert ATTEST.ACCELERATE_HEAD == "300eebe175c44d41366a892f6be60f8e0cd43ef1"
    assert ATTEST.TRANSFORMERS_DIST_VERSION == "5.6.0.post1"
    assert ATTEST.ACCELERATE_DIST_VERSION == "1.14.0.post1"
