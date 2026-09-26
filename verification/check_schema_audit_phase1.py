"""Bounded local QA adapter; only actual observations can produce PASS cases."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipts", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, PYTHONPATH=str(root / "src"))
    pytest = subprocess.run([
        sys.executable, "-m", "pytest", "-q", "tests/schema_audit",
        "tests/test_public_surface.py", "tests/test_core_redesign.py", "tests/test_core_architecture.py",
        "tests/test_docs_surface_alignment.py", "tests/test_docs_public_portal.py",
    ], cwd=root, env=env, capture_output=True, timeout=90)
    local = pytest.returncode == 0 and b"72 passed" in pytest.stdout
    schema = False
    sys.path.insert(0, str(root / "src"))
    try:
        import jsonschema

        from truthound.schema_audit import json_schema, to_dict

        spec = importlib.util.spec_from_file_location("synthetic_fixture", root / "tests/schema_audit/test_contract.py")
        fixture = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fixture)
        validator = jsonschema.Draft202012Validator(json_schema())
        validator.check_schema(json_schema())
        for factory in (fixture.standard, fixture.design, fixture.catalog):
            validator.validate(to_dict(factory()))
        schema = True
    except Exception:
        # The QA adapter fails closed on missing tooling or schema-validator errors.
        schema = False

    package = remote = False
    try:
        receipts = json.loads(args.receipts.read_text())
        wheel = next(args.artifacts.glob("truthound-*.whl"))
        sdist = next(args.artifacts.glob("truthound-*.tar.gz"))
        expected = {"artifacts/" + p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (wheel, sdist)}
        expected["tests/test_contract.py"] = hashlib.sha256((root / "tests/schema_audit/test_contract.py").read_bytes()).hexdigest()
        with zipfile.ZipFile(wheel) as archive:
            for source in (root / "src/truthound/schema_audit").rglob("*"):
                if source.is_file() and source.suffix in (".py", ".json"):
                    assert archive.read(source.relative_to(root / "src").as_posix()) == source.read_bytes()
        baseline = receipts["local"]["runs"]["wheel"]["semantic_digests"]
        for receipt in receipts.values():
            assert receipt["status"] == "PASS" and receipt["temporary_directory_removed"] is True
            assert all(receipt["file_digests"].get(key) == value for key, value in expected.items())
            assert set(receipt["runs"]) == {"wheel", "sdist"}
            for run in receipt["runs"].values():
                assert run["tests"] == 26 and all(run[k] == 0 for k in ("exit_code", "failures", "errors", "skipped"))
                assert run["semantic_digests"] == baseline
        package = receipts["local"]["python"].startswith("3.11.") and receipts["python312"]["python"].startswith("3.12.")
        remote = receipts["remote"]["architecture"] == "x86_64"
    except (ValueError, KeyError, AssertionError, OSError, StopIteration, zipfile.BadZipFile):
        pass
    results = {"model": local, "json": local and schema, "digest": local,
               "normalize": local, "package": local and package, "remote": remote}
    print(json.dumps({"schema_version": 1, "cases": [
        {"id": key, "status": "PASS" if passed else "FAIL"} for key, passed in results.items()
    ]}))
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    environment = Path(__file__).resolve().parents[1] / ".venv"
    if Path(sys.prefix) != environment:
        # Use the project's test dependencies when invoked by a system Python.
        os.execv(str(environment / "bin/python"), [str(environment / "bin/python"), __file__, *sys.argv[1:]])
    raise SystemExit(main())
