"""Isolated release-wheel verifier contracts; no remote calls or release writes."""

from __future__ import annotations

import importlib.util
import json
import os
import py_compile
import re
import subprocess
import sys
import types
import zipfile
from pathlib import Path

import pytest

pytestmark = pytest.mark.contract


@pytest.fixture
def verifier():
    path = Path(__file__).resolve().parents[1] / "scripts/benchmarks/verify_release_package.py"
    spec = importlib.util.spec_from_file_location("release_package_verifier_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("value", ["3.1.15; echo unsafe", "../3.1.15", "3.1", "3.1.15.dev0"])
def test_release_version_is_not_a_shell_or_ref_input(verifier, value):
    with pytest.raises(verifier.VerificationError, match="RELEASE_VERSION_INVALID"):
        verifier.validate_identity(value, "v" + value, "a" * 40, "b" * 64, "c" * 40)


@pytest.mark.parametrize("prior", [None, "original-fixture-root"])
def test_failed_execution_restores_caller_environment_and_arguments(verifier, monkeypatch, prior):
    if prior is None:
        monkeypatch.delenv("TRUTHOUND_BENCHMARK_WORKLOAD_ROOT", raising=False)
    else:
        monkeypatch.setenv("TRUTHOUND_BENCHMARK_WORKLOAD_ROOT", prior)
    arguments = sys.argv

    def fail(*args):
        os.environ["TRUTHOUND_BENCHMARK_WORKLOAD_ROOT"] = "temporary-fixture-root"
        sys.argv = ["temporary-worker"]
        raise verifier.VerificationError("SYNTHETIC_FAILURE")

    monkeypatch.setattr(verifier, "_execute", fail)
    with pytest.raises(verifier.VerificationError, match="SYNTHETIC_FAILURE"):
        verifier.execute(None, "run", [])
    assert os.environ.get("TRUTHOUND_BENCHMARK_WORKLOAD_ROOT") == prior
    assert sys.argv is arguments


def test_release_tag_must_match_version(verifier):
    with pytest.raises(verifier.VerificationError, match="RELEASE_TAG_INVALID"):
        verifier.validate_identity("3.1.15", "v3.1.14", "a" * 40, "b" * 64, "c" * 40)


@pytest.mark.parametrize("field", ["release_commit", "harness_commit", "wheel_sha256"])
def test_identity_requires_full_lowercase_digests(verifier, field):
    values = dict(
        version="3.1.15",
        tag="v3.1.15",
        release_commit="a" * 40,
        wheel_sha256="b" * 64,
        harness_commit="c" * 40,
    )
    values[field] = "latest"
    with pytest.raises(verifier.VerificationError):
        verifier.validate_identity(**values)


@pytest.mark.parametrize(
    "url",
    [
        "http://files.pythonhosted.org/a.whl",
        "https://evil.example/a.whl",
        "https://user@files.pythonhosted.org/a.whl",
        "https://files.pythonhosted.org:444/a.whl",
    ],
)
def test_download_origin_is_fixed(verifier, url):
    with pytest.raises(verifier.VerificationError, match="DOWNLOAD_ORIGIN_INVALID"):
        verifier.validate_download_url(url, "files.pythonhosted.org")


def test_harness_namespace_does_not_replace_product_namespace(verifier, tmp_path):
    root = tmp_path / "src/truthound/benchmark"
    root.mkdir(parents=True)
    (root / "parity.py").write_text("SENTINEL = 'corrected harness'\n")
    package = verifier.load_harness(tmp_path)
    assert package.__name__ == "_truthound_release_harness.parity"
    assert package.SENTINEL == "corrected harness"
    assert str(tmp_path / "src") not in sys.path
    assert package.__name__ != "truthound.benchmark.parity"


def test_untracked_harness_module_is_rejected_before_execution(verifier, tmp_path):
    root = tmp_path / "src/truthound/benchmark"
    root.mkdir(parents=True)
    (root / "parity.py").write_text("SENTINEL = 'corrected harness'\n")
    (root / "untracked.py").write_text("raise RuntimeError('must not execute')\n")
    verifier.load_harness(tmp_path)
    with pytest.raises(verifier.VerificationError, match="HARNESS_MODULE_NOT_ALLOWED"):
        importlib.import_module("_truthound_release_harness.untracked")


@pytest.mark.parametrize(
    "mode",
    [py_compile.PycInvalidationMode.TIMESTAMP, py_compile.PycInvalidationMode.UNCHECKED_HASH],
)
def test_harness_executes_source_not_valid_looking_poisoned_bytecode(verifier, tmp_path, mode):
    root = tmp_path / "src/truthound/benchmark"
    root.mkdir(parents=True)
    source = root / "parity.py"
    source.write_text("VALUE = 'evil'\n")
    stamp = source.stat()
    py_compile.compile(str(source), doraise=True, invalidation_mode=mode)
    source.write_text("VALUE = 'safe'\n")
    os.utime(source, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    # The fixture is accepted by Python's normal cached source loader despite
    # no longer matching the visible source bytes.
    loader = importlib.machinery.SourceFileLoader("cache_fixture_control", str(source))
    control = {}
    exec(loader.get_code("cache_fixture_control"), control)
    assert control["VALUE"] == "evil"
    module = verifier.load_harness(tmp_path)
    assert module.VALUE == "safe"
    assert Path(module.__file__) == source
    verifier.verify_harness_origins(tmp_path)


def test_harness_rejects_source_changed_since_provenance_was_verified(verifier, tmp_path):
    root = tmp_path / "src/truthound/benchmark"
    root.mkdir(parents=True)
    receipt = {}
    for name in verifier.HARNESS_FILES:
        source = root / name
        source.write_text("VALUE = 'safe'\n")
        receipt[f"src/truthound/benchmark/{name}"] = verifier.sha256(source.read_bytes())
    (root / "parity.py").write_text("VALUE = 'evil'\n")
    with pytest.raises(verifier.VerificationError, match="HARNESS_SOURCE_DIGEST_MISMATCH"):
        verifier.load_harness(tmp_path, expected_files=receipt)


def test_dependency_artifact_contains_only_bounded_names_and_versions(
    verifier, tmp_path, monkeypatch
):
    distributions = [
        types.SimpleNamespace(
            metadata={"Name": "safe_package", "Secret": "ignored"},
            version="1.2.3",
            origin="https://credential.invalid",
        )
    ]
    monkeypatch.setattr(verifier.importlib.metadata, "distributions", lambda: distributions)
    output = tmp_path / "installed-dependencies.json"
    verifier.write_dependencies(output)
    assert json.loads(output.read_text()) == [{"name": "safe-package", "version": "1.2.3"}]


def test_dependency_metadata_url_cannot_escape_into_artifact(verifier, tmp_path, monkeypatch):
    distributions = [
        types.SimpleNamespace(metadata={"Name": "safe"}, version="https://credential.invalid")
    ]
    monkeypatch.setattr(verifier.importlib.metadata, "distributions", lambda: distributions)
    output = tmp_path / "installed-dependencies.json"
    with pytest.raises(verifier.VerificationError, match="DEPENDENCY_METADATA_INVALID"):
        verifier.write_dependencies(output)
    assert not output.exists()


def test_worker_environment_cannot_import_checkout_or_user_site(verifier, monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "/untrusted/checkout/src")
    monkeypatch.setenv("PYTHONHOME", "/untrusted/python")
    monkeypatch.setenv("TRUTHOUND_BENCHMARK_WORKLOAD_ROOT", "/untrusted/workloads")
    env = verifier.worker_environment(Path("/released/benchmarks/workloads"))
    assert "PYTHONPATH" not in env
    assert "PYTHONHOME" not in env
    assert env["PYTHONNOUSERSITE"] == "1"
    assert env["TRUTHOUND_BENCHMARK_WORKLOAD_ROOT"] == "/released/benchmarks/workloads"


def test_workload_manifest_is_full_content_addressed(verifier, tmp_path):
    root = tmp_path / "benchmarks/workloads"
    (root / "data").mkdir(parents=True)
    (root / "data/input.csv").write_text("x\n1\n")
    for name in verifier.RELEASE_WORKLOADS:
        (root / f"{name}.json").write_text(
            json.dumps({"id": name, "suites": ["release-ga"], "dataset": "data/input.csv"})
        )
    first = verifier.workload_manifest(tmp_path)
    assert len(first["sha256"]) == 64
    (root / "local-null.json").write_text(
        json.dumps(
            {
                "id": "local-null",
                "suites": ["release-ga"],
                "dataset": "data/input.csv",
                "expected": False,
            }
        )
    )
    assert verifier.workload_manifest(tmp_path)["sha256"] != first["sha256"]


def test_workload_escape_is_rejected(verifier, tmp_path):
    root = tmp_path / "benchmarks/workloads"
    root.mkdir(parents=True)
    for name in verifier.RELEASE_WORKLOADS:
        (root / f"{name}.json").write_text(
            json.dumps({"id": name, "suites": ["release-ga"], "dataset": "../../outside.csv"})
        )
    with pytest.raises(verifier.VerificationError, match="WORKLOAD_PATH_INVALID"):
        verifier.workload_manifest(tmp_path)


def test_workflow_preserves_source_lane_and_has_explicit_wheel_lane():
    path = Path(__file__).resolve().parents[1] / ".github/workflows/benchmarks-release.yml"
    source = path.read_text()
    assert "verify_release_package.py prepare" in source
    assert "verify_release_package.py run" in source
    assert '".[dev,benchmarks]"' in source
    assert "persist-credentials: false" in source
    assert "VERIFY_RELEASE_COMMIT: ${{ inputs.release_commit }}" in source
    assert "VERIFY_WHEEL_SHA256: ${{ inputs.wheel_sha256 }}" in source
    assert source.index("verify_release_package.py validate") < source.index(
        "ref: ${{ inputs.release_commit }}"
    )
    assert "uv pip freeze" not in source
    assert "--clear" not in source
    assert "truthound-release-${GITHUB_RUN_ID}-${GITHUB_RUN_ATTEMPT}" in source
    assert "printf 'RELEASE_VENV=%s/venv\\n'" in source


def _release_workflow_job():
    import yaml

    path = Path(__file__).resolve().parents[1] / ".github/workflows/benchmarks-release.yml"
    return yaml.safe_load(path.read_text())["jobs"]["release-parity"]


def test_release_job_env_only_uses_available_github_contexts():
    allowed = {"github", "needs", "strategy", "matrix", "vars", "secrets", "inputs"}
    job = _release_workflow_job()
    for value in job["env"].values():
        contexts = re.findall(r"\$\{\{\s*([A-Za-z_][A-Za-z_0-9]*)[.\[]", str(value))
        assert set(contexts) <= allowed
    assert "RELEASE_VENV" not in job["env"]
    assert "VERIFY_WHEEL_PATH" not in job["env"]


@pytest.mark.parametrize("version", ["", "3.1.15"])
def test_release_runtime_paths_initialize_before_venv_creation(tmp_path, version):
    job = _release_workflow_job()
    steps = job["steps"]
    index = next(
        i for i, step in enumerate(steps) if step.get("name") == "Initialize release runtime paths"
    )
    create_index = next(
        i
        for i, step in enumerate(steps)
        if step.get("name") == "Create release benchmark virtual environment"
    )
    assert index < create_index
    step = steps[index]
    assert step["shell"] == "bash"
    environment_file = tmp_path / "environment"
    runner_temp = tmp_path / "runner temp"
    result = subprocess.run(
        ["bash", "-c", step["run"]],
        env={
            **os.environ,
            "RUNNER_TEMP": str(runner_temp),
            "GITHUB_RUN_ID": "123",
            "GITHUB_RUN_ATTEMPT": "2",
            "VERIFY_VERSION": version,
            "GITHUB_ENV": str(environment_file),
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    workdir = runner_temp / "truthound-release-123-2"
    assert environment_file.read_text().splitlines() == [
        f"RELEASE_VENV={workdir}/venv",
        f"VERIFY_WHEEL_PATH={workdir}/truthound-{version}-py3-none-any.whl",
    ]


@pytest.mark.parametrize("version", ["3.1.15\nINJECTED_ENV=unexpected", "3.1.15; echo unsafe"])
def test_release_runtime_path_inputs_cannot_inject_environment(tmp_path, version):
    steps = _release_workflow_job()["steps"]
    step = next(step for step in steps if step.get("name") == "Initialize release runtime paths")
    environment_file = tmp_path / "environment"
    result = subprocess.run(
        ["bash", "-c", step["run"]],
        env={
            **os.environ,
            "RUNNER_TEMP": str(tmp_path),
            "GITHUB_RUN_ID": "123",
            "GITHUB_RUN_ATTEMPT": "2",
            "VERIFY_VERSION": version,
            "GITHUB_ENV": str(environment_file),
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "RELEASE_VERSION_INVALID" in result.stderr
    assert not environment_file.exists()


def _installed_fixture(verifier, tmp_path, monkeypatch):
    prefix = tmp_path / "venv"
    site = prefix / "site-packages"
    package = site / "truthound"
    package.mkdir(parents=True)
    files = {
        "truthound/__init__.py": b"VERSION = '3.1.15'\n",
        "truthound/api.py": b"def check(): pass\n",
    }
    for name, data in files.items():
        (site / name).write_bytes(data)
        release_source = tmp_path / "release/src" / name
        release_source.parent.mkdir(parents=True, exist_ok=True)
        release_source.write_bytes(data)
    wheel = tmp_path / "truthound-3.1.15-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        for name, data in files.items():
            archive.writestr(name, data)
        archive.writestr(
            "truthound-3.1.15.dist-info/METADATA", "Name: truthound\nVersion: 3.1.15\n"
        )
    distribution = types.SimpleNamespace(version="3.1.15", locate_file=lambda name: site / name)
    th = types.SimpleNamespace(__file__=str(package / "__init__.py"), check=lambda: None)
    fake_sys = types.SimpleNamespace(
        flags=types.SimpleNamespace(isolated=1),
        prefix=str(prefix),
        base_prefix=str(tmp_path / "base"),
        modules={"truthound": th},
    )
    monkeypatch.setattr(verifier, "sys", fake_sys)
    monkeypatch.setattr(verifier.importlib.metadata, "distribution", lambda name: distribution)
    monkeypatch.setattr(verifier.importlib, "import_module", lambda name: th)
    monkeypatch.setattr(verifier.inspect, "getsourcefile", lambda value: str(package / "api.py"))
    config = verifier.Configuration(
        "3.1.15",
        "v3.1.15",
        "a" * 40,
        verifier.sha256(wheel.read_bytes()),
        "b" * 40,
        tmp_path / "harness",
        tmp_path / "release",
        wheel,
        tmp_path / "result.json",
    )
    return config, th, files, site


def test_exact_installed_wheel_bytes_and_origin_pass(verifier, tmp_path, monkeypatch):
    config, th, files, _ = _installed_fixture(verifier, tmp_path, monkeypatch)
    actual, contents = verifier.verify_installed_wheel(config)
    assert actual is th
    assert contents == files


def test_same_version_checkout_cannot_impersonate_wheel(verifier, tmp_path, monkeypatch):
    config, th, _, _ = _installed_fixture(verifier, tmp_path, monkeypatch)
    th.__file__ = str(tmp_path / "checkout/src/truthound/__init__.py")
    with pytest.raises(verifier.VerificationError, match="PRODUCT_MODULE_OUTSIDE_WHEEL"):
        verifier.verify_installed_wheel(config)


def test_changed_installed_product_file_is_rejected(verifier, tmp_path, monkeypatch):
    config, _, _, site = _installed_fixture(verifier, tmp_path, monkeypatch)
    (site / "truthound/api.py").write_text("def check(): return True\n")
    with pytest.raises(verifier.VerificationError, match="INSTALLED_FILE_MISMATCH"):
        verifier.verify_installed_wheel(config)


def test_released_wheel_cannot_be_attributed_to_different_tag_source(
    verifier, tmp_path, monkeypatch
):
    config, _, _, _ = _installed_fixture(verifier, tmp_path, monkeypatch)
    (config.release_root / "src/truthound/api.py").write_text("def check(): return True\n")
    with pytest.raises(verifier.VerificationError, match="WHEEL_RELEASE_SOURCE_MISMATCH"):
        verifier.verify_installed_wheel(config)


def test_extra_imported_product_module_not_in_wheel_is_rejected(verifier, tmp_path, monkeypatch):
    config, _, _, site = _installed_fixture(verifier, tmp_path, monkeypatch)
    extra = site / "truthound/fake.py"
    extra.write_text("FAKE = True\n")
    verifier.sys.modules["truthound.fake"] = types.SimpleNamespace(__file__=str(extra))
    with pytest.raises(verifier.VerificationError, match="PRODUCT_MODULE_NOT_IN_WHEEL"):
        verifier.verify_installed_wheel(config)


def test_nonisolated_interpreter_is_rejected_before_import(verifier, tmp_path, monkeypatch):
    config, _, _, _ = _installed_fixture(verifier, tmp_path, monkeypatch)
    verifier.sys.flags.isolated = 0
    with pytest.raises(verifier.VerificationError, match="ISOLATED_VENV_REQUIRED"):
        verifier.verify_installed_wheel(config)


def test_wheel_digest_mismatch_is_rejected(verifier, tmp_path, monkeypatch):
    config, _, _, _ = _installed_fixture(verifier, tmp_path, monkeypatch)
    config.wheel_path.write_bytes(b"wrong wheel")
    with pytest.raises(verifier.VerificationError, match="WHEEL_DIGEST_MISMATCH"):
        verifier.verify_installed_wheel(config)


def test_worker_prefix_uses_isolation_and_exact_provenance(verifier, tmp_path):
    config = verifier.Configuration(
        "3.1.15",
        "v3.1.15",
        "a" * 40,
        "b" * 64,
        "c" * 40,
        tmp_path / "harness",
        tmp_path / "release",
        tmp_path / "truthound-3.1.15-py3-none-any.whl",
        tmp_path / "result.json",
    )
    command = verifier.worker_prefix(config)
    assert command[1] == "-I"
    assert "-m" not in command
    assert command[command.index("--release-commit") + 1] == config.release_commit
    assert command[command.index("--wheel-sha256") + 1] == config.wheel_sha256
    assert command[command.index("--harness-commit") + 1] == config.harness_commit


def test_prepare_digest_mismatch_does_not_write_wheel(verifier, tmp_path, monkeypatch):
    config = verifier.Configuration(
        "3.1.15",
        "v3.1.15",
        "a" * 40,
        "b" * 64,
        "c" * 40,
        tmp_path / "harness",
        tmp_path / "release",
        tmp_path / "truthound-3.1.15-py3-none-any.whl",
        tmp_path / "result.json",
    )
    monkeypatch.setattr(verifier, "checkout_provenance", lambda config: {})
    metadata = {
        "info": {"version": "3.1.15"},
        "urls": [
            {
                "filename": config.wheel_path.name,
                "packagetype": "bdist_wheel",
                "digests": {"sha256": "b" * 64},
                "size": 3,
                "url": "https://files.pythonhosted.org/fixture.whl",
            }
        ],
    }
    monkeypatch.setattr(
        verifier,
        "download",
        lambda url, host, limit: json.dumps(metadata).encode() if host == "pypi.org" else b"bad",
    )
    with pytest.raises(verifier.VerificationError, match="WHEEL_DIGEST_MISMATCH"):
        verifier.prepare(config)
    assert not config.wheel_path.exists()


@pytest.mark.parametrize(
    "missing_origin, failed_gate, expected_exit",
    [(False, False, 0), (True, False, 1), (False, True, 1)],
)
def test_canonical_failure_and_missing_worker_provenance_remain_blocking(
    verifier, tmp_path, monkeypatch, missing_origin, failed_gate, expected_exit
):
    from datetime import datetime

    from truthound.benchmark.parity import (
        BenchmarkMethodology,
        FrameworkObservation,
        ParityAssertion,
        ParityResult,
    )

    provenance = {"fixture": "unchanged wheel", "harness_files": {}}
    provenance_id = verifier.digest_json(provenance)
    observations = tuple(
        FrameworkObservation(
            framework,
            "3.1.15" if framework == "truthound" else "1.15.0",
            name,
            "fixture",
            "local",
            "exact",
            0.1,
            0.1,
            100,
            True,
            3,
            3,
            metadata={}
            if missing_origin
            else {
                "release_package_provenance": provenance_id,
                "release_package_measurement_integrity": True,
            },
        )
        for name in verifier.RELEASE_WORKLOADS
        for framework in ("truthound", "gx")
    )
    result = ParityResult(
        "release-ga",
        observations,
        (ParityAssertion("fixture:performance", not failed_gate, "fixture"),),
        BenchmarkMethodology(),
        {},
        str(tmp_path),
        datetime.now(),
        datetime.now(),
        {"release_claim_ready": not failed_gate},
    )

    class FakeRunner:
        def run_suite(self, suite, **kwargs):
            assert suite == "release-ga"
            assert kwargs["frameworks"] == "both"
            assert "backend" not in kwargs
            return result

    written = []
    fake_parity = types.SimpleNamespace(
        ParityRunner=FakeRunner,
        ParityAssertion=ParityAssertion,
        classify_release_blockers=lambda result: {},
        write_parity_artifacts=lambda result, output: written.append(result),
        write_environment_manifest=lambda *args: None,
        write_release_summary=lambda *args, **kwargs: None,
    )
    config = verifier.Configuration(
        "3.1.15",
        "v3.1.15",
        "a" * 40,
        "b" * 64,
        "c" * 40,
        tmp_path / "harness",
        tmp_path / "release",
        tmp_path / "truthound-3.1.15-py3-none-any.whl",
        tmp_path / "result.json",
    )
    th = types.SimpleNamespace(check=lambda: None)
    monkeypatch.setattr(verifier, "checkout_provenance", lambda config: provenance)
    monkeypatch.setattr(verifier, "verify_installed_wheel", lambda config: (th, {}))
    monkeypatch.setattr(verifier, "verify_loaded_origins", lambda *args: None)
    monkeypatch.setattr(verifier, "verify_harness_origins", lambda *args: None)
    monkeypatch.setattr(verifier, "load_harness", lambda root, **kwargs: fake_parity)
    monkeypatch.setattr(verifier.importlib.metadata, "distribution", lambda name: None)
    prior_workloads = os.environ.get("TRUTHOUND_BENCHMARK_WORKLOAD_ROOT")
    assert verifier.execute(config, "run", []) == expected_exit
    assert os.environ.get("TRUTHOUND_BENCHMARK_WORKLOAD_ROOT") == prior_workloads
    assert len(written) == 1
    assert written[0].assertions[0].passed is (not failed_gate)
    assert written[0].metadata["release_claim_ready"] is (expected_exit == 0)
