"""Verify an immutable released wheel with a separately identified benchmark harness.

Run with ``python -I`` in a fresh virtual environment. This script never installs
checkout product code, changes a tag, or publishes a package or a release.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.abc
import importlib.metadata
import importlib.util
import inspect
import json
import os
import re
import subprocess
import sys
import types
import urllib.parse
import urllib.request
import zipfile
from dataclasses import dataclass, replace
from email.parser import BytesParser
from pathlib import Path

NAMESPACE = "_truthound_release_harness"
RELEASE_WORKLOADS = frozenset(
    {
        "local-null",
        "local-unique",
        "local-range",
        "local-schema",
        "local-mixed-core-suite",
        "sqlite-null",
        "sqlite-unique",
        "sqlite-range",
    }
)
HARNESS_FILES = ("base.py", "workloads.py", "parity.py", "_parity_worker.py")


class VerificationError(RuntimeError):
    """A fixed, safe verifier failure code."""


def require(condition: bool, code: str) -> None:
    if not condition:
        raise VerificationError(code)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def digest_json(data: object) -> str:
    return sha256(json.dumps(data, sort_keys=True, separators=(",", ":")).encode())


def validate_identity(
    version: str, tag: str, release_commit: str, wheel_sha256: str, harness_commit: str
) -> None:
    require(re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", version) is not None, "RELEASE_VERSION_INVALID")
    require(tag == f"v{version}", "RELEASE_TAG_INVALID")
    for value in (release_commit, harness_commit):
        require(re.fullmatch(r"[0-9a-f]{40}", value) is not None, "COMMIT_INVALID")
    require(re.fullmatch(r"[0-9a-f]{64}", wheel_sha256) is not None, "WHEEL_DIGEST_INVALID")


@dataclass(frozen=True)
class Configuration:
    version: str
    tag: str
    release_commit: str
    wheel_sha256: str
    harness_commit: str
    harness_root: Path
    release_root: Path
    wheel_path: Path
    output: Path

    def validate(self) -> None:
        validate_identity(
            self.version, self.tag, self.release_commit, self.wheel_sha256, self.harness_commit
        )
        require(
            self.wheel_path.name == f"truthound-{self.version}-py3-none-any.whl",
            "WHEEL_NAME_INVALID",
        )
        require(self.harness_root != self.release_root, "CHECKOUTS_NOT_SEPARATE")


def git(root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), *args], capture_output=True, text=True, timeout=30, check=False
    )
    require(completed.returncode == 0, "CHECKOUT_VERIFICATION_FAILED")
    return completed.stdout.strip()


def checkout_provenance(config: Configuration) -> dict[str, object]:
    config.validate()
    require(
        git(config.harness_root, "rev-parse", "HEAD") == config.harness_commit,
        "HARNESS_COMMIT_MISMATCH",
    )
    require(
        git(config.release_root, "rev-parse", "HEAD") == config.release_commit,
        "RELEASE_COMMIT_MISMATCH",
    )
    require(
        git(config.release_root, "rev-parse", f"refs/tags/{config.tag}^{{commit}}")
        == config.release_commit,
        "RELEASE_TAG_MISMATCH",
    )
    for root in (config.harness_root, config.release_root):
        git(root, "diff", "--exit-code", "HEAD", "--")
    harness = {}
    for name in HARNESS_FILES:
        relative = f"src/truthound/benchmark/{name}"
        path = config.harness_root / relative
        require(path.is_file() and not path.is_symlink(), "HARNESS_FILE_INVALID")
        git(config.harness_root, "ls-files", "--error-unmatch", relative)
        harness[relative] = sha256(path.read_bytes())
    script = Path(__file__).resolve()
    require(
        script == config.harness_root / "scripts/benchmarks/verify_release_package.py",
        "BOOTSTRAP_ORIGIN_INVALID",
    )
    git(
        config.harness_root,
        "ls-files",
        "--error-unmatch",
        "scripts/benchmarks/verify_release_package.py",
    )
    harness["scripts/benchmarks/verify_release_package.py"] = sha256(script.read_bytes())
    return {
        "release_version": config.version,
        "release_tag": config.tag,
        "release_commit": config.release_commit,
        "wheel_sha256": config.wheel_sha256,
        "harness_commit": config.harness_commit,
        "harness_files": harness,
        "harness_sha256": digest_json(harness),
        "workloads": workload_manifest(config.release_root),
        "measurement_target": "immutable-installed-wheel",
        "harness_namespace": NAMESPACE,
        "product_source_policy": "wheel-python-files-byte-match-release-commit",
    }


def workload_manifest(release_root: Path) -> dict[str, object]:
    root = (release_root / "benchmarks/workloads").resolve()
    selected = {}
    for path in sorted(root.glob("*.json")):
        require(not path.is_symlink(), "WORKLOAD_PATH_INVALID")
        payload = json.loads(path.read_bytes())
        if "release-ga" not in payload.get("suites", []):
            continue
        name = payload.get("id")
        require(
            name in RELEASE_WORKLOADS and name not in selected and path.stem == name,
            "WORKLOAD_CATALOG_INVALID",
        )
        dataset = (root / payload["dataset"]).resolve()
        require(dataset.is_relative_to(root) and dataset.is_file(), "WORKLOAD_PATH_INVALID")
        selected[name] = {
            "manifest_sha256": sha256(path.read_bytes()),
            "dataset_sha256": sha256(dataset.read_bytes()),
        }
    require(set(selected) == RELEASE_WORKLOADS, "WORKLOAD_CATALOG_INVALID")
    return {"count": len(selected), "sha256": digest_json(selected), "files": selected}


def validate_download_url(url: str, host: str) -> None:
    parsed = urllib.parse.urlsplit(url)
    require(
        parsed.scheme == "https"
        and parsed.hostname == host
        and parsed.port in (None, 443)
        and parsed.username is None
        and parsed.password is None
        and not parsed.fragment,
        "DOWNLOAD_ORIGIN_INVALID",
    )


def download(url: str, host: str, limit: int) -> bytes:
    class FixedOriginRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, request, fp, code, message, headers, newurl):
            validate_download_url(newurl, host)
            return super().redirect_request(request, fp, code, message, headers, newurl)

    validate_download_url(url, host)
    opener = urllib.request.build_opener(FixedOriginRedirect())
    with opener.open(url, timeout=45) as response:
        validate_download_url(response.geturl(), host)
        data = response.read(limit + 1)
    require(len(data) <= limit, "DOWNLOAD_SIZE_EXCEEDED")
    return data


def prepare(config: Configuration) -> None:
    checkout_provenance(config)
    payload = json.loads(
        download(f"https://pypi.org/pypi/truthound/{config.version}/json", "pypi.org", 2_000_000)
    )
    require(payload["info"]["version"] == config.version, "PYPI_VERSION_MISMATCH")
    matches = [item for item in payload["urls"] if item.get("filename") == config.wheel_path.name]
    require(len(matches) == 1, "PYPI_ARTIFACT_CARDINALITY")
    artifact = matches[0]
    require(
        not artifact.get("yanked")
        and artifact.get("packagetype") == "bdist_wheel"
        and artifact["digests"]["sha256"] == config.wheel_sha256,
        "PYPI_ARTIFACT_MISMATCH",
    )
    data = download(artifact["url"], "files.pythonhosted.org", 50_000_000)
    require(
        len(data) == artifact["size"] and sha256(data) == config.wheel_sha256,
        "WHEEL_DIGEST_MISMATCH",
    )
    config.wheel_path.parent.mkdir(parents=True, exist_ok=True)
    if config.wheel_path.exists():
        require(
            not config.wheel_path.is_symlink() and config.wheel_path.read_bytes() == data,
            "WHEEL_ALREADY_EXISTS_MISMATCH",
        )
    else:
        with config.wheel_path.open("xb") as destination:
            destination.write(data)
    print(
        json.dumps(
            {
                "prepared": True,
                "release_version": config.version,
                "wheel_sha256": config.wheel_sha256,
            }
        )
    )


def verify_installed_wheel(config: Configuration) -> tuple[object, dict[str, bytes]]:
    require(sys.flags.isolated == 1 and sys.prefix != sys.base_prefix, "ISOLATED_VENV_REQUIRED")
    require(
        not config.wheel_path.is_symlink()
        and sha256(config.wheel_path.read_bytes()) == config.wheel_sha256,
        "WHEEL_DIGEST_MISMATCH",
    )
    with zipfile.ZipFile(config.wheel_path) as wheel:
        names = wheel.namelist()
        require(len(names) == len(set(names)), "WHEEL_DUPLICATE_MEMBER")
        metadata_names = [name for name in names if name.endswith(".dist-info/METADATA")]
        require(len(metadata_names) == 1, "WHEEL_METADATA_INVALID")
        metadata = BytesParser().parsebytes(wheel.read(metadata_names[0]))
        require(
            metadata["Name"] == "truthound" and metadata["Version"] == config.version,
            "WHEEL_METADATA_INVALID",
        )
        files = {
            name: wheel.read(name)
            for name in names
            if name.startswith("truthound/") and not name.endswith("/")
        }
    require(
        "truthound/__init__.py" in files and "truthound/api.py" in files,
        "WHEEL_PRODUCT_FILES_MISSING",
    )
    require(all(".." not in Path(name).parts for name in files), "WHEEL_PATH_INVALID")
    for name, data in files.items():
        if name.endswith(".py"):
            source = config.release_root / "src" / name
            require(
                source.is_file() and not source.is_symlink() and source.read_bytes() == data,
                "WHEEL_RELEASE_SOURCE_MISMATCH",
            )
    distribution = importlib.metadata.distribution("truthound")
    require(distribution.version == config.version, "INSTALLED_VERSION_MISMATCH")
    package_root = Path(distribution.locate_file("truthound")).resolve()
    require(package_root.is_relative_to(Path(sys.prefix).resolve()), "INSTALLED_ORIGIN_INVALID")
    for name, data in files.items():
        path = Path(distribution.locate_file(name))
        require(
            not path.is_symlink()
            and path.resolve().is_relative_to(package_root)
            and path.read_bytes() == data,
            "INSTALLED_FILE_MISMATCH",
        )
    th = importlib.import_module("truthound")
    verify_loaded_origins(th, distribution, files)
    return th, files


def verify_loaded_origins(th: object, distribution: object, files: dict[str, bytes]) -> None:
    root = Path(distribution.locate_file("")).resolve()
    for name, module in tuple(sys.modules.items()):
        if name != "truthound" and not name.startswith("truthound."):
            continue
        origin = getattr(module, "__file__", None)
        require(origin is not None, "PRODUCT_MODULE_ORIGIN_MISSING")
        path = Path(origin).resolve()
        require(path.is_relative_to(root), "PRODUCT_MODULE_OUTSIDE_WHEEL")
        relative = path.relative_to(root).as_posix()
        require(
            relative in files and path.read_bytes() == files[relative],
            "PRODUCT_MODULE_NOT_IN_WHEEL",
        )
    require(
        Path(inspect.getsourcefile(th.check)).resolve()
        == Path(distribution.locate_file("truthound/api.py")).resolve(),
        "PUBLIC_CHECK_ORIGIN_INVALID",
    )


def load_harness(harness_root: Path, *, expected_files: dict[str, str] | None = None):
    sources: dict[str, tuple[Path, bytes]] = {}
    for name in HARNESS_FILES:
        relative = f"src/truthound/benchmark/{name}"
        source = harness_root / relative
        if expected_files is None and not source.exists():
            continue
        require(source.is_file() and not source.is_symlink(), "HARNESS_FILE_INVALID")
        data = source.read_bytes()
        if expected_files is not None:
            require(expected_files.get(relative) == sha256(data), "HARNESS_SOURCE_DIGEST_MISMATCH")
        sources[name] = (source, data)

    class SourceOnlyLoader(importlib.abc.Loader):
        def __init__(self, source: Path, data: bytes):
            self.source = source
            self.data = data

        def create_module(self, spec):
            return None

        def exec_module(self, module):
            # Compile the bytes checked against the harness receipt, not a
            # timestamp-valid or unchecked-hash __pycache__ entry.
            module.__file__ = str(self.source)
            exec(compile(self.data, str(self.source), "exec", dont_inherit=True), module.__dict__)

    class HarnessOnlyFinder(importlib.abc.MetaPathFinder):
        truthound_release_harness_guard = True

        def find_spec(self, fullname, path=None, target=None):
            if not fullname.startswith(NAMESPACE + "."):
                return None
            name = fullname.removeprefix(NAMESPACE + ".") + ".py"
            require(name in HARNESS_FILES, "HARNESS_MODULE_NOT_ALLOWED")
            require(name in sources, "HARNESS_FILE_INVALID")
            source, data = sources[name]
            return importlib.util.spec_from_file_location(
                fullname, source, loader=SourceOnlyLoader(source, data)
            )

    sys.meta_path[:] = [
        finder
        for finder in sys.meta_path
        if not getattr(finder, "truthound_release_harness_guard", False)
    ]
    sys.meta_path.insert(0, HarnessOnlyFinder())
    for name in tuple(sys.modules):
        if name == NAMESPACE or name.startswith(NAMESPACE + "."):
            del sys.modules[name]
    package = types.ModuleType(NAMESPACE)
    package.__path__ = [str(harness_root / "src/truthound/benchmark")]
    sys.modules[NAMESPACE] = package
    return importlib.import_module(NAMESPACE + ".parity")


def verify_harness_origins(harness_root: Path) -> None:
    root = harness_root / "src/truthound/benchmark"
    for name, module in tuple(sys.modules.items()):
        if not name.startswith(NAMESPACE + "."):
            continue
        filename = name.removeprefix(NAMESPACE + ".") + ".py"
        require(
            filename in HARNESS_FILES
            and getattr(module, "__file__", None) is not None
            and Path(module.__file__).resolve() == root / filename,
            "HARNESS_MODULE_NOT_ALLOWED",
        )


def write_dependencies(output: Path) -> None:
    records = []
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name", "")
        version = distribution.version
        require(
            re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,199}", name) is not None
            and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.!+_-]{0,127}", version) is not None,
            "DEPENDENCY_METADATA_INVALID",
        )
        records.append({"name": re.sub(r"[-_.]+", "-", name).lower(), "version": version})
        require(len(records) <= 2000, "DEPENDENCY_COUNT_EXCEEDED")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(sorted(records, key=lambda item: (item["name"], item["version"])), indent=2)
    )


def worker_environment(workloads: Path) -> dict[str, str]:
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    env["PYTHONNOUSERSITE"] = "1"
    env["TRUTHOUND_BENCHMARK_WORKLOAD_ROOT"] = str(workloads)
    return env


def worker_prefix(config: Configuration) -> list[str]:
    command = [sys.executable, "-I", str(Path(__file__).resolve()), "worker"]
    for name in (
        "version",
        "tag",
        "release_commit",
        "wheel_sha256",
        "harness_commit",
        "harness_root",
        "release_root",
        "wheel_path",
    ):
        command.extend(["--" + name.replace("_", "-"), str(getattr(config, name))])
    return command


def execute(config: Configuration, phase: str, worker_args: list[str]) -> int:
    previous_workloads = os.environ.get("TRUTHOUND_BENCHMARK_WORKLOAD_ROOT")
    previous_argv = sys.argv
    try:
        return _execute(config, phase, worker_args)
    finally:
        if previous_workloads is None:
            os.environ.pop("TRUTHOUND_BENCHMARK_WORKLOAD_ROOT", None)
        else:
            os.environ["TRUTHOUND_BENCHMARK_WORKLOAD_ROOT"] = previous_workloads
        sys.argv = previous_argv


def _execute(config: Configuration, phase: str, worker_args: list[str]) -> int:
    provenance = checkout_provenance(config)
    th, wheel_files = verify_installed_wheel(config)
    original_check = th.check
    parity = load_harness(config.harness_root, expected_files=provenance["harness_files"])
    provenance_id = digest_json(provenance)
    workloads_path = config.release_root / "benchmarks/workloads"
    os.environ["TRUTHOUND_BENCHMARK_WORKLOAD_ROOT"] = str(workloads_path)
    if phase == "worker":
        worker = importlib.import_module(NAMESPACE + "._parity_worker")
        sys.argv = [str(Path(__file__)), *worker_args]
        args = worker._parse_args()
        methodology = parity.BenchmarkMethodology()
        require(
            args.warm_iterations == methodology.warm_iterations == 7
            and methodology.worker_threads == 1
            and methodology.name == "truthound-parity-thread-budget-v2",
            "WORKER_MEASUREMENT_POLICY_INVALID",
        )
        require(
            args.manifest.parent == workloads_path and args.manifest.stem in RELEASE_WORKLOADS,
            "WORKER_MANIFEST_INVALID",
        )
        workload = worker.load_workload(args.manifest)
        observation = worker.execute_framework_observation(
            workload,
            framework=args.framework,
            artifact_dir=args.artifact_dir,
            warm_iterations=args.warm_iterations,
        )
        require(th.check is original_check, "PUBLIC_CHECK_REPLACED")
        verify_loaded_origins(th, importlib.metadata.distribution("truthound"), wheel_files)
        verify_harness_origins(config.harness_root)
        require(checkout_provenance(config) == provenance, "PROVENANCE_CHANGED_DURING_RUN")
        metadata = dict(
            observation.metadata,
            release_package_provenance=provenance_id,
            release_package_measurement_integrity=parity._measurement_integrity(
                observation, workload, methodology
            ),
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(replace(observation, metadata=metadata).to_dict(), indent=2)
        )
        return 0

    require(not worker_args, "UNKNOWN_ARGUMENTS")

    class ReleasedWheelRunner(parity.ParityRunner):
        def _worker_command(self):
            return worker_prefix(config)

        def _worker_environment(self):
            return worker_environment(workloads_path)

    result = ReleasedWheelRunner().run_suite(
        "release-ga", frameworks="both", artifact_root=config.output.parent / "run"
    )
    require(th.check is original_check, "PUBLIC_CHECK_REPLACED")
    verify_loaded_origins(th, importlib.metadata.distribution("truthound"), wheel_files)
    verify_harness_origins(config.harness_root)
    require(checkout_provenance(config) == provenance, "PROVENANCE_CHANGED_DURING_RUN")
    expected_pairs = {
        (name, framework) for name in RELEASE_WORKLOADS for framework in ("truthound", "gx")
    }
    actual_pairs = [(item.workload_id, item.framework) for item in result.observations]
    origins_ok = (
        len(actual_pairs) == len(expected_pairs)
        and set(actual_pairs) == expected_pairs
        and all(
            item.metadata.get("release_package_provenance") == provenance_id
            and item.metadata.get("release_package_measurement_integrity") is True
            and (item.framework != "truthound" or item.framework_version == config.version)
            for item in result.observations
        )
    )
    assertion = parity.ParityAssertion(
        name="release-ga:immutable-wheel-provenance",
        passed=origins_ok,
        message="All observations measure the exact released wheel with the separately identified harness.",
    )
    metadata = dict(
        result.metadata,
        release_package=provenance,
        release_claim_ready=bool(result.metadata.get("release_claim_ready")) and origins_ok,
    )
    result = replace(result, assertions=(*result.assertions, assertion), metadata=metadata)
    metadata["release_blockers"] = parity.classify_release_blockers(result)
    parity.write_parity_artifacts(result, config.output)
    parity.write_environment_manifest(result, config.output.parent / "env-manifest.json")
    parity.write_release_summary(
        result,
        config.output.parent / "latest-benchmark-summary.md",
        env_manifest_path=config.output.parent / "env-manifest.json",
    )
    print(
        json.dumps(
            {
                "verified": not result.has_blocking_failures and metadata["release_claim_ready"],
                "release_version": config.version,
                "harness_commit": config.harness_commit,
                "wheel_sha256": config.wheel_sha256,
                "observations": len(result.observations),
            }
        )
    )
    return 1 if result.has_blocking_failures or not metadata["release_claim_ready"] else 0


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("validate", "prepare", "run", "worker", "dependencies"))
    for name in (
        "version",
        "tag",
        "release_commit",
        "wheel_sha256",
        "harness_commit",
        "harness_root",
        "release_root",
        "wheel_path",
    ):
        parser.add_argument(
            "--" + name.replace("_", "-"),
            default=os.environ.get("VERIFY_" + name.upper(), ""),
        )
    parser.add_argument("--result-output", default="benchmark-artifacts/release-ga.json")
    args, remaining = parser.parse_known_args()
    config = Configuration(
        args.version,
        args.tag,
        args.release_commit,
        args.wheel_sha256,
        args.harness_commit,
        Path(args.harness_root).resolve(),
        Path(args.release_root).resolve(),
        Path(args.wheel_path).absolute(),
        Path(args.result_output).resolve(),
    )
    return args.phase, config, remaining


def main() -> int:
    try:
        phase, config, remaining = parse_args()
        if phase == "dependencies":
            require(not remaining, "UNKNOWN_ARGUMENTS")
            write_dependencies(config.output)
            return 0
        if phase == "validate":
            require(not remaining, "UNKNOWN_ARGUMENTS")
            config.validate()
            return 0
        if phase == "prepare":
            require(not remaining, "UNKNOWN_ARGUMENTS")
            prepare(config)
            return 0
        return execute(config, phase, remaining)
    except VerificationError as exc:
        print(json.dumps({"verified": False, "code": str(exc)}), file=sys.stderr)
        return 1
    except Exception as exc:
        print(
            json.dumps(
                {"verified": False, "code": "VERIFIER_EXECUTION_ERROR", "kind": type(exc).__name__}
            ),
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
