"""Required MCP lane collection/source/execution identity, independent of JUnit labels."""

import hashlib
import inspect
import json
from collections import Counter
from pathlib import Path

import pytest


def partition(nodes, count, index):
    if not 1 <= count <= 16 or not 0 <= index < count:
        raise ValueError("MCP_INVALID_SHARD")
    return {
        node: value
        for node, value in nodes.items()
        if int.from_bytes(hashlib.sha256(node.encode()).digest(), "big") % count == index
    }


def identity(item):
    path = Path(item.path).resolve()
    module = Path(item.module.__file__).resolve()
    implementation = inspect.getsourcefile(inspect.unwrap(item.obj))
    if module != path or implementation is None or Path(implementation).resolve() != path:
        raise ValueError("MCP_TEST_SOURCE_MISMATCH")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sources():
    import truthound

    root = Path(truthound.__file__).resolve().parent
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.suffix in {".py", ".json"}
    }


def supporting_sources():
    """Fixture/helper changes invalidate a proof just like a test-module edit."""
    root = Path(__file__).resolve().parents[1]
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for directory in (root / "tests", root / "verification")
        for path in sorted(directory.rglob("*.py"))
        if path.is_file()
    }


def pytest_addoption(parser):
    parser.addoption("--mcp-source-proof")
    parser.addoption("--mcp-shard-count", type=int, default=1)
    parser.addoption("--mcp-shard-index", type=int, default=0)


def project_sources():
    root = Path(__file__).resolve().parents[1]
    return {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in ("pyproject.toml", "uv.lock")
    }


def pytest_configure(config):
    if config.getoption("--mcp-source-proof"):
        config.pluginmanager.register(Proof(config), "mcp-source-proof")


class Proof:
    def __init__(self, config):
        self.config, self.collected, self.phases = config, {}, []
        self.before = sources()
        self.support_before = supporting_sources()
        self.project_before = project_sources()
        self.test_paths = {}
        self.invalid = False
        self.full_collection = {}
        self.shard_count = config.getoption("--mcp-shard-count")
        self.shard_index = config.getoption("--mcp-shard-index")

    @pytest.hookimpl(trylast=True)
    def pytest_collection_modifyitems(self, session, config, items):
        nodes = [item.nodeid for item in items]
        required = {Path(argument.split("::", 1)[0]).resolve() for argument in config.args}
        observed = {Path(item.path).resolve() for item in items}
        try:
            if not nodes or len(set(nodes)) != len(nodes) or required != observed:
                raise ValueError("MCP_TEST_COLLECTION_MISMATCH")
            self.full_collection = {item.nodeid: identity(item) for item in items}
            self.collected = partition(self.full_collection, self.shard_count, self.shard_index)
            if not self.collected:
                raise ValueError("MCP_EMPTY_SHARD")
            rejected = [item for item in items if item.nodeid not in self.collected]
            items[:] = [item for item in items if item.nodeid in self.collected]
            if rejected:
                config.hook.pytest_deselected(items=rejected)
            self.test_paths = {item.nodeid: Path(item.path).resolve() for item in items}
        except ValueError as exc:
            self.invalid = True
            raise pytest.UsageError(str(exc)) from exc

    def pytest_runtest_logstart(self, nodeid, location):
        # Job termination can prevent final JUnit/proof output. This is progress
        # evidence only; an unfinished shard still cannot pass the aggregate.
        print(f"MCP_START {nodeid}", flush=True)

    def pytest_runtest_logreport(self, report):
        self.phases.append(
            (report.nodeid, report.when, report.outcome, bool(getattr(report, "wasxfail", False)))
        )

    def pytest_sessionfinish(self, session, exitstatus):
        counts = Counter((node, phase) for node, phase, _, _ in self.phases)
        expected = {
            (node, phase) for node in self.collected for phase in ("setup", "call", "teardown")
        }
        valid = (
            not self.invalid
            and bool(self.collected)
            and set(counts) == expected
            and all(count == 1 for count in counts.values())
            and all(outcome == "passed" and not xfail for _, _, outcome, xfail in self.phases)
            and self.before == sources()
            and self.support_before == supporting_sources()
            and self.project_before == project_sources()
            and all(
                hashlib.sha256(path.read_bytes()).hexdigest() == self.collected[node]
                for node, path in self.test_paths.items()
            )
            and exitstatus == 0
        )
        proof = {
            "valid": valid,
            "collection": self.collected,
            "package_sources": self.before,
            "test_and_verification_sources": self.support_before,
            "project_sources": self.project_before,
            "tests": len(self.collected),
            "execution_phases": len(self.phases),
            "full_collection": self.full_collection,
            "shard_count": self.shard_count,
            "shard_index": self.shard_index,
        }
        Path(self.config.getoption("--mcp-source-proof")).write_text(
            json.dumps(proof, sort_keys=True), encoding="utf-8"
        )
        if not valid:
            session.exitstatus = pytest.ExitCode.TESTS_FAILED
