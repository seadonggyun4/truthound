"""Native SDK limit contexts must not permanently reduce host process limits."""

import subprocess
import sys

import pytest

pytestmark = pytest.mark.contract


@pytest.mark.parametrize("kind", ["memory", "cpu", "combined"])
@pytest.mark.parametrize("failure", [False, True, "startup"])
def test_native_limits_restore_host_state_after_scope_and_exception(kind, failure):
    code = r"""
import sys
from truthound.validators.sdk.enterprise.resources import (
    MemoryLimiter, CPULimiter, CombinedResourceLimiter, ResourceLimits, ResourceMonitor,
)
try:
    import resource
except ImportError:
    resource = None
keys = (resource.RLIMIT_AS, resource.RLIMIT_CPU, resource.RLIMIT_NOFILE) if resource else ()
before = {key: resource.getrlimit(key) for key in keys}
mode, fail = sys.argv[1], sys.argv[2] != "False"
if sys.argv[2] == "startup":
    def broken_start(self):
        raise RuntimeError("synthetic scoped validator failure")
    ResourceMonitor.start = broken_start
limiter = {
    "memory": lambda: MemoryLimiter(4096),
    "cpu": lambda: CPULimiter(100000),
    "combined": lambda: CombinedResourceLimiter(ResourceLimits(
        max_memory_mb=0, max_cpu_seconds=100000, max_file_descriptors=4096,
    )),
}[mode]()
try:
    with limiter.enforce():
        if resource:
            assert {key: resource.getrlimit(key)[1] for key in keys} == {
                key: value[1] for key, value in before.items()
            }, "Host hard limits must not be lowered"
            if mode in {"cpu", "combined"}:
                assert resource.getrlimit(resource.RLIMIT_CPU)[0] <= 100000
        if fail:
            raise RuntimeError("synthetic scoped validator failure")
except RuntimeError as error:
    assert fail and str(error) == "synthetic scoped validator failure"
assert not resource or {key: resource.getrlimit(key) for key in keys} == before
print("native-host-limits-restored" if resource else "monitor-only-platform-cleanup")
"""
    result = subprocess.run(
        [sys.executable, "-c", code, kind, str(failure)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() in {
        "native-host-limits-restored",
        "monitor-only-platform-cleanup",
    }


@pytest.mark.parametrize(
    "original, expected", [((-1, -1), 80), ((-1, 100), 80), ((50, 100), 50), ((100, 100), 80)]
)
def test_temporary_limit_preserves_finite_and_infinite_host_constraints(original, expected):
    from truthound.validators.sdk.enterprise.resources import _bounded_soft_limit

    assert _bounded_soft_limit(80, original) == expected


def test_positive_fractional_cpu_limit_is_not_silently_disabled(monkeypatch):
    from types import SimpleNamespace

    from truthound.validators.sdk.enterprise.resources import (
        CombinedResourceLimiter,
        ResourceLimits,
        ResourceMonitor,
    )

    state = {1: (-1, -1), 2: (-1, -1), 3: (-1, -1)}
    calls = []

    def set_limit(key, value):
        calls.append((key, value))
        state[key] = value

    fake = SimpleNamespace(
        RLIMIT_AS=1,
        RLIMIT_CPU=2,
        RLIMIT_NOFILE=3,
        getrlimit=lambda key: state[key],
        setrlimit=set_limit,
    )
    monkeypatch.setitem(sys.modules, "resource", fake)
    monkeypatch.setattr(ResourceMonitor, "start", lambda self: None)
    monkeypatch.setattr(ResourceMonitor, "stop", lambda self: None)
    with CombinedResourceLimiter(
        ResourceLimits(
            max_memory_mb=0,
            max_cpu_seconds=0.5,
            max_file_descriptors=0,
        )
    ).enforce():
        assert state[2] == (0, -1)
    assert calls == [(2, (0, -1)), (2, (-1, -1))]
