"""Synthetic child-process cases, never part of normal pytest discovery."""

import os
import time

import pytest


def test_boundary():
    time.sleep(float(os.environ.get("NIGHTLY_CASE_SECONDS", "0")))
    assert 2 + 2 == 4


def native_value():
    return 42


def test_spawned_callable():
    from truthound.profiler.process_timeout import ProcessExecutionStrategy

    result = ProcessExecutionStrategy().execute(native_value, timeout_seconds=5)
    assert result.success and result.value == 42


def test_required_failure():
    raise AssertionError("synthetic required failure")


def test_subtests(subtests):
    for i in range(3):
        with subtests.test(i=i):
            assert i >= 0


def test_subtest_failure(subtests):
    for i in range(3):
        with subtests.test(i=i):
            assert i < 2


def test_subtest_skip(subtests):
    for i in range(3):
        with subtests.test(i=i):
            if i == 1:
                pytest.skip("synthetic runtime skip must not pass the gate")
            assert i >= 0
