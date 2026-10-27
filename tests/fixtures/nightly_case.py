"""Synthetic child-process cases, never part of normal pytest discovery."""

import os
import time

import pytest


def test_boundary():
    time.sleep(float(os.environ.get("NIGHTLY_CASE_SECONDS", "0")))
    assert 2 + 2 == 4


def native_value():
    return os.getpid(), 42


def test_spawned_callable():
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    # This boundary tests the manifest's spawn import path. Core process
    # strategy tests separately exercise native execution and termination.
    with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn")) as worker:
        pid, value = worker.submit(native_value).result()
    assert pid != os.getpid() and value == 42


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
