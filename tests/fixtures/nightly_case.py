"""Synthetic child-process cases, never part of normal pytest discovery."""

import os
import time

import pytest


def test_boundary():
    time.sleep(float(os.environ.get("NIGHTLY_CASE_SECONDS", "0")))
    assert 2 + 2 == 4


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
