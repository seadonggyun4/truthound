"""Synthetic child-process cases, never part of normal pytest discovery."""

import os
import time


def test_boundary():
    time.sleep(float(os.environ.get("NIGHTLY_CASE_SECONDS", "0")))
    assert 2 + 2 == 4


def test_required_failure():
    raise AssertionError("synthetic required failure")
