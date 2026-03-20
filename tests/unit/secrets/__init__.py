"""Tests for secrets module.

This test package name shadows the stdlib ``secrets`` module when pytest adds
``tests/unit`` to ``sys.path``. Re-export the stdlib module so third-party
libraries that import ``secrets`` still receive the expected API.
"""

from __future__ import annotations

import importlib.util
import sysconfig
from pathlib import Path


def _load_stdlib_secrets():
    stdlib_dir = Path(sysconfig.get_paths()["stdlib"])
    secrets_path = stdlib_dir / "secrets.py"
    spec = importlib.util.spec_from_file_location("_truthound_stdlib_secrets", secrets_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load stdlib secrets module from {secrets_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_stdlib_secrets = _load_stdlib_secrets()

for _name in getattr(_stdlib_secrets, "__all__", ()):
    globals()[_name] = getattr(_stdlib_secrets, _name)

__all__ = list(getattr(_stdlib_secrets, "__all__", ()))
