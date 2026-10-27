"""Load repository-only probe helpers without depending on the launcher's sys.path."""

import importlib.util
import sys
from pathlib import Path


def _load(filename):
    name = "truthound_test_probe_" + filename
    if name in sys.modules:
        return sys.modules[name]
    path = Path(__file__).resolve().parents[1] / "verification" / (filename + ".py")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError("Repository verification helper is unavailable")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


OwnedProbe = _load("native_distributed_runner").OwnedProbe
environment = _load("native_distributed_runner").environment
receiver = _load("openlineage_receiver").receiver
