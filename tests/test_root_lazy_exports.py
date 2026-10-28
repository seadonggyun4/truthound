"""Cold public imports stay light; first use remains the actual native facade."""

import os
from pathlib import Path
import subprocess
import sys

import pytest
import truthound

pytestmark = pytest.mark.contract


def run_fresh(tmp_path, script):
    root = str(Path(truthound.__file__).resolve().parent.parent)
    environment = {**os.environ, 'HOME': str(tmp_path), 'USERPROFILE': str(tmp_path),
                   'POLARS_MAX_THREADS': '1', 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'}
    return subprocess.run([sys.executable, '-I', '-c',
        'import sys; sys.path.insert(0, sys.argv[1]);\n' + script, root, str(tmp_path)],
        capture_output=True, text=True, timeout=30, env=environment)


def test_cold_import_defers_facade_and_polars_without_creating_workspace(tmp_path):
    result = run_fresh(tmp_path, '''
import truthound as th
assert 'truthound.api' not in sys.modules
assert 'polars' not in sys.modules
assert 'check' in dir(th) and 'ValidationRunResult' in dir(th)
assert th.__version__
''')
    assert result.returncode == 0, result.stderr
    assert not (tmp_path / '.truthound').exists()


def test_concurrent_first_access_preserves_native_validation_and_reload(tmp_path):
    result = run_fresh(tmp_path, '''
import concurrent.futures
import importlib
import threading
from pathlib import Path
import truthound as th
barrier = threading.Barrier(8)
first_names = ('check', 'ValidationSuite', 'SchemaSpec', 'Schema', 'learn',
               'TruthoundContext', 'get_context', 'validator')
def resolve(index):
    barrier.wait()
    getattr(th, first_names[index])
    return th.check, th.ValidationSuite, th.SchemaSpec
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
    values = list(pool.map(resolve, range(8)))
assert all(value == values[0] for value in values)
from truthound.api import check
from truthound.core import ValidationSuite, SchemaSpec
assert values[0] == (check, ValidationSuite, SchemaSpec)
from truthound.context import TruthoundContext, TruthoundContextConfig
import polars as pl
context = TruthoundContext(Path(sys.argv[2]), config=TruthoundContextConfig(
    persist_runs=False, persist_docs=False, auto_create_baseline=False, auto_create_workspace=False))
run = th.check(pl.DataFrame({'x': [1, None]}), validators=['null'], context=context, result_format='complete')
assert isinstance(run, th.ValidationRunResult)
assert run.row_count == 2 and len(run.issues) == 1 and run.checks[0].issue_count == 1
import truthound.api as api
original = api.check
sentinel = lambda: 'reloaded'
try:
    api.check = sentinel
    importlib.reload(th)
    assert th.check is sentinel
finally:
    api.check = original
    importlib.reload(th)
assert th.check is original
''')
    assert result.returncode == 0, result.stderr


def test_failed_first_import_preserves_error_and_can_be_retried(tmp_path):
    result = run_fresh(tmp_path, '''
import truthound as th
from truthound import _lazy
original = _lazy.importlib.import_module
def fail_api(name):
    if name == 'truthound.api':
        raise ModuleNotFoundError('missing test dependency', name='test_dependency')
    return original(name)
_lazy.importlib.import_module = fail_api
try:
    try:
        th.check
    except ModuleNotFoundError as error:
        assert error.name == 'test_dependency'
    else:
        raise AssertionError('dependency failure was hidden')
    assert 'check' not in vars(th)
finally:
    _lazy.importlib.import_module = original
from truthound.api import check
assert th.check is check
''')
    assert result.returncode == 0, result.stderr
