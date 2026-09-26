"""Verify built artifacts locally or over existing, strict-known-host SSH.

Only synthetic tests, reviewed distributions and dependency wheels are transferred.
No server dependency installation, source access, services or database operations.
Credentials are handled by SSH's terminal prompt, not this program.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import shlex
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

RECEIVER = r'''
import hashlib, io, json, os, pathlib, platform, subprocess, sys, tarfile, tempfile, zipfile
expected = json.loads(sys.argv[1])
payload = sys.stdin.buffer.read(100 * 1024 * 1024 + 1)
if len(payload) > 100 * 1024 * 1024:
    raise SystemExit(2)
def safe(name):
    p = pathlib.PurePosixPath(name)
    if p.is_absolute() or '..' in p.parts or '\\' in name:
        raise ValueError('unsafe_member')
    if any(x in ('.golem', '.git', 'AGENTS.md', 'CLAUDE.md', '.env') or x.startswith('.env.') for x in p.parts):
        raise ValueError('private_member')
    return p
def unzip(path, dest):
    with zipfile.ZipFile(path) as z:
        if sum(i.file_size for i in z.infolist()) > 350 * 1024 * 1024:
            raise ValueError('expanded_size_limit')
        for item in z.infolist():
            relative = safe(item.filename)
            if item.is_dir():
                continue
            target = dest / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(z.read(item))
TEST = r"""
import hashlib, importlib, io, json, pathlib, sys, unittest
import truthound
assert pathlib.Path(truthound.__file__).is_relative_to(pathlib.Path(sys.argv[1]))
suite = unittest.defaultTestLoader.discover(sys.argv[2])
r = unittest.TextTestRunner(stream=io.StringIO()).run(suite)
import test_contract as f
from truthound.schema_audit import semantic_digest
print(json.dumps(dict(tests=r.testsRun, failures=len(r.failures), errors=len(r.errors), skipped=len(r.skipped),
    failed_cases=[case.id() for case,_ in r.failures+r.errors],
    semantic_digests=[semantic_digest(x()) for x in (f.standard,f.design,f.catalog)])))
raise SystemExit(0 if r.wasSuccessful() and r.testsRun >= 26 and not r.skipped else 1)
"""
with tempfile.TemporaryDirectory(prefix='truthound-erd-phase1-') as temporary:
    root = pathlib.Path(temporary)
    with tarfile.open(fileobj=io.BytesIO(payload), mode='r:') as archive:
        entries = archive.getmembers()
        if len(entries) != len(expected) or {e.name for e in entries} != set(expected):
            raise ValueError('unexpected_members')
        for entry in entries:
            relative = safe(entry.name)
            if not entry.isfile() or entry.size > 60 * 1024 * 1024:
                raise ValueError('invalid_member')
            data = archive.extractfile(entry).read()
            if hashlib.sha256(data).hexdigest() != expected[entry.name]:
                raise ValueError('digest_mismatch')
            target = root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
    deps = root / 'deps'
    for wheel in (root/'dependencies').glob('*.whl') if (root/'dependencies').exists() else []:
        unzip(wheel, deps)
    wheelroot = root/'wheel'
    for wheel in (root/'artifacts').glob('*.whl'):
        unzip(wheel, wheelroot)
    sdistroot = root/'sdist'
    for source in (root/'artifacts').glob('*.tar.gz'):
        with tarfile.open(source) as archive:
            for entry in archive.getmembers():
                relative = safe(entry.name)
                if not entry.isfile():
                    if not entry.isdir(): raise ValueError('invalid_sdist_member')
                    continue
                target = sdistroot/relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.extractfile(entry).read())
    source = next(sdistroot.iterdir())/'src'
    runs = {}
    for label, library in [('wheel', wheelroot), ('sdist', source)]:
        env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(library), str(deps)]),
                   PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1')
        for key in ('PYTHONSTARTUP','PYTHONHOME'):
            env.pop(key, None)
        r = subprocess.run([sys.executable, '-s', '-c', TEST, str(library), str(root/'tests')],
                           cwd=root, env=env, capture_output=True, timeout=90)
        if not r.stdout:
            runs[label] = dict(status='ERROR', reason='import_or_execution_failed', exit_code=r.returncode)
        else:
            runs[label] = json.loads(r.stdout)
            runs[label]['exit_code'] = r.returncode
    report = dict(scope='synthetic-phase1-package-contracts', runs=runs,
                  python=platform.python_version(), architecture=platform.machine(), file_digests=expected)
report['temporary_directory_removed'] = not root.exists()
report['status'] = 'PASS' if all(r.get('exit_code') == 0 for r in runs.values()) and report['temporary_directory_removed'] else 'FAIL'
print(json.dumps(report))
raise SystemExit(0 if report['status'] == 'PASS' else 1)
'''


def bundle(artifacts: Path, dependencies: Path | None, *, include_standards: bool = False, include_implementation: bool = False) -> tuple[bytes, dict]:
    files = [("tests/test_contract.py", Path(__file__).resolve().parents[1] / "tests/schema_audit/test_contract.py")]
    if include_standards or include_implementation:
        files.append(("tests/test_standards.py", Path(__file__).resolve().parents[1] / "tests/schema_audit/test_standards.py"))
    if include_implementation:
        files.append(("tests/test_implementation.py", Path(__file__).resolve().parents[1] / "tests/schema_audit/test_implementation.py"))
    wheels = list(artifacts.glob("truthound-*.whl"))
    sources = list(artifacts.glob("truthound-*.tar.gz"))
    if len(wheels) != 1 or len(sources) != 1:
        raise ValueError("exactly_one_distribution_required")
    files.extend(("artifacts/" + p.name, p) for p in wheels + sources)
    if dependencies:
        files.extend(("dependencies/" + p.name, p) for p in sorted(dependencies.glob("*.whl")))
    out, digests = io.BytesIO(), {}
    with tarfile.open(fileobj=out, mode="w") as archive:
        for name, path in files:
            if path.is_symlink() or not path.is_file():
                raise ValueError("invalid_source")
            data = path.read_bytes()
            digests[name] = hashlib.sha256(data).hexdigest()
            item = tarfile.TarInfo(name)
            item.mode, item.size = 0o600, len(data)
            archive.addfile(item, io.BytesIO(data))
    return out.getvalue(), digests


def main(*, include_standards: bool = False, include_implementation: bool = False) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--dependencies", type=Path)
    parser.add_argument("--ssh-target")
    parser.add_argument("--remote-python", default="python3.11")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    data, digests = bundle(args.artifacts, args.dependencies, include_standards=include_standards, include_implementation=include_implementation)
    if args.ssh_target and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.@-]*", args.ssh_target):
        parser.error("invalid_ssh_target")
    if not re.fullmatch(r"python3(?:\.\d+)?", args.remote_python):
        parser.error("invalid_interpreter")
    with tempfile.TemporaryDirectory(prefix="erd-phase1-ssh-") as temp:
        base = ["ssh", "-S", temp + "/control", "-o", "StrictHostKeyChecking=yes", "-o", "ConnectTimeout=8"]
        authenticated = False
        try:
            if args.ssh_target:
                auth = subprocess.run(base + ["-M", "-f", "-N", args.ssh_target], check=False)
                if auth.returncode:
                    return auth.returncode
                authenticated = True
                command = base + ["-o", "BatchMode=yes", args.ssh_target,
                    args.remote_python + " -c " + shlex.quote(RECEIVER) + " " + shlex.quote(json.dumps(digests))]
            else:
                command = [sys.executable, "-c", RECEIVER, json.dumps(digests)]
            result = subprocess.run(command, input=data, capture_output=True, timeout=240)
            try:
                report = json.loads(result.stdout)
            except ValueError:
                print(json.dumps(dict(status="ERROR", reason="receiver_failed", exit_code=result.returncode)))
                return 2
            if report.get("file_digests") != digests or report.get("scope") != "synthetic-phase1-package-contracts":
                raise ValueError("receipt_mismatch")
            args.report.write_text(json.dumps(report, indent=2) + "\n")
            args.report.chmod(0o600)
            print(json.dumps({key: value for key, value in report.items() if key != "file_digests"}))
            return result.returncode
        finally:
            if authenticated:
                subprocess.run(base + ["-O", "exit", args.ssh_target], capture_output=True, timeout=10)


if __name__ == "__main__":
    raise SystemExit(main())
