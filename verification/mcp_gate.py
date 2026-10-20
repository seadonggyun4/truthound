"""Translate actual pytest results into a bounded machine-readable gate."""
import json
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

root = Path(__file__).resolve().parents[1]
with tempfile.TemporaryDirectory(prefix="truthound-mcp-gate-") as temp:
    report = Path(temp) / "results.xml"
    p = subprocess.run([str(root / ".venv/bin/python"), "-m", "pytest", "-q", "tests/test_mcp.py", "tests/test_mcp_bundle.py", "-p", "no:cacheprovider", f"--junitxml={report}"], cwd=root, capture_output=True)
    valid = False
    if report.exists():
        suites = ET.parse(report).getroot().findall("testsuite")
        valid = bool(suites) and sum(int(s.get("tests", 0)) for s in suites) > 0 and all(int(s.get("skipped",0)) == 0 for s in suites)
    passed = p.returncode == 0 and valid
    print(json.dumps({"schema_version":1,"cases":[{"id":i,"status":"PASS" if passed else "FAIL"} for i in ("api", "safety", "package", "docs")]}))
    sys.exit(0 if passed else 1)
