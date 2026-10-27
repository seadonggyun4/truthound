"""Public build evidence never sweeps private files or overwrites on failure."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.contract
ROOT = Path(__file__).resolve().parents[1]


def load():
    spec = importlib.util.spec_from_file_location(
        "deployment_identity", ROOT / "docs/scripts/write_deployment_manifest.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare(site, module):
    for url in module.PAGES:
        page = site / url.strip("/") / "index.html"
        page.parent.mkdir(parents=True)
        page.write_text("<html>Public " + url + "</html>", encoding="utf-8")


def test_public_manifest_hashes_exact_pages_without_collecting_private_build_state(tmp_path):
    module = load()
    prepare(tmp_path, module)
    marker = "synthetic-private-environment-value"
    (tmp_path / "private.json").write_text(marker)
    value = module.write_manifest(tmp_path, "a" * 40, "3.3.0")
    stored = (tmp_path / "deployment-manifest.json").read_text()
    assert marker not in stored and str(tmp_path) not in stored
    assert json.loads(stored) == value
    assert set(value["page_sha256"]) == set(module.PAGES)
    for url, digest in value["page_sha256"].items():
        assert (
            hashlib.sha256((tmp_path / url.strip("/") / "index.html").read_bytes()).hexdigest()
            == digest
        )


@pytest.mark.parametrize("failure", ["missing", "invalid_commit", "invalid_version"])
def test_invalid_build_identity_cannot_replace_previous_deployment_evidence(tmp_path, failure):
    module = load()
    prepare(tmp_path, module)
    destination = tmp_path / "deployment-manifest.json"
    destination.write_text("previous evidence")
    commit, version = "a" * 40, "3.3.0"
    if failure == "missing":
        (tmp_path / module.PAGES[-1].strip("/") / "index.html").unlink()
    elif failure == "invalid_commit":
        commit = "private-build-value"
    else:
        version = "private-build-value"
    with pytest.raises((ValueError, OSError)):
        module.write_manifest(tmp_path, commit, version)
    assert destination.read_text() == "previous evidence"
