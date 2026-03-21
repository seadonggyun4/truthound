from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_public_manifest_module():
    module_path = REPO_ROOT / "docs" / "scripts" / "public_manifest.py"
    spec = importlib.util.spec_from_file_location("truthound_docs_public_manifest", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_mkdocs(path: Path) -> dict:
    raw_text = path.read_text(encoding="utf-8")
    sanitized = re.sub(r"!!python/name:[^\s]+", "python-ref", raw_text)
    return yaml.safe_load(sanitized) or {}


def _nav_labels(node) -> list[str]:
    labels: list[str] = []
    if isinstance(node, list):
        for item in node:
            labels.extend(_nav_labels(item))
    elif isinstance(node, dict):
        for key, value in node.items():
            labels.append(key)
            labels.extend(_nav_labels(value))
    return labels


def test_public_docs_manifest_exposes_full_portal():
    manifest_module = _load_public_manifest_module()
    manifest = manifest_module.load_manifest(REPO_ROOT / "docs" / "public_docs.yml")
    docs = manifest_module.resolve_public_docs(manifest, REPO_ROOT / "docs")

    assert "index.md" in docs
    assert "tutorials/index.md" in docs
    assert "cli/index.md" in docs
    assert "python-api/index.md" in docs
    assert "reference/index.md" in docs
    assert "dashboard/index.md" in docs
    assert "guides/checkpoint/index.md" in docs
    assert "guides/validators/index.md" in docs
    assert "concepts/API_REFERENCE.md" not in docs
    assert "guides/datasources/ARCHITECTURE.md" not in docs


def test_public_docs_expected_page_count_matches_manifest():
    manifest_module = _load_public_manifest_module()
    manifest = manifest_module.load_manifest(REPO_ROOT / "docs" / "public_docs.yml")
    docs = manifest_module.resolve_public_docs(manifest, REPO_ROOT / "docs")

    public_urls = {f"/{Path(path).with_suffix('').as_posix()}/" for path in docs if not path.endswith("index.md")}
    public_urls.update(
        "/"
        if path == "index.md"
        else f"/{Path(path).parent.as_posix()}/"
        for path in docs
        if path.endswith("index.md")
    )

    assert manifest["expected_page_count"] == len(public_urls)
    assert manifest["expected_markdown_count"] == len(docs)
    assert len(docs) > 36


def test_mkdocs_nav_exposes_major_hubs_in_main_and_public_configs():
    expected_labels = [
        "Getting Started",
        "Tutorials",
        "Guides",
        "Reference",
        "Orchestration",
        "Concepts & Architecture",
        "Release Notes",
        "ADRs",
        "Legacy / Archive",
        "Experimental",
    ]

    for config_path in [REPO_ROOT / "mkdocs.yml", REPO_ROOT / "mkdocs.public.yml"]:
        config = _load_mkdocs(config_path)
        labels = _nav_labels(config.get("nav", []))
        for label in expected_labels:
            assert label in labels


def test_mkdocs_brand_assets_are_preserved():
    main_config = _load_mkdocs(REPO_ROOT / "mkdocs.yml")
    public_config = _load_mkdocs(REPO_ROOT / "mkdocs.public.yml")

    for config in [main_config, public_config]:
        theme = config["theme"]
        assert theme["logo"] == "assets/truthound_icon.png"
        assert theme["favicon"] == "assets/truthound_icon.png"
        assert theme["palette"][0]["primary"] == "custom"
        assert theme["palette"][1]["primary"] == "custom"
