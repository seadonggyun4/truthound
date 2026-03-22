from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_public_manifest_module():
    module_path = REPO_ROOT / "docs" / "scripts" / "public_manifest.py"
    spec = importlib.util.spec_from_file_location("truthound_docs_public_manifest", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_external_docs_module():
    module_path = REPO_ROOT / "docs" / "scripts" / "external_docs.py"
    spec = importlib.util.spec_from_file_location("truthound_docs_external_docs", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
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
    assert "dashboard/quickstart/install-and-run.md" in docs
    assert "dashboard/concepts/architecture.md" in docs
    assert "dashboard/guides/reports-and-datadocs.md" in docs
    assert "dashboard/operations/ci-and-quality-gates.md" in docs
    assert "dashboard/api-reference/artifacts.md" in docs
    assert "dashboard/reference/saved-view-scope-matrix.md" in docs
    assert "guides/checkpoint/index.md" in docs
    assert "guides/validators/index.md" in docs
    assert "concepts/API_REFERENCE.md" not in docs
    assert "guides/datasources/ARCHITECTURE.md" not in docs
    assert not any((REPO_ROOT / "docs" / "dashboard").rglob("*.md"))


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


def test_public_docs_manifest_keeps_orchestration_counts_without_external_checkout():
    manifest_module = _load_public_manifest_module()
    manifest = manifest_module.load_manifest(REPO_ROOT / "docs" / "public_docs.yml")

    original = manifest_module._optional_external_nav_doc_paths
    try:
        manifest_module._optional_external_nav_doc_paths = (
            lambda repo_root, source: []
            if source.name == "orchestration"
            else original(repo_root, source)
        )
        docs = manifest_module.resolve_public_docs(manifest, REPO_ROOT / "docs")
    finally:
        manifest_module._optional_external_nav_doc_paths = original

    public_urls = {f"/{Path(path).with_suffix('').as_posix()}/" for path in docs if not path.endswith("index.md")}
    public_urls.update(
        "/"
        if path == "index.md"
        else f"/{Path(path).parent.as_posix()}/"
        for path in docs
        if path.endswith("index.md")
    )

    assert "orchestration/testing-ci-ownership.md" in docs
    assert manifest["expected_markdown_count"] == len(docs)
    assert manifest["expected_page_count"] == len(public_urls)


def test_mkdocs_nav_exposes_major_hubs_in_main_and_public_configs():
    expected_labels = [
        "Getting Started",
        "Tutorials",
        "Guides",
        "Dashboard",
        "Reference",
        "Orchestration",
        "Concepts & Architecture",
        "Release Notes",
        "ADRs",
        "Legacy / Archive",
    ]

    for config_path in [REPO_ROOT / "mkdocs.yml", REPO_ROOT / "mkdocs.public.yml"]:
        config = _load_mkdocs(config_path)
        labels = _nav_labels(config.get("nav", []))
        for label in expected_labels:
            assert label in labels
        assert "Experimental" not in labels


def test_mkdocs_dashboard_nav_exposes_major_subsections() -> None:
    expected_dashboard_labels = [
        "Overview",
        "Quickstart",
        "Concepts",
        "Guides",
        "Operations",
        "API Reference",
        "Reference",
        "Migration",
    ]

    for config_path in [REPO_ROOT / "mkdocs.yml", REPO_ROOT / "mkdocs.public.yml"]:
        config = _load_mkdocs(config_path)
        nav = config.get("nav", [])
        dashboard_entry = next(
            (entry["Dashboard"] for entry in nav if isinstance(entry, dict) and "Dashboard" in entry),
            None,
        )
        assert dashboard_entry is not None
        labels = _nav_labels(dashboard_entry)
        for label in expected_dashboard_labels:
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


def test_dashboard_external_banner_markup_and_asset_contract() -> None:
    manifest_module = _load_public_manifest_module()
    external_module = _load_external_docs_module()
    manifest = manifest_module.load_manifest(REPO_ROOT / "docs" / "public_docs.yml")
    sources = external_module.load_external_sources(manifest)
    dashboard = next(source for source in sources if source.name == "dashboard")

    homepage_banner = external_module.build_source_banner(Path("dashboard/index.md"), dashboard)
    inner_banner = external_module.build_source_banner(
        Path("dashboard/guides/reports-and-datadocs.md"),
        dashboard,
    )

    assert '/assets/dashboard/truthound-dashboard-banner.png' in homepage_banner
    assert "dashboard-external-banner--hero" in homepage_banner
    assert "dashboard-external-banner--compact" in inner_banner
    assert '!!! note "Upstream Source"' in homepage_banner
