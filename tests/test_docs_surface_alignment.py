from __future__ import annotations

from pathlib import Path
import re


REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS_ROOT = REPO_ROOT / "docs"


def _doc(rel_path: str) -> Path:
    return REPO_ROOT / rel_path


def _read(rel_path: str) -> str:
    return _doc(rel_path).read_text(encoding="utf-8")


def _all_docs() -> list[Path]:
    return sorted(DOCS_ROOT.rglob("*.md"))


def _rel(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def _assert_no_matches(
    *,
    patterns: list[str],
    allowlist: set[str] | None = None,
) -> None:
    allowlist = allowlist or set()
    violations: list[str] = []

    compiled = [re.compile(pattern) for pattern in patterns]
    for path in _all_docs():
        rel_path = _rel(path)
        if rel_path in allowlist:
            continue

        text = path.read_text(encoding="utf-8")
        for pattern in compiled:
            if pattern.search(text):
                violations.append(f"{rel_path}: {pattern.pattern}")

    assert not violations, "Found stale 3.0 doc surface patterns:\n" + "\n".join(violations)


def test_no_removed_root_compare_in_current_docs() -> None:
    _assert_no_matches(
        patterns=[
            r"\bth\.compare\(",
            r"from truthound import compare\b",
        ],
        allowlist={
            "docs/getting-started/quickstart.md",
            "docs/guides/migration-2.0.md",
            "docs/guides/migration-3.0.md",
        },
    )


def test_no_stale_checkpoint_result_surface_in_current_checkpoint_docs() -> None:
    for rel_path in (
        "docs/guides/checkpoint/basics.md",
        "docs/guides/checkpoint/index.md",
        "docs/guides/checkpoint/actions/custom.md",
        "docs/guides/configuration/checkpoint-config.md",
        "docs/tutorials/enterprise-setup.md",
    ):
        text = _read(rel_path)
        assert ".validation_result" not in text, rel_path
        assert '["validation_result"]' not in text, rel_path


def test_no_stale_validator_catalog_claims_in_current_docs() -> None:
    _assert_no_matches(
        patterns=[
            r"\b264 validators\b",
            r"\b289\+\b",
            r"\b28 categories\b",
            r"\b400\+ validators\b",
        ],
        allowlist={
            "docs/guides/migration-2.0.md",
            "docs/guides/migration-3.0.md",
            "docs/releases/truthound-2.0.md",
        },
    )


def test_validators_docs_use_registry_backed_wording_instead_of_hardcoded_totals() -> None:
    for rel_path in (
        "docs/guides/validators/index.md",
        "docs/guides/validators/categories.md",
    ):
        text = _read(rel_path)
        assert "list_validators" in text, rel_path
        assert "list_categories" in text, rel_path
        assert re.search(r"\b\d+\s+(loadable\s+)?validators\b", text) is None, rel_path
        assert re.search(r"\b\d+\s+(validator\s+)?categories\b", text) is None, rel_path


def test_no_issue_validator_field_in_current_docs() -> None:
    _assert_no_matches(
        patterns=[r"\bissue\.validator\b"],
        allowlist={
            "docs/guides/migration-2.0.md",
            "docs/guides/migration-3.0.md",
        },
    )


def test_no_removed_exception_summary_or_validation_report_kwargs_in_current_docs() -> None:
    _assert_no_matches(
        patterns=[
            r"\bexception_summary\b",
            r"\bvalidation_report\s*=",
        ],
        allowlist={
            "docs/guides/migration-2.0.md",
            "docs/guides/migration-3.0.md",
        },
    )


def test_readme_preserves_brand_assets_and_slogan() -> None:
    text = _read("README.md")
    assert "docs/assets/truthound_banner.png" in text
    assert "docs/assets/Truthound_icon_banner.png" in text
    assert "Sniffs out bad data." in text
    assert "awesome.re/badge.svg" in text
    assert "Powered%20by-Polars" in text


def test_home_surfaces_define_truthound_as_a_layered_system() -> None:
    readme = _read("README.md")
    home = _read("docs/index.md")

    assert "layered data quality system" in readme
    assert "Truthound Product Line" in readme
    assert "Truthound Core" in readme
    assert "Truthound Orchestration" in readme
    assert "Truthound Dashboard" in readme

    assert "layered data quality system" in home
    assert "Truthound By Layer" in home
    assert "Truthound Core" in home
    assert "Truthound Orchestration" in home
    assert "Truthound Dashboard" in home


def test_concepts_docs_define_explicit_layer_boundaries() -> None:
    text = _read("docs/concepts/index.md").lower()
    assert "data-plane / validation kernel" in text
    assert "execution integration layer" in text
    assert "control-plane" in text


def test_dashboard_term_is_disambiguated_for_datadocs_ui() -> None:
    for rel_path in (
        "docs/guides/datadocs/dashboard.md",
        "docs/guides/datadocs/index.md",
        "docs/cli/index.md",
        "docs/cli/dashboard.md",
    ):
        text = _read(rel_path)
        assert "Truthound Dashboard" in text, rel_path
        assert "Data Docs dashboard" in text or "Data Docs Dashboard UI" in text, rel_path
