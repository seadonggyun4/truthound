from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

import yaml

LINK_PATTERN = re.compile(r'!?\[[^\]]*\]\(([^)\s]+)(?:\s+"[^"]*")?\)')
SKIP_PREFIXES = ("http://", "https://", "mailto:", "tel:", "data:")


def _iter_nav_paths(node: Any) -> list[str]:
    paths: list[str] = []
    if isinstance(node, list):
        for item in node:
            paths.extend(_iter_nav_paths(item))
    elif isinstance(node, dict):
        for value in node.values():
            paths.extend(_iter_nav_paths(value))
    elif isinstance(node, str) and node.endswith(".md"):
        paths.append(node)
    return paths


def _mkdocs_paths(mkdocs_file: Path) -> list[Path]:
    raw_text = mkdocs_file.read_text(encoding="utf-8")
    sanitized = re.sub(r"!!python/name:[^\s]+", "python-ref", raw_text)
    config = yaml.safe_load(sanitized) or {}
    docs_dir = (mkdocs_file.parent / config.get("docs_dir", "docs")).resolve()
    nav_paths = _iter_nav_paths(config.get("nav", []))
    return [docs_dir / relative for relative in nav_paths]


def _resolve_docs_root(mkdocs_file: Path | None, repo_root: Path) -> Path:
    if mkdocs_file is None:
        return (repo_root / "docs").resolve()

    raw_text = mkdocs_file.read_text(encoding="utf-8")
    sanitized = re.sub(r"!!python/name:[^\s]+", "python-ref", raw_text)
    config = yaml.safe_load(sanitized) or {}
    return (mkdocs_file.parent / config.get("docs_dir", "docs")).resolve()


def _collect_markdown_files(paths: list[Path], mkdocs_file: Path | None) -> list[Path]:
    files: set[Path] = set()
    for path in paths:
        if path.is_dir():
            files.update(sorted(path.rglob("*.md")))
        elif path.suffix == ".md":
            files.add(path)
    if mkdocs_file is not None:
        files.update(_mkdocs_paths(mkdocs_file))
    return sorted(path.resolve() for path in files if path.exists())


def _candidate_targets(base_dir: Path, target: str, repo_root: Path) -> list[Path]:
    target = target.split("#", 1)[0].split("?", 1)[0]
    if not target:
        return []

    raw = repo_root / target.lstrip("/") if target.startswith("/") else base_dir / target

    candidates = [raw]
    if raw.suffix == "":
        candidates.append(raw.with_suffix(".md"))
        candidates.append(raw / "index.md")
    if raw.is_dir():
        candidates.append(raw / "index.md")
    return candidates


def _exists_case_sensitive(path: Path) -> bool:
    path = path.resolve(strict=False)
    if not path.exists():
        return False

    current = Path(path.anchor)
    for part in path.parts[1:]:
        try:
            names = {child.name for child in current.iterdir()}
        except OSError:
            return False
        if part not in names:
            return False
        current = current / part
    return True


def _find_broken_links(file_path: Path, repo_root: Path, docs_root: Path) -> list[str]:
    text = file_path.read_text(encoding="utf-8")
    failures: list[str] = []

    for match in LINK_PATTERN.finditer(text):
        target = match.group(1)
        if target.startswith("#") or target.startswith(SKIP_PREFIXES):
            continue
        if "{" in target or "}" in target:
            failures.append(f"{file_path}: {target}")
            continue

        candidates = [candidate.resolve(strict=False) for candidate in _candidate_targets(file_path.parent, target, repo_root)]
        if (
            file_path.is_relative_to(docs_root)
            and not target.startswith("/")
            and candidates
            and not any(candidate.resolve().is_relative_to(docs_root) for candidate in candidates)
        ):
            failures.append(f"{file_path}: {target}")
            continue

        if candidates and any(_exists_case_sensitive(candidate) for candidate in candidates):
            continue

        failures.append(f"{file_path}: {target}")

    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description="Check markdown links for local docs pages.")
    parser.add_argument("paths", nargs="*", default=["docs"], help="Markdown files or directories.")
    parser.add_argument(
        "--mkdocs",
        type=Path,
        help="Optional mkdocs.yml path. When provided, all markdown files referenced by nav are checked.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    mkdocs_file = (repo_root / args.mkdocs).resolve() if args.mkdocs else None
    docs_root = _resolve_docs_root(mkdocs_file, repo_root)
    input_paths = [(repo_root / path).resolve() for path in args.paths]

    files = _collect_markdown_files(input_paths, mkdocs_file)
    failures: list[str] = []
    for file_path in files:
        failures.extend(_find_broken_links(file_path, repo_root, docs_root))

    if failures:
        print("Broken markdown links detected:")
        for failure in failures:
            print(f" - {failure}")
        return 1

    print(f"Checked {len(files)} markdown files. No broken local links found.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
