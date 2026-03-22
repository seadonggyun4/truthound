from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from external_docs import external_nav_doc_paths, load_external_sources


def _optional_external_nav_doc_paths(repo_root: Path, source: Any) -> list[Path]:
    try:
        return external_nav_doc_paths(repo_root, source)
    except FileNotFoundError:
        return []


def load_manifest(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Manifest must be a mapping: {path}")
    return data


def _validate_relative_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        raise ValueError(f"Manifest paths must be relative: {value}")
    return path


def _normalize_prefix(value: str) -> str:
    normalized = _validate_relative_path(value).as_posix().strip("/")
    return f"{normalized}/" if normalized else ""


def _matches_prefix(doc_path: str, prefix: str) -> bool:
    normalized_doc = doc_path.strip("/")
    normalized_prefix = prefix.strip("/")
    if not normalized_prefix:
        return True
    return normalized_doc == normalized_prefix or normalized_doc.startswith(
        f"{normalized_prefix}/"
    )


def resolve_public_docs(manifest: dict[str, Any], docs_root: Path) -> list[str]:
    docs: set[str] = set()
    repo_root = docs_root.parent
    external_sources = load_external_sources(manifest)
    external_by_prefix = {
        source.normalized_prefix: source for source in external_sources
    }

    for value in manifest.get("docs", []):
        relative = _validate_relative_path(value)
        source_path = docs_root / relative
        if source_path.exists():
            if source_path.suffix != ".md":
                raise ValueError(f"Public docs entry must be markdown: {relative}")
            docs.add(relative.as_posix())
            continue
        prefix = relative.parts[0] if relative.parts else ""
        source = external_by_prefix.get(prefix)
        if source is None:
            raise FileNotFoundError(f"Public docs entry does not exist: {relative}")
        relative_tail = Path(*relative.parts[1:]).as_posix()
        available = {path.as_posix() for path in external_nav_doc_paths(repo_root, source)}
        if relative_tail not in available:
            raise FileNotFoundError(f"Public external docs entry does not exist: {relative}")
        docs.add(relative.as_posix())

    for value in manifest.get("include_prefixes", []):
        relative = _validate_relative_path(value)
        source_path = docs_root / relative
        prefix = relative.as_posix().strip("/")
        source = external_by_prefix.get(prefix)
        if source_path.exists():
            if source_path.is_file():
                if source_path.suffix != ".md":
                    raise ValueError(f"Public docs prefix must target markdown: {relative}")
                docs.add(relative.as_posix())
                continue

            local_docs = {
                child.relative_to(docs_root).as_posix()
                for child in source_path.rglob("*.md")
            }
            if local_docs:
                docs.update(local_docs)
                if source is None:
                    continue

            if source is not None:
                for child in _optional_external_nav_doc_paths(repo_root, source):
                    docs.add(Path(prefix, child).as_posix())
                continue

            continue

        if source is None:
            raise FileNotFoundError(f"Public docs prefix does not exist: {relative}")
        for child in _optional_external_nav_doc_paths(repo_root, source):
            docs.add(Path(prefix, child).as_posix())

    excluded_docs = {
        _validate_relative_path(value).as_posix()
        for value in manifest.get("excluded_docs", [])
    }
    excluded_prefixes = [
        _normalize_prefix(value) for value in manifest.get("excluded_prefixes", [])
    ]

    filtered = [
        doc_path
        for doc_path in sorted(docs)
        if doc_path not in excluded_docs
        and not any(_matches_prefix(doc_path, prefix) for prefix in excluded_prefixes)
    ]
    return filtered
