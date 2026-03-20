from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Any

import yaml

from external_docs import build_source_banner, inject_source_banner, load_external_sources, match_external_source


def _load_manifest(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Manifest must be a mapping: {path}")
    return data


def _copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _copy_directory(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    shutil.copytree(src, dst, dirs_exist_ok=True)


def _validate_relative_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        raise ValueError(f"Manifest paths must be relative: {value}")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Stage the strict public Truthound docs set into a temporary docs root."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("docs/public_docs.yml"),
        help="Path to the public docs manifest.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("build/public-docs"),
        help="Directory to populate with the staged public docs tree.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    docs_root = repo_root / "docs"
    manifest_path = (repo_root / args.manifest).resolve()
    output_dir = (repo_root / args.output_dir).resolve()

    manifest = _load_manifest(manifest_path)
    doc_paths = [_validate_relative_path(value) for value in manifest.get("docs", [])]
    support_files = [_validate_relative_path(value) for value in manifest.get("support_files", [])]
    support_directories = [
        _validate_relative_path(value) for value in manifest.get("support_directories", [])
    ]
    external_sources = load_external_sources(manifest)

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    copied_docs = 0
    for relative_path in doc_paths:
        source_path = docs_root / relative_path
        if not source_path.exists():
            raise FileNotFoundError(f"Public docs entry does not exist: {relative_path}")
        destination_path = output_dir / relative_path
        external_source = match_external_source(relative_path, external_sources)
        if external_source is None:
            _copy_file(source_path, destination_path)
        else:
            banner = build_source_banner(relative_path, external_source)
            rendered = inject_source_banner(source_path.read_text(encoding="utf-8"), banner)
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            destination_path.write_text(rendered, encoding="utf-8")
        copied_docs += 1

    for relative_path in support_files:
        source_path = docs_root / relative_path
        if not source_path.exists():
            raise FileNotFoundError(f"Support file does not exist: {relative_path}")
        _copy_file(source_path, output_dir / relative_path)

    for relative_path in support_directories:
        _copy_directory(docs_root / relative_path, output_dir / relative_path)

    print(
        f"Staged {copied_docs} public markdown pages into {output_dir.relative_to(repo_root)} "
        f"using {manifest_path.relative_to(repo_root)}."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
