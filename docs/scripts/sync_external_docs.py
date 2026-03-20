from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from external_docs import load_mkdocs_config, load_nav_doc_paths


def _copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _copy_directory(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    shutil.copytree(src, dst, dirs_exist_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Mirror the public external docs surface defined by an upstream MkDocs nav."
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        required=True,
        help="Repository root of the upstream docs project.",
    )
    parser.add_argument(
        "--target-dir",
        type=Path,
        required=True,
        help="Destination directory to populate with the mirrored docs snapshot.",
    )
    parser.add_argument(
        "--mkdocs-file",
        type=Path,
        default=Path("mkdocs.yml"),
        help="Path to the upstream MkDocs config relative to --source-root.",
    )
    parser.add_argument(
        "--include-dir",
        action="append",
        default=[],
        help="Extra docs-relative directories to mirror alongside the nav pages.",
    )
    args = parser.parse_args()

    source_root = args.source_root.resolve()
    target_dir = args.target_dir.resolve()
    mkdocs_file = (source_root / args.mkdocs_file).resolve()

    config = load_mkdocs_config(mkdocs_file)
    docs_dir = (mkdocs_file.parent / config.get("docs_dir", "docs")).resolve()
    nav_paths = load_nav_doc_paths(mkdocs_file)

    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    copied_docs = 0
    for relative_path in nav_paths:
        source_path = docs_dir / relative_path
        if not source_path.exists():
            raise FileNotFoundError(f"Upstream nav entry does not exist: {relative_path}")
        _copy_file(source_path, target_dir / relative_path)
        copied_docs += 1

    for include_dir in args.include_dir:
        relative_dir = Path(include_dir)
        if relative_dir.is_absolute():
            raise ValueError(f"--include-dir must be relative: {include_dir}")
        _copy_directory(docs_dir / relative_dir, target_dir / relative_dir)

    print(
        f"Mirrored {copied_docs} markdown pages from {mkdocs_file.relative_to(source_root)} "
        f"into {target_dir}."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
