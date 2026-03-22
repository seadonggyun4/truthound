from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from external_docs import discover_external_source_root, load_external_sources
from public_manifest import load_manifest


def _clone_source(*, repo_root: Path, source) -> Path:
    target_dir = repo_root / ".external" / Path(source.repo_name).name

    if target_dir.exists() and not (target_dir / source.normalized_docs_root).exists():
        shutil.rmtree(target_dir)

    target_dir.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "git",
            "clone",
            "--depth",
            "1",
            "--branch",
            source.branch,
            source.repo_url,
            str(target_dir),
        ],
        check=True,
    )
    return target_dir


def ensure_external_sources(
    *,
    repo_root: Path,
    manifest: dict[str, Any],
    selected_names: set[str] | None = None,
) -> int:
    fetched = 0
    for source in load_external_sources(manifest):
        if selected_names and source.name not in selected_names:
            continue

        existing = discover_external_source_root(repo_root, source)
        if existing is not None:
            print(f"Using existing external docs source for {source.name}: {existing}")
            continue

        cloned = _clone_source(repo_root=repo_root, source=source)
        if not (cloned / source.normalized_docs_root).exists():
            raise FileNotFoundError(
                f"Cloned {source.name} into {cloned}, but {source.normalized_docs_root}/ was not found."
            )
        print(f"Fetched external docs source for {source.name}: {cloned}")
        fetched += 1
    return fetched


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Ensure external docs sources declared in the public docs manifest are available."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("docs/public_docs.yml"),
        help="Path to the public docs manifest.",
    )
    parser.add_argument(
        "--source",
        action="append",
        default=[],
        help="Optional source names to fetch. Defaults to all external sources.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Repository root to populate. Defaults to the script's parent repository.",
    )
    args = parser.parse_args()

    repo_root = (args.repo_root or Path(__file__).resolve().parents[2]).resolve()
    manifest = load_manifest((repo_root / args.manifest).resolve())
    selected_names = set(args.source)
    fetched = ensure_external_sources(
        repo_root=repo_root,
        manifest=manifest,
        selected_names=selected_names,
    )
    print(f"External docs sources ready. Newly fetched: {fetched}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
