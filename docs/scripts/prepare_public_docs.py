from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from external_docs import (
    build_source_banner,
    discover_external_source_root,
    inject_source_banner,
    load_external_sources,
    load_mkdocs_config,
    load_nav_doc_paths,
    match_external_source,
)
from public_manifest import load_manifest, resolve_public_docs


def _copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _copy_directory(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    shutil.copytree(src, dst, dirs_exist_ok=True)


def _parse_external_overrides(values: list[str]) -> dict[str, Path]:
    overrides: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--external-root values must use NAME=PATH")
        name, raw_path = value.split("=", 1)
        overrides[name.strip()] = Path(raw_path).expanduser().resolve()
    return overrides


def _external_source_file(
    repo_root: Path,
    relative_path: Path,
    *,
    source,
    overrides: dict[str, Path],
) -> Path:
    source_root = discover_external_source_root(repo_root, source, overrides)
    if source_root is None:
        raise FileNotFoundError(
            f"Unable to locate external source checkout for {source.name!r}. "
            f"Provide --external-root {source.name}=PATH or set "
            f"TRUTHOUND_EXTERNAL_SOURCE_{source.name.upper()}."
        )
    suffix = relative_path.as_posix().lstrip("/")[len(source.normalized_prefix) + 1 :]
    return source_root / source.normalized_docs_root / suffix


def _copy_external_doc(
    repo_root: Path,
    relative_path: Path,
    destination_path: Path,
    *,
    source,
    overrides: dict[str, Path],
) -> None:
    source_path = _external_source_file(
        repo_root,
        relative_path,
        source=source,
        overrides=overrides,
    )
    if not source_path.exists():
        raise FileNotFoundError(
            f"External docs entry does not exist for {source.name}: {relative_path}"
        )
    banner = build_source_banner(relative_path, source)
    rendered = inject_source_banner(source_path.read_text(encoding="utf-8"), banner)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    destination_path.write_text(rendered, encoding="utf-8")


def _stage_full_docs(
    *,
    repo_root: Path,
    docs_root: Path,
    output_dir: Path,
    manifest: dict[str, object],
    overrides: dict[str, Path],
) -> int:
    _copy_directory(docs_root, output_dir)

    external_sources = load_external_sources(manifest)
    copied_docs = len(list(output_dir.rglob("*.md")))

    for source in external_sources:
        source_root = discover_external_source_root(repo_root, source, overrides)
        if source_root is None:
            continue

        for relative_path in load_nav_doc_paths(source_root / "mkdocs.yml"):
            prefixed_path = Path(source.normalized_prefix) / relative_path
            destination_path = output_dir / prefixed_path
            _copy_external_doc(
                repo_root,
                prefixed_path,
                destination_path,
                source=source,
                overrides=overrides,
            )
        copied_docs = len(list(output_dir.rglob("*.md")))

    return copied_docs


def _stage_public_docs(
    *,
    repo_root: Path,
    docs_root: Path,
    output_dir: Path,
    manifest: dict[str, object],
    overrides: dict[str, Path],
) -> int:
    doc_paths = [Path(value) for value in resolve_public_docs(manifest, docs_root)]
    support_files = [Path(value) for value in manifest.get("support_files", [])]
    support_directories = [
        Path(value) for value in manifest.get("support_directories", [])
    ]
    external_sources = load_external_sources(manifest)

    copied_docs = 0
    for relative_path in doc_paths:
        source_path = docs_root / relative_path
        destination_path = output_dir / relative_path
        external_source = match_external_source(relative_path, external_sources)
        if external_source is None:
            if not source_path.exists():
                raise FileNotFoundError(f"Public docs entry does not exist: {relative_path}")
            _copy_file(source_path, destination_path)
        elif source_path.exists():
            banner = build_source_banner(relative_path, external_source)
            rendered = inject_source_banner(source_path.read_text(encoding="utf-8"), banner)
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            destination_path.write_text(rendered, encoding="utf-8")
        else:
            _copy_external_doc(
                repo_root,
                relative_path,
                destination_path,
                source=external_source,
                overrides=overrides,
            )
        copied_docs += 1

    for relative_path in support_files:
        source_path = docs_root / relative_path
        if not source_path.exists():
            raise FileNotFoundError(f"Support file does not exist: {relative_path}")
        _copy_file(source_path, output_dir / relative_path)

    for relative_path in support_directories:
        _copy_directory(docs_root / relative_path, output_dir / relative_path)

    expected_markdown_count = manifest.get("expected_markdown_count")
    if expected_markdown_count is not None and copied_docs != int(expected_markdown_count):
        raise ValueError(
            "Public docs manifest expected "
            f"{expected_markdown_count} staged markdown pages, found {copied_docs}."
        )
    return copied_docs


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Stage Truthound docs for strict full or public builds."
    )
    parser.add_argument(
        "--mode",
        choices=["full", "public"],
        default="public",
        help="Stage the full docs tree or the public docs tree.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("docs/public_docs.yml"),
        help="Path to the public docs manifest.",
    )
    parser.add_argument(
        "--mkdocs-file",
        type=Path,
        default=None,
        help="MkDocs config to mirror for full mode. Defaults to mkdocs.yml.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to populate with the staged docs tree.",
    )
    parser.add_argument(
        "--external-root",
        action="append",
        default=[],
        help="Override external source checkout roots with NAME=PATH.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    docs_root = repo_root / "docs"
    manifest_path = (repo_root / args.manifest).resolve()
    manifest = load_manifest(manifest_path)
    overrides = _parse_external_overrides(args.external_root)

    default_output = Path("build/public-docs" if args.mode == "public" else "build/full-docs")
    output_dir = (repo_root / (args.output_dir or default_output)).resolve()

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "public":
        copied_docs = _stage_public_docs(
            repo_root=repo_root,
            docs_root=docs_root,
            output_dir=output_dir,
            manifest=manifest,
            overrides=overrides,
        )
        print(
            f"Staged {copied_docs} public markdown pages into "
            f"{output_dir.relative_to(repo_root)} using {manifest_path.relative_to(repo_root)}."
        )
        return 0

    mkdocs_file = (repo_root / (args.mkdocs_file or Path("mkdocs.yml"))).resolve()
    config = load_mkdocs_config(mkdocs_file)
    staged_docs_dir = (mkdocs_file.parent / config.get("docs_dir", "docs")).resolve()
    if staged_docs_dir != output_dir:
        raise ValueError(
            f"Full mode expects mkdocs docs_dir to match the output dir: "
            f"{staged_docs_dir} != {output_dir}"
        )

    copied_docs = _stage_full_docs(
        repo_root=repo_root,
        docs_root=docs_root,
        output_dir=output_dir,
        manifest=manifest,
        overrides=overrides,
    )
    print(
        f"Staged {copied_docs} markdown pages into {output_dir.relative_to(repo_root)} "
        f"for {mkdocs_file.relative_to(repo_root)}."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
