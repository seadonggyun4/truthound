from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

_FRONTMATTER_PATTERN = re.compile(r"\A---\n.*?\n---\n*", re.DOTALL)


@dataclass(frozen=True)
class ExternalSource:
    name: str
    prefix: str
    label: str
    repo_url: str
    repo_name: str
    branch: str
    docs_root: str

    @property
    def normalized_prefix(self) -> str:
        return self.prefix.strip("/")

    @property
    def normalized_docs_root(self) -> str:
        return self.docs_root.strip("/")


def sanitize_mkdocs_text(raw_text: str) -> str:
    return re.sub(r"!!python/name:[^\s]+", "python-ref", raw_text)


def load_mkdocs_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(sanitize_mkdocs_text(path.read_text(encoding="utf-8"))) or {}
    if not isinstance(config, dict):
        raise ValueError(f"MkDocs config must be a mapping: {path}")
    return config


def iter_nav_paths(node: Any) -> list[str]:
    paths: list[str] = []
    if isinstance(node, list):
        for item in node:
            paths.extend(iter_nav_paths(item))
    elif isinstance(node, dict):
        for value in node.values():
            paths.extend(iter_nav_paths(value))
    elif isinstance(node, str) and node.endswith(".md"):
        paths.append(node)
    return paths


def load_nav_doc_paths(mkdocs_file: Path) -> list[Path]:
    config = load_mkdocs_config(mkdocs_file)
    return [Path(value) for value in iter_nav_paths(config.get("nav", []))]


def load_external_sources(manifest: dict[str, Any]) -> list[ExternalSource]:
    raw_sources = manifest.get("external_sources", {})
    if not raw_sources:
        return []
    if not isinstance(raw_sources, dict):
        raise ValueError("external_sources must be a mapping")

    sources: list[ExternalSource] = []
    for name, raw_config in raw_sources.items():
        if not isinstance(raw_config, dict):
            raise ValueError(f"external_sources.{name} must be a mapping")
        sources.append(
            ExternalSource(
                name=name,
                prefix=str(raw_config["prefix"]),
                label=str(raw_config["label"]),
                repo_url=str(raw_config["repo_url"]).rstrip("/"),
                repo_name=str(raw_config["repo_name"]),
                branch=str(raw_config.get("branch", "main")),
                docs_root=str(raw_config.get("docs_root", "docs")),
            )
        )
    return sources


def discover_external_source_root(
    repo_root: Path,
    source: ExternalSource,
    overrides: dict[str, Path] | None = None,
) -> Path | None:
    override = (overrides or {}).get(source.name)
    candidates: list[Path] = []
    if override is not None:
        candidates.append(override)

    env_key = f"TRUTHOUND_EXTERNAL_SOURCE_{source.name.upper()}"
    env_value = os.environ.get(env_key)
    if env_value:
        candidates.append(Path(env_value))

    repo_basename = Path(source.repo_name).name
    candidates.extend(
        [
            repo_root.parent / repo_basename,
            repo_root / ".external" / repo_basename,
            repo_root / ".external" / source.name,
        ]
    )

    for candidate in candidates:
        resolved = candidate.resolve()
        if (resolved / source.normalized_docs_root).exists():
            return resolved
    return None


def external_nav_doc_paths(
    repo_root: Path,
    source: ExternalSource,
    overrides: dict[str, Path] | None = None,
) -> list[Path]:
    source_root = discover_external_source_root(repo_root, source, overrides)
    if source_root is None:
        raise FileNotFoundError(
            f"Unable to locate external source checkout for {source.name!r}. "
            f"Set TRUTHOUND_EXTERNAL_SOURCE_{source.name.upper()} or provide an override."
        )
    return load_nav_doc_paths(source_root / "mkdocs.yml")


def match_external_source(relative_path: Path, external_sources: list[ExternalSource]) -> ExternalSource | None:
    normalized_path = relative_path.as_posix().lstrip("/")
    for source in external_sources:
        prefix = source.normalized_prefix
        if normalized_path.startswith(f"{prefix}/"):
            return source
    return None


def upstream_doc_path(relative_path: Path, source: ExternalSource) -> str:
    normalized_path = relative_path.as_posix().lstrip("/")
    prefix = source.normalized_prefix
    if not normalized_path.startswith(f"{prefix}/"):
        raise ValueError(f"{relative_path} is not within the {source.name} source prefix")
    suffix = normalized_path[len(prefix) + 1 :]
    return f"{source.normalized_docs_root}/{suffix}"


def upstream_source_url(relative_path: Path, source: ExternalSource) -> str:
    return f"{source.repo_url}/blob/{source.branch}/{upstream_doc_path(relative_path, source)}"


def upstream_edit_url(relative_path: Path, source: ExternalSource) -> str:
    return f"{source.repo_url}/edit/{source.branch}/{upstream_doc_path(relative_path, source)}"


def build_source_banner(relative_path: Path, source: ExternalSource) -> str:
    upstream_path = upstream_doc_path(relative_path, source)
    source_url = upstream_source_url(relative_path, source)
    edit_url = upstream_edit_url(relative_path, source)
    return "\n".join(
        [
            '!!! note "Upstream Source"',
            f"    This page is part of {source.label}.",
            "",
            f"    Source repository: [{source.repo_name}]({source.repo_url})",
            f"    Upstream docs path: [`{upstream_path}`]({source_url})",
            f"    Edit upstream page: [Edit in {source.name}]({edit_url})",
        ]
    )


def inject_source_banner(markdown: str, banner: str) -> str:
    match = _FRONTMATTER_PATTERN.match(markdown)
    if match:
        head = markdown[: match.end()]
        tail = markdown[match.end() :].lstrip("\n")
        if tail:
            return f"{head}\n{banner}\n\n{tail}"
        return f"{head}\n{banner}\n"
    return f"{banner}\n\n{markdown.lstrip()}"
