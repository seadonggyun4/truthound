from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import urlparse

import yaml


def _load_manifest(path: Path) -> dict:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Manifest must be a mapping: {path}")
    return data


def _doc_to_public_path(doc_path: str) -> str:
    path = Path(doc_path)
    if path.name == "index.md":
        if path.parent == Path("."):
            return "/"
        return f"/{path.parent.as_posix()}/"
    return f"/{(path.parent / path.stem).as_posix()}/"


def _normalize_location(location: str) -> str:
    parsed = urlparse(location)
    path = parsed.path or "/"
    if not path.startswith("/"):
        path = f"/{path}"
    if path != "/" and not path.endswith("/"):
        path = f"{path}/"
    return path


def _base_location(location: str) -> str:
    return _normalize_location(location.split("#", 1)[0])


def _read_sitemap_paths(site_dir: Path) -> list[str]:
    sitemap = site_dir / "sitemap.xml"
    tree = ET.parse(sitemap)
    root = tree.getroot()
    namespace = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
    return [_normalize_location(node.text or "") for node in root.findall("sm:url/sm:loc", namespace)]


def _read_search_paths(site_dir: Path) -> list[str]:
    search_index = site_dir / "search" / "search_index.json"
    data = json.loads(search_index.read_text(encoding="utf-8"))
    return [
        _base_location(entry.get("location", ""))
        for entry in data.get("docs", [])
        if isinstance(entry, dict) and entry.get("location")
    ]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify that the generated MkDocs site only exposes the strict public docs surface."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("docs/public_docs.yml"),
        help="Path to the public docs manifest.",
    )
    parser.add_argument(
        "--site-dir",
        type=Path,
        default=Path("site"),
        help="Path to the generated site directory.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    manifest = _load_manifest((repo_root / args.manifest).resolve())
    site_dir = (repo_root / args.site_dir).resolve()

    allowlisted_paths = {_doc_to_public_path(doc_path) for doc_path in manifest.get("docs", [])}
    excluded_prefixes = tuple(f"/{prefix}" for prefix in manifest.get("excluded_prefixes", []))
    expected_page_count = int(manifest.get("expected_page_count", len(allowlisted_paths)))

    sitemap_paths = _read_sitemap_paths(site_dir)
    search_paths = _read_search_paths(site_dir)

    failures: list[str] = []

    unexpected_sitemap = sorted({path for path in sitemap_paths if path not in allowlisted_paths})
    if unexpected_sitemap:
        failures.append(
            "Unexpected sitemap paths: " + ", ".join(unexpected_sitemap[:10])
        )

    unexpected_search = sorted({path for path in search_paths if path not in allowlisted_paths})
    if unexpected_search:
        failures.append(
            "Unexpected search index paths: " + ", ".join(unexpected_search[:10])
        )

    leaked_prefixes = sorted(
        {
            path
            for path in set(sitemap_paths + search_paths)
            if any(path.startswith(prefix) for prefix in excluded_prefixes)
        }
    )
    if leaked_prefixes:
        failures.append(
            "Excluded families leaked into the public site: " + ", ".join(leaked_prefixes[:10])
        )

    sitemap_unique = set(sitemap_paths)
    search_unique = set(search_paths)
    if len(sitemap_unique) != expected_page_count:
        failures.append(
            f"Expected {expected_page_count} sitemap pages, found {len(sitemap_unique)}."
        )
    if len(search_unique) != expected_page_count:
        failures.append(
            f"Expected {expected_page_count} search-index pages, found {len(search_unique)}."
        )

    if failures:
        print("Public docs surface verification failed:")
        for failure in failures:
            print(f" - {failure}")
        return 1

    print(
        "Verified strict public docs surface: "
        f"{len(sitemap_unique)} sitemap pages and {len(search_unique)} search-index pages."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
