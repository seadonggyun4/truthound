"""Record public commit/version and hashes of the deployment verification pages."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAGES = (
    "/guides/mcp/",
    "/en/guides/mcp/",
    "/guides/ci-testing/",
    "/en/guides/ci-testing/",
    "/guides/docs-deployment-verification/",
    "/en/guides/docs-deployment-verification/",
)


def write_manifest(site, commit, version):
    if not re.fullmatch(r"[0-9a-f]{40}", commit) or not re.fullmatch(
        r"[0-9]+\.[0-9]+\.[0-9]+(?:[a-z0-9.+-]+)?", version
    ):
        raise ValueError("Invalid public build identity")
    site = site.resolve(strict=True)
    hashes = {}
    for url in PAGES:
        page = site / url.strip("/") / "index.html"
        if page.is_symlink() or not page.resolve(strict=True).is_relative_to(site):
            raise ValueError("Invalid public page")
        hashes[url] = hashlib.sha256(page.read_bytes()).hexdigest()
    value = {
        "schema_version": 1,
        "source_commit": commit,
        "package_version": version,
        "page_sha256": hashes,
    }
    output = site / "deployment-manifest.json"
    if output.is_symlink():
        raise ValueError("Invalid manifest destination")
    output.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site-dir", type=Path, required=True)
    args = parser.parse_args()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    version = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"][
        "version"
    ]
    value = write_manifest(args.site_dir, commit, version)
    print(json.dumps({"source_commit": value["source_commit"], "public_pages": len(PAGES)}))


if __name__ == "__main__":
    main()
