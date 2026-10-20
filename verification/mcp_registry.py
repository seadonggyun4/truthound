"""Validate Registry metadata and render its exact UV argv; never authenticate/publish."""

from __future__ import annotations

import argparse
import json
import tomllib
from pathlib import Path

from jsonschema import Draft7Validator, FormatChecker

ROOT = Path(__file__).resolve().parents[1]
NAME = "io.github.seadonggyun4/truthound"
SCHEMA = "https://static.modelcontextprotocol.io/schemas/2025-12-11/server.schema.json"


def validate_metadata(document: dict, version: str, readmes: list[str]) -> None:
    schema = json.loads(
        (ROOT / "verification/schemas/mcp-server-2025-12-11.schema.json").read_text()
    )
    Draft7Validator.check_schema(schema)
    Draft7Validator(schema, format_checker=FormatChecker()).validate(document)
    if document["$schema"] != SCHEMA or document["name"] != NAME or document["version"] != version:
        raise ValueError("registry_identity_mismatch")
    if any(f"mcp-name: {NAME}" not in text for text in readmes):
        raise ValueError("registry_readme_ownership_missing")
    if document.get("remotes") or len(document["packages"]) != 1:
        raise ValueError("unexpected_registry_target")
    package = document["packages"][0]
    if (
        package["registryType"],
        package["registryBaseUrl"],
        package["identifier"],
        package["version"],
        package["runtimeHint"],
        package["transport"],
    ) != ("pypi", "https://pypi.org", "truthound", version, "uvx", {"type": "stdio"}):
        raise ValueError("registry_package_mismatch")
    if package["runtimeArguments"] != [
        {"type": "named", "name": "--with", "value": f"truthound[mcp]=={version}"}
    ]:
        raise ValueError("registry_extra_not_installed")
    args = package["packageArguments"]
    if (
        args[:2]
        != [{"type": "positional", "value": "mcp"}, {"type": "positional", "value": "serve"}]
        or len(args) != 3
    ):
        raise ValueError("registry_not_mcp_server_command")
    if (
        args[2].get("name") != "--allow-dir"
        or args[2].get("isRequired") is not True
        or args[2].get("format") != "filepath"
        or any(k in args[2] for k in ("value", "default"))
    ):
        raise ValueError("registry_implicit_file_access")


def runtime_argv(document: dict, directory: Path) -> list[str]:
    if not directory.is_absolute():
        raise ValueError("absolute_directory_required")
    package = document["packages"][0]
    return [
        "uvx",
        "--with",
        package["runtimeArguments"][0]["value"],
        f"{package['identifier']}@{package['version']}",
        "mcp",
        "serve",
        "--allow-dir",
        str(directory),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--print-command", type=Path)
    args = parser.parse_args()
    document = json.loads((ROOT / "server.json").read_text())
    version = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]
    validate_metadata(
        document, version, [(ROOT / name).read_text() for name in ("README.md", "README.en.md")]
    )
    print(
        json.dumps(
            dict(
                name=NAME,
                version=version,
                schema_valid=True,
                publication_verified=False,
                argv=runtime_argv(document, args.print_command) if args.print_command else None,
            )
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
