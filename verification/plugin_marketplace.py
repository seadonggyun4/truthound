"""Create an allowlisted, temporary client-installation fixture for product CI."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from core_plugin import FILES as CORE_FILES
from openai_plugin import FILES as LOCAL_FILES

ROOT = Path(__file__).resolve().parents[1]


def build_marketplace(output: Path, source_root: Path = ROOT / "plugins") -> Path:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    entries = []
    for name, files in (
        ("truthound", LOCAL_FILES),
        ("truthound-core", CORE_FILES),
    ):
        source = source_root / name
        for relative in files:
            path = source / relative
            if (
                path.is_symlink()
                or not path.is_file()
                or not path.resolve().is_relative_to(source.resolve())
            ):
                raise ValueError("Unsafe plugin fixture member")
            target = output / "plugins" / name / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
        entries.append(
            {
                "name": name,
                "source": {"source": "local", "path": "./plugins/" + name},
                "policy": {"installation": "AVAILABLE", "authentication": "ON_INSTALL"},
                "category": "Developer Tools",
            }
        )
    manifest = output / ".agents/plugins/marketplace.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        json.dumps({"name": "truthound-development", "plugins": entries}, indent=2) + "\n",
        encoding="utf-8",
    )
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(build_marketplace(args.output))
