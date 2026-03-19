# Installation

## Core Package

```bash
pip install truthound
```

This installs the default CLI and the Polars-based validation kernel.

## Optional Extras

```bash
pip install truthound[reports]
pip install truthound[drift]
pip install truthound[anomaly]
pip install truthound[stores]
pip install truthound[streaming]
pip install truthound[docs]
pip install truthound[dev]
```

## Verify the Installation

```bash
truthound --version
python -c "import truthound as th; print(th.__version__)"
```

## Development Setup

```bash
uv sync --extra dev --extra docs
uv run python -m pytest -q
uv run python docs/scripts/check_links.py --mkdocs mkdocs.yml README.md CLAUDE.md
uv run mkdocs build --strict
```

## Next Step

Continue with the [Quick Start](quickstart.md).
