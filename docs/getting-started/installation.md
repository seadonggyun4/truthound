# Installation

Install Truthound with the smallest dependency set that matches how you plan to use it. The core package is enough for zero-config validation and the default CLI. Add extras only when you need reporting, statistics-heavy workflows, streaming, or contributor tooling.

## Prerequisites

- Python `3.11` or newer
- `pip` or `uv`
- a local environment where Polars-based data processing is allowed

## Recommended Install Matrix

```bash
pip install truthound
```

Use the core package when you want:

- the default CLI
- `th.check()`, `th.profile()`, `th.mask()`, `th.scan()`, and core result types
- zero-config local validation with `.truthound/`

Common extras:

| Use case | Command |
|----------|---------|
| HTML report generation | `pip install truthound[reports]` |
| Drift detection | `pip install truthound[drift]` |
| ML anomaly detection | `pip install truthound[anomaly]` |
| Storage backends | `pip install truthound[stores]` |
| Streaming validation | `pip install truthound[streaming]` |
| Contributor docs + checks | `pip install truthound[dev,docs]` |
| Broad local feature coverage | `pip install truthound[all]` |

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

## Verify The Installation

```bash
truthound --version
python -c "import truthound as th; print(th.__version__)"
```

If both commands succeed, move to [Quick Start](quickstart.md).

## Development Setup

```bash
uv sync --extra dev --extra docs
uv run python -m pytest -q
uv run python docs/scripts/check_links.py --mkdocs mkdocs.yml README.md CLAUDE.md
uv run mkdocs build --strict
```

## Troubleshooting

### Import or optional dependency errors

If you see errors involving `jinja2`, `scipy`, `pyarrow`, or other optional packages, install the matching extra instead of the full `all` bundle by default.

### CLI available but a feature is missing

The CLI is installed with the core package, but some feature families still depend on extras. For example:

- report generation needs `truthound[reports]`
- streaming workflows need `truthound[streaming]`
- some statistical and ML workflows need `truthound[drift]` or `truthound[anomaly]`

### Unsure which install you need

Start with `truthound`, run the workflow you care about, and only then add the smallest extra required by the error message or guide.

## Next Step

Continue with the [Quick Start](quickstart.md), then run [First Validation](first-validation.md).
