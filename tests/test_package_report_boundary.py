"""Generated diagnostic output must not enter source distributions."""
import tomllib
from pathlib import Path


def test_sdist_excludes_generated_verification_output():
    root = Path(__file__).resolve().parents[1]
    config = tomllib.loads((root / "pyproject.toml").read_text())
    sdist = config["tool"]["hatch"]["build"]["targets"]["sdist"]
    assert "/verification/**" in sdist["exclude"]
    assert all(not key.startswith("verification/") for key in sdist.get("force-include", {}))
