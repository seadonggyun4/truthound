import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1] / "src" / "truthound"
TARGETS = (
    ROOT / "checkpoint",
    ROOT / "profiler",
    ROOT / "realtime",
    ROOT / "ml",
    ROOT / "lineage",
)
FORBIDDEN = (
    "truthound.report",
    "truthound.stores.results",
    "truthound.reporters",
    "truthound.datadocs",
    "truthound.cli_modules",
)
ALLOWED = {
    ROOT / "checkpoint" / "adapters.py",
}


def test_peripheral_packages_depend_on_core_first_boundaries():
    violations: list[str] = []

    for target in TARGETS:
        for path in target.rglob("*.py"):
            if path in ALLOWED:
                continue

            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    if module.startswith(FORBIDDEN):
                        violations.append(str(path.relative_to(ROOT.parent)))
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name.startswith(FORBIDDEN):
                            violations.append(str(path.relative_to(ROOT.parent)))

    assert violations == []
