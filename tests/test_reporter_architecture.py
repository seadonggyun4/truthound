import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1] / "src" / "truthound"
TARGETS = (
    ROOT / "reporters",
    ROOT / "datadocs",
    ROOT / "plugins",
    ROOT / "html_reporter.py",
)
FORBIDDEN = "truthound.stores.results"
ALLOWED = {
    ROOT / "reporters" / "adapters.py",
}


def test_reporting_layers_do_not_import_storage_dto_directly():
    violations: list[str] = []

    for target in TARGETS:
        paths = [target] if target.is_file() else list(target.rglob("*.py"))
        for path in paths:
            if path in ALLOWED:
                continue

            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(FORBIDDEN):
                    violations.append(str(path.relative_to(ROOT.parent)))
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name.startswith(FORBIDDEN):
                            violations.append(str(path.relative_to(ROOT.parent)))

    assert violations == []
