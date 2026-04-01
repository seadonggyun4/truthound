import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / 'src' / 'truthound' / 'core'
FORBIDDEN = (
    'truthound.reporters',
    'truthound.plugins',
    'truthound.datadocs',
    'truthound.cli_modules',
    'truthound.ai',
)


def test_core_modules_do_not_depend_on_outer_layers():
    violations: list[str] = []

    for path in ROOT.glob('*.py'):
        tree = ast.parse(path.read_text(encoding='utf-8'))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith(FORBIDDEN):
                        violations.append(f'{path.name}:{alias.name}')
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ''
                if module.startswith(FORBIDDEN):
                    violations.append(f'{path.name}:{module}')

    assert violations == []
