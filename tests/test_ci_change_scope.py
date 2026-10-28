"""Check push/PR routing against real workflow filters, including mixed changes."""
from fnmatch import fnmatchcase
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.contract
ROOT = Path(__file__).resolve().parents[1]


def workflows():
    return {
        path.stem: yaml.load(path.read_text(), Loader=yaml.BaseLoader)
        for path in (ROOT / '.github/workflows').glob('*.yml')
    }


def selected(paths, event):
    result = set()
    for name, workflow in workflows().items():
        trigger = workflow['on'].get(event)
        if trigger is None:
            continue
        if event == 'push' and 'branches' in trigger:
            if not any(fnmatchcase('main', branch) for branch in trigger['branches']):
                continue
        patterns = trigger.get('paths', ['**'])
        for path in paths:
            matched = False
            for pattern in patterns:
                if fnmatchcase(path, pattern.removeprefix('!')):
                    matched = not pattern.startswith('!')
            if matched:
                result.add(name)
                break
    return result


@pytest.mark.parametrize('event', ['push', 'pull_request'])
@pytest.mark.parametrize('paths,expected', [
    (['README.md'], {'readme'}),
    (['README.en.md'], {'readme'}),
    (['docs/guides/mcp.md'], {'docs'}),
    (['docs/locales/ko/guides/ci-testing.md'], {'docs'}),
    (['src/truthound/mcp/server.py'], {'mcp-checks'}),
    (['tests/test_mcp_approval.py'], {'mcp-checks'}),
    (['plugins/truthound/.codex-plugin/plugin.json'], {'openai-plugin'}),
    (['server.json'], {'mcp-checks'}),
    (['.github/workflows/readme.yml'], {'readme', 'ci-policy'}),
    (['.github/workflows/tests-pr.yml'], {'ci-policy'}),
    (['.github/workflows/windows-native.yml'], {'ci-policy'}),
    (['tests/test_ci_change_scope.py'], {'ci-policy'}),
    (['src/truthound/__init__.py'], {'tests-pr', 'docs', 'mcp-checks', 'mcp-streaming', 'mcp-sql-isolation','mcp-oracle-isolation','mcp-sqlserver-isolation'}),
    (['src/truthound/core/runtime.py'], {'tests-pr', 'docs', 'mcp-checks'}),
    (['src/truthound/benchmark/runner.py'], {'tests-pr', 'docs', 'mcp-checks', 'benchmarks-pr'}),
    (['src/truthound/multitenancy/sources.py'], {'tests-pr','docs','mcp-checks','mcp-sql-isolation','mcp-oracle-isolation','mcp-sqlserver-isolation'}),
    (['src/truthound/mcp/bound_services.py'], {'mcp-checks','mcp-streaming','mcp-sql-isolation','mcp-oracle-isolation','mcp-sqlserver-isolation'}),
    (['src/truthound/mcp/checkpoint_worker.py'], {'mcp-checks','mcp-streaming','mcp-sql-isolation','mcp-oracle-isolation','mcp-sqlserver-isolation'}),
    (['src/truthound/datasources/sql/oracle.py'], {'tests-pr','docs','mcp-checks','mcp-oracle-isolation'}),
    (['tests/test_tenant_oracle_isolation.py'], {'mcp-oracle-isolation'}),
    (['tests/test_tenant_postgresql_isolation.py'], {'mcp-sql-isolation'}),
    (['tests/test_tenant_sqlserver_isolation.py'], {'mcp-sqlserver-isolation'}),
    (['tests/mcp_sql_client.py'], {'mcp-oracle-isolation','mcp-sqlserver-isolation'}),
    (['tests/test_tenant_mysql_isolation.py'], {'mcp-sql-isolation'}),
    (['src/truthound/mcp/stream_validation.py'], {'mcp-checks','mcp-streaming'}),
    (['src/truthound/validators/streaming/sources.py'], {'docs','mcp-checks','mcp-streaming'}),
    (['README.md', 'src/truthound/mcp/server.py'], {'readme', 'mcp-checks'}),
    (['verification/ci/requirements-lite.txt'], {'readme', 'mcp-checks', 'ci-policy', 'docs', 'mcp-streaming', 'mcp-sql-isolation','mcp-oracle-isolation','mcp-sqlserver-isolation'}),
    (['uv.lock'], {'tests-pr', 'docs', 'mcp-checks', 'benchmarks-pr', 'mcp-streaming', 'mcp-sql-isolation','mcp-oracle-isolation','mcp-sqlserver-isolation'}),
])
def test_component_changes_only_run_affected_workflows(event, paths, expected):
    assert selected(paths, event) == expected


def test_full_verification_and_publication_are_not_ordinary_push_gates():
    all_workflows = workflows()
    for name in ('mcp', 'tests-nightly', 'release-verify', 'release-pypi',
                 'benchmarks-nightly', 'windows-native', 'windows-hadoop',
                 'mcp-runtime-diagnostic', 'local-brokers', 'ai-live-smoke'):
        assert not {'push', 'pull_request', 'merge_group'} & set(all_workflows[name]['on'])
    assert 'schedule' in all_workflows['mcp']['on']
    assert all_workflows['release-verify']['jobs']['mcp']['with']['full_compatibility'] == 'true'
    assert all_workflows['release-verify']['jobs']['full']['with']['run_full_suite'] == 'true'


def test_small_lanes_do_not_install_distributed_development_sdks():
    import tomllib
    versions = {p['name']: p['version'] for p in tomllib.loads((ROOT / 'uv.lock').read_text())['package']}
    for filename in ('requirements-lite.txt', 'requirements-streaming.txt', 'requirements-sql-isolation.txt', 'requirements-oracle-isolation.txt'):
        for line in (ROOT / 'verification/ci' / filename).read_text().splitlines():
            if line and not line.startswith('#'):
                name, version = line.split('==')
                assert versions[name] == version
    for name in ('readme', 'mcp-checks', 'ci-policy', 'docs', 'mcp-streaming', 'mcp-sql-isolation', 'mcp-oracle-isolation'):
        jobs = workflows()[name]['jobs']
        for job in jobs.values():
            commands = '\n'.join(step.get('run', '') for step in job['steps'])
            assert '--extra dev' not in commands
            assert '[docs,dev]' not in commands
            assert 'requirements-lite.txt' in commands
