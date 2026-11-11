"""Persistence must not reassign evidence or accept missing verdict fields."""
import json

import polars as pl
import pytest

import truthound as th
from truthound.context import TruthoundContext
from truthound.core.results import ValidationRunResult
from truthound.stores.backends.filesystem import FileSystemStore
from truthound.stores.results import ValidationResult
from truthound.types import Severity
from truthound.validators import get_validator

pytestmark = [pytest.mark.contract, pytest.mark.fault]


def make_run(tmp_path):
    between = get_validator('between')
    return th.check(pl.DataFrame({'x': [1, 50], 'y': [1, 2]}), validators=[
        between(columns=['x'], min_value=0, max_value=5),
        between(columns=['x'], min_value=0, max_value=100),
        between(columns=['y'], min_value=0, max_value=5),
    ], context=TruthoundContext(tmp_path))


def test_same_name_checks_keep_exact_evidence_after_file_reopen(tmp_path):
    run = make_run(tmp_path)
    store_path = str(tmp_path / 'store')
    key = FileSystemStore(base_path=store_path).save(ValidationResult.from_validation_run_result(run))
    restored = FileSystemStore(base_path=store_path).get(key).to_validation_run_result()
    assert [c.to_dict() for c in restored.checks] == [c.to_dict() for c in run.checks]
    assert [i.to_dict() for i in restored.issues] == [i.to_dict() for i in run.issues]
    assert [c.success for c in restored.filter_by_severity(Severity.LOW).checks] == [False, True, True]


@pytest.mark.parametrize('field', ['checks', 'issues', 'execution_issues'])
def test_checkpoint_rejects_missing_canonical_evidence(tmp_path, field):
    from truthound.checkpoint.adapters import validation_run_from_checkpoint_dict
    payload = make_run(tmp_path).to_dict()
    del payload[field]
    with pytest.raises(ValueError, match='artifact'):
        validation_run_from_checkpoint_dict({'validation_run': payload})


@pytest.mark.parametrize('payload', [{}, [], 'invalid'])
def test_checkpoint_rejects_invalid_nested_payload(payload):
    from truthound.checkpoint.adapters import validation_run_from_checkpoint_dict
    with pytest.raises(ValueError, match='artifact'):
        validation_run_from_checkpoint_dict({'validation_run': payload})


def test_ai_loader_rejects_missing_evidence_and_wrong_identity(tmp_path):
    from truthound.ai.analysis import RunInputResolver
    context = TruthoundContext(tmp_path)
    run = make_run(tmp_path)
    path = context.runs_dir / f'{run.run_id}.json'
    for payload in ({'run_id': run.run_id}, {**run.to_dict(), 'run_id': 'other'}):
        path.write_text(json.dumps(payload))
        with pytest.raises(ValueError, match='artifact'):
            RunInputResolver()._load_run(run.run_id, context)


def test_explicit_empty_run_and_permissive_constructor_remain_supported():
    from truthound.checkpoint.adapters import validation_run_from_checkpoint_dict
    run = ValidationRunResult(suite_name='empty', source='synthetic', row_count=0, column_count=0)
    restored = validation_run_from_checkpoint_dict({'validation_run': run.to_dict()})
    assert restored.to_dict() == run.to_dict()
    assert ValidationRunResult.from_dict({}).success
    assert validation_run_from_checkpoint_dict({'validation_run': None}) is None


def test_legacy_contract_rejects_ambiguous_names_but_keeps_unique_names(tmp_path):
    run = make_run(tmp_path)
    stored = ValidationResult.from_validation_run_result(run)
    contract = stored.metadata['_validation_run_contract']
    del contract['version']
    del contract['issues']
    for check in contract['checks']:
        del check['issues']
    with pytest.raises(ValueError, match='Ambiguous legacy'):
        stored.to_validation_run_result()
    contract['checks'] = contract['checks'][:1]
    restored = stored.to_validation_run_result()
    assert not restored.checks[0].success
    assert len(restored.checks[0].issues) == 1
    # Pre-contract DTOs still retain their row-level evidence.
    del stored.metadata['_validation_run_contract']
    assert not stored.to_validation_run_result().success


def test_complete_details_aliases_and_execution_failure_keep_ownership(tmp_path):
    from dataclasses import replace

    from truthound.core.results import CheckResult, ExecutionIssue
    from truthound.validators.base import ValidationIssue
    low = ValidationIssue(column='x', issue_type='low', count=1, severity=Severity.LOW,
                          validator_name='underlying')
    high = ValidationIssue(column='y', issue_type='high', count=1, severity=Severity.HIGH,
                           validator_name='underlying')
    run = replace(make_run(tmp_path), checks=(
        CheckResult(name='alias-a', success=False, issue_count=1, issues=(low,)),
        CheckResult(name='alias-b', success=False, issue_count=1, issues=(high,)),
        CheckResult(name='same', success=False, metadata={'_truthound_execution_incomplete': True}),
        CheckResult(name='same', success=True, metadata={'_truthound_execution_incomplete': False}),
    ), issues=(low, high), execution_issues=(ExecutionIssue(check_name='same', message='synthetic'),))
    wire = json.loads(json.dumps(ValidationResult.from_validation_run_result(run).to_dict()))
    restored = ValidationResult.from_dict(wire).to_validation_run_result()
    assert [c.to_dict() for c in restored.checks] == [c.to_dict() for c in run.checks]
    assert restored.execution_issues == run.execution_issues
    assert [c.success for c in restored.filter_by_severity(Severity.HIGH).checks] == [True, False, False, True]


@pytest.mark.parametrize('version', [0, 3, '2'])
def test_unknown_storage_contract_version_is_rejected(tmp_path, version):
    stored = ValidationResult.from_validation_run_result(make_run(tmp_path))
    stored.metadata['_validation_run_contract']['version'] = version
    with pytest.raises(ValueError, match='version'):
        stored.to_validation_run_result()


def test_artifact_rejects_invalid_json_and_missing_nested_outcome(tmp_path):
    from truthound.ai.analysis import RunInputResolver
    context = TruthoundContext(tmp_path)
    run = make_run(tmp_path)
    path = context.runs_dir / f'{run.run_id}.json'
    path.write_text('{')
    with pytest.raises(ValueError):
        RunInputResolver()._load_run(run.run_id, context)
    payload = run.to_dict()
    del payload['checks'][0]['success']
    with pytest.raises(ValueError, match='artifact'):
        ValidationRunResult.from_artifact_dict(payload)


def test_legacy_alias_cannot_reassign_evidence_to_successful_check(tmp_path):
    stored = ValidationResult.from_validation_run_result(make_run(tmp_path))
    contract = stored.metadata['_validation_run_contract']
    del contract['version']
    contract['checks'][0]['name'] = 'custom-display-name'
    # Only successful checks now share the issue's underlying name.
    with pytest.raises(ValueError, match='Ambiguous legacy'):
        stored.to_validation_run_result()
