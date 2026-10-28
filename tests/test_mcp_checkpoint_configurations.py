"""Approved snapshot discovery and cleanup across the native service lifecycle."""
import json

import pytest

from tests.test_mcp_checkpoint_services import approve, command, invoke, prepare, runtime
from truthound.mcp.approval import ApprovalStore

pytestmark = pytest.mark.contract


def test_snapshot_discovery_approved_delete_replay_and_surviving_restore(runtime, monkeypatch):
    adapter, control, marker, _, _ = runtime
    identity = prepare(runtime, monkeypatch, 'runner')
    receipts = [command(runtime, identity, 'persist_configuration',
                        checkpoint_names=['approved-checkpoint'], key=f'snapshot-create-{n}')
                for n in range(2)]
    first = command(runtime, identity, 'configuration_list', configuration_limit=1)
    assert first['total'] == 2 and len(first['snapshots']) == 1
    second = command(runtime, identity, 'configuration_list', configuration_limit=1,
                     configuration_offset=first['next_offset'], key='snapshots-page-two')
    listed = first['snapshots'] + second['snapshots']
    assert {item['snapshot_id'] for item in listed} == {r['snapshot_id'] for r in receipts}
    assert all(set(item) == {'snapshot_id', 'sha256', 'bytes', 'service_kind'} for item in listed)
    request = dict(binding_id='service', service_id=identity, command='configuration_delete',
                   snapshot_ids=[receipts[0]['snapshot_id']], idempotency_key='snapshot-delete-one')
    before = marker.read_bytes()
    pending = invoke(adapter, 'checkpoint_services_execute', **request)
    assert pending['issues'] == [{'code': 'approval_required'}]
    assert pending['data']['effects'] == ['persistent_configuration_delete']
    store = ApprovalStore(control)
    try:
        assert store.connection.execute('SELECT COUNT(*) FROM checkpoint_service_snapshots').fetchone()[0] == 2
    finally:
        store.close()
    result = approve(adapter, control, 'checkpoint_services_execute', **request)
    assert invoke(adapter, 'checkpoint_services_execute', **request) == result
    assert marker.read_bytes() == before
    remaining = command(runtime, identity, 'configuration_list', key='remaining-snapshots')
    assert remaining['total'] == 1 and remaining['snapshots'][0]['snapshot_id'] == receipts[1]['snapshot_id']
    missing = invoke(adapter, 'checkpoint_services_open', kind='runner', binding_id='service',
                     snapshot_id=receipts[0]['snapshot_id'], idempotency_key='deleted-snapshot-restore')
    assert missing['issues'] == [{'code': 'invalid_handle'}]
    restored = approve(adapter, control, 'checkpoint_services_open', kind='runner', binding_id='service',
                       snapshot_id=receipts[1]['snapshot_id'], idempotency_key='remaining-snapshot-restore')
    assert restored['execution_status'] == 'succeeded'


@pytest.mark.parametrize('mutation', ['missing', 'signature', 'payload', 'binding', 'kind'])
def test_snapshot_cleanup_rejects_invalid_batch_without_deleting_valid_record(runtime, monkeypatch, mutation):
    adapter, control, _, _, _ = runtime
    identity = prepare(runtime, monkeypatch, 'runner')
    receipts = [command(runtime, identity, 'persist_configuration',
                        checkpoint_names=['approved-checkpoint'], key=f'snapshot-atomic-{n}')
                for n in range(2)]
    store = ApprovalStore(control)
    second = receipts[1]['snapshot_id']
    try:
        if mutation == 'missing':
            second = 'servicesnapshot_' + '0' * 64
        elif mutation == 'signature':
            store.connection.execute('UPDATE checkpoint_service_snapshots SET signature=? WHERE identity=?', ('bad', second))
        elif mutation == 'payload':
            store.connection.execute('UPDATE checkpoint_service_snapshots SET payload=? WHERE identity=?', (b'{}', second))
        else:
            row = store.connection.execute('SELECT * FROM checkpoint_service_snapshots WHERE identity=?', (second,)).fetchone()
            metadata = json.loads(row['metadata'])
            metadata['checkpoint_binding' if mutation == 'binding' else 'service_kind'] = 'another' if mutation == 'binding' else 'escalation'
            signature = store.digest({'identity': second, 'metadata': metadata})
            store.connection.execute('UPDATE checkpoint_service_snapshots SET metadata=?,signature=? WHERE identity=?',
                                     (json.dumps(metadata), signature, second))
        store.connection.commit()
    finally:
        store.close()
    request = dict(binding_id='service', service_id=identity, command='configuration_delete',
                   snapshot_ids=[receipts[0]['snapshot_id'], second], idempotency_key='snapshot-invalid-batch')
    pending = invoke(adapter, 'checkpoint_services_execute', **request)
    assert pending['issues'] == [{'code': 'approval_required'}]
    assert pending['data']['effects'] == ['persistent_configuration_delete']
    store = ApprovalStore(control)
    try:
        store.approve(pending['data']['plan_id'])
    finally:
        store.close()
    failed = invoke(adapter, 'checkpoint_services_execute', **request)
    assert failed['execution_status'] != 'succeeded'
    store = ApprovalStore(control)
    try:
        assert store.connection.execute('SELECT COUNT(*) FROM checkpoint_service_snapshots WHERE identity=?',
                                        (receipts[0]['snapshot_id'],)).fetchone()[0] == 1
    finally:
        store.close()


def test_approved_cleanup_reclaims_the_real_snapshot_count_limit(runtime, monkeypatch):
    from truthound.mcp.durable_checkpoint_services import persist
    from truthound.mcp.policy import MCPError

    adapter, control, _, _, _ = runtime
    identity = prepare(runtime, monkeypatch, 'runner')
    receipt = command(runtime, identity, 'persist_configuration',
                      checkpoint_names=['approved-checkpoint'])
    store = ApprovalStore(control)
    try:
        row = store.connection.execute('SELECT * FROM checkpoint_service_snapshots').fetchone()
        metadata, payload = json.loads(row['metadata']), row['payload']
        for index in range(127):
            clone = f'servicesnapshot_{index + 4096:064x}'
            store.connection.execute('INSERT INTO checkpoint_service_snapshots VALUES (?,?,?,?)',
                (clone, row['metadata'], payload, store.digest({'identity': clone, 'metadata': metadata})))
        store.connection.commit()
    finally:
        store.close()
    with pytest.raises(MCPError, match='resource_exhausted'):
        persist(adapter._session, {'plan_id': 'capacity-before-cleanup'}, metadata, payload)
    result = command(runtime, identity, 'configuration_delete', snapshot_ids=[receipt['snapshot_id']])
    assert result['deleted'] == 1
    created = persist(adapter._session, {'plan_id': 'capacity-after-cleanup'}, metadata, payload)
    assert created['snapshot_id'] != receipt['snapshot_id']
    store = ApprovalStore(control)
    try:
        assert store.connection.execute('SELECT COUNT(*) FROM checkpoint_service_snapshots').fetchone()[0] == 128
    finally:
        store.close()


@pytest.mark.parametrize('arguments', [
    {'command': 'configuration_delete'},
    {'command': 'configuration_delete', 'snapshot_ids': []},
    {'command': 'configuration_delete', 'snapshot_ids': ['servicesnapshot_' + 'a'*64]*2},
    {'command': 'configuration_list', 'snapshot_ids': ['servicesnapshot_' + 'a'*64]},
    {'command': 'status', 'configuration_limit': 1},
    {'command': 'configuration_list', 'configuration_limit': 0},
])
def test_snapshot_command_rejects_ambiguous_or_unbounded_selection(arguments):
    from pydantic import ValidationError
    from truthound.mcp.checkpoint_services import CheckpointServiceCommand

    with pytest.raises(ValidationError):
        CheckpointServiceCommand.model_validate(dict(binding_id='service', service_id='monitor_'+'a'*32,
            idempotency_key='snapshot-invalid-shape', **arguments))


def test_snapshot_delete_checks_receipt_budget_before_removing_any_row(runtime, monkeypatch, tmp_path):
    from truthound.mcp.checkpoint_services import CheckpointServiceCommand
    from truthound.mcp.durable_checkpoint_services import manage
    from truthound.mcp.policy import MCPError

    adapter, control, _, _, _ = runtime
    identity = prepare(runtime, monkeypatch, 'runner')
    receipt = command(runtime, identity, 'persist_configuration', checkpoint_names=['approved-checkpoint'])
    arguments = CheckpointServiceCommand.model_validate(dict(binding_id='service', service_id=identity,
        command='configuration_delete', snapshot_ids=[receipt['snapshot_id']], idempotency_key='budget-delete'))
    metadata = adapter._session.resources.get(identity, 'monitor').metadata
    with pytest.raises(MCPError, match='resource_exhausted'):
        manage(adapter._session, metadata, arguments, tmp_path/'bounded-receipt', 1)
    store = ApprovalStore(control)
    try:
        assert store.connection.execute('SELECT COUNT(*) FROM checkpoint_service_snapshots').fetchone()[0] == 1
    finally:
        store.close()


def test_worker_bootstrap_failure_does_not_consume_approval_or_claim_native_effects(runtime, monkeypatch):
    from truthound.mcp import checkpoint_services

    adapter, control, marker, _, _ = runtime
    request = prepare(runtime, monkeypatch, 'runner', approved=False)
    pending = invoke(adapter, 'checkpoint_services_open', **request)
    plan_id = pending['data']['plan_id']
    store = ApprovalStore(control)
    try:
        store.approve(plan_id)
    finally:
        store.close()
    created = []

    class UnreadyWorker:
        def __init__(self, ttl):
            self.closed = False
            created.append(self)
        def watch_policy(self, callback):
            assert callback()
        async def call(self, payload, timeout):
            assert payload == {'prepare': True}
            raise TimeoutError
        def close(self):
            self.closed = True

    monkeypatch.setattr(checkpoint_services, 'ServiceProcess', UnreadyWorker)
    for _ in range(2):
        result = invoke(adapter, 'checkpoint_services_open', **request)
        assert result['execution_status'] != 'outcome_unknown'
        assert result['issues'] == [{'code': 'timeout'}]
        assert not marker.exists()
        assert created[-1].closed
        assert adapter._session.approvals().status(plan_id, session=adapter._session.identity)['state'] == 'approved'
    assert len(created) == 2


def test_worker_readiness_and_factory_share_deadline_and_uncertain_factory_is_not_replayed(runtime, monkeypatch):
    from types import SimpleNamespace
    from truthound.mcp import checkpoint_services

    adapter, control, _, _, _ = runtime
    request = prepare(runtime, monkeypatch, 'runner', approved=False)
    pending = invoke(adapter, 'checkpoint_services_open', **request)
    store = ApprovalStore(control)
    try:
        store.approve(pending['data']['plan_id'])
    finally:
        store.close()
    calls = []
    clock = iter([100., 101., 102., 107.])
    monkeypatch.setattr(checkpoint_services, 'time', SimpleNamespace(monotonic=lambda: next(clock)))

    class ReadyThenUncertain:
        def __init__(self, ttl):
            self.closed = False
        def watch_policy(self, callback):
            assert callback()
        async def call(self, payload, timeout):
            calls.append((payload, timeout))
            if payload == {'prepare': True}:
                return {'ok': True}
            raise TimeoutError
        def close(self):
            self.closed = True

    monkeypatch.setattr(checkpoint_services, 'ServiceProcess', ReadyThenUncertain)
    first = invoke(adapter, 'checkpoint_services_open', **request)
    assert first['issues'] == [{'code': 'outcome_unknown'}]
    assert len(calls) == 2
    assert calls[0][1] == adapter.policy.timeout_seconds - 1
    assert calls[1][1] == adapter.policy.timeout_seconds - 7
    replay = invoke(adapter, 'checkpoint_services_open', **request)
    assert replay['issues'] == first['issues'] and len(calls) == 2
