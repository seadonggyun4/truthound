"""Signed automatic configuration retention delegates selection to native Core."""
import json
import time

import pytest

from tests.test_mcp_checkpoint_services import runtime,prepare,command,approve,invoke
from truthound.mcp.approval import ApprovalStore

pytestmark=pytest.mark.contract


def test_approved_policy_survives_reopen_and_enforces_count_on_save(runtime,monkeypatch):
    adapter,control,_,_,_=runtime
    identity=prepare(runtime,monkeypatch,'runner')
    receipts=[command(runtime,identity,'persist_configuration',checkpoint_names=['approved-checkpoint'],key=f'retention-save-{n}') for n in range(3)]
    request=dict(binding_id='service',service_id=identity,command='configuration_retention_set',
        retention={'max_count':2,'expires_at':time.time()+3600},idempotency_key='retention-configure')
    pending=invoke(adapter,'checkpoint_services_execute',**request)
    assert pending['issues']==[{'code':'approval_required'}]
    store=ApprovalStore(control)
    try:assert store.connection.execute('SELECT COUNT(*) FROM checkpoint_service_snapshots').fetchone()[0]==3
    finally:store.close()
    approve(adapter,control,'checkpoint_services_execute',**request)
    saved=command(runtime,identity,'persist_configuration',checkpoint_names=['approved-checkpoint'],key='retention-new-save')
    rows=command(runtime,identity,'configuration_list')['snapshots']
    assert len(rows)==2 and saved['retained']
    assert {r['snapshot_id'] for r in rows}=={receipts[-1]['snapshot_id'],saved['snapshot_id']}
    # Reopen the private SQLite policy store independently, preserving signatures.
    adapter._session._approvals.close();adapter._session._approvals=None
    state=command(runtime,identity,'configuration_retention_status')
    assert state['active'] and state['policy']['max_count']==2
    newer=command(runtime,identity,'persist_configuration',checkpoint_names=['approved-checkpoint'],key='retention-after-reopen')
    assert newer['retained']
    assert len(command(runtime,identity,'configuration_list',key='list-after-reopen')['snapshots'])==2


@pytest.mark.parametrize('change',['signature','policy','metadata'])
def test_tampered_policy_never_deletes_snapshots(runtime,monkeypatch,change):
    adapter,control,_,_,_=runtime
    identity=prepare(runtime,monkeypatch,'runner')
    command(runtime,identity,'persist_configuration',checkpoint_names=['approved-checkpoint'])
    command(runtime,identity,'configuration_retention_set',retention={'max_count':1,'expires_at':time.time()+3600})
    store=ApprovalStore(control)
    try:
        store.connection.execute(f'UPDATE checkpoint_snapshot_retention SET {change}=?',('corrupt',))
        store.connection.commit()
    finally:store.close()
    request=dict(binding_id='service',service_id=identity,command='configuration_retention_apply',idempotency_key='tampered-cleanup')
    pending=invoke(adapter,'checkpoint_services_execute',**request)
    store=ApprovalStore(control)
    try:store.approve(pending['data']['plan_id'])
    finally:store.close()
    result=invoke(adapter,'checkpoint_services_execute',**request)
    assert result['execution_status']!='succeeded'
    store=ApprovalStore(control)
    try:assert store.connection.execute('SELECT COUNT(*) FROM checkpoint_service_snapshots').fetchone()[0]==1
    finally:store.close()


@pytest.mark.parametrize('mode',['all','any'])
def test_age_count_and_size_selection_matches_core_and_preserves_legacy(runtime,monkeypatch,mode):
    from datetime import datetime
    from truthound.stores.retention.base import ItemMetadata,PolicyMode
    from truthound.stores.retention.policies import CompositePolicy,TimeBasedPolicy,CountBasedPolicy,SizeBasedPolicy
    from truthound.mcp.approval import canonical
    adapter,control,_,_,_=runtime
    identity=prepare(runtime,monkeypatch,'runner')
    for n in range(4):command(runtime,identity,'persist_configuration',checkpoint_names=['approved-checkpoint'],key=f'age-source-{n}')
    store=ApprovalStore(control)
    native_items=[]
    try:
        rows=store.connection.execute('SELECT * FROM checkpoint_service_snapshots ORDER BY identity').fetchall()
        for index,row in enumerate(rows):
            metadata=json.loads(row['metadata'])
            if index==0:metadata.pop('created_at')
            else:metadata['created_at']=time.time()-{1:3600,2:30,3:1}[index]
            signature=store.digest({'identity':row['identity'],'metadata':metadata})
            store.connection.execute('UPDATE checkpoint_service_snapshots SET metadata=?,signature=? WHERE identity=?',
                (canonical(metadata),signature,row['identity']))
            if index:native_items.append(ItemMetadata(row['identity'],'service',datetime.fromtimestamp(metadata['created_at']),len(row['payload'])))
        store.connection.commit()
    finally:store.close()
    budget=sum(item.size_bytes for item in native_items)//2
    policy=CompositePolicy([TimeBasedPolicy(max_age_minutes=1),CountBasedPolicy(1),SizeBasedPolicy(max_size_bytes=budget)],mode=PolicyMode(mode))
    policy.prepare_batch(native_items)
    expected={item.item_id for item in native_items if not policy.should_retain(item)}
    command(runtime,identity,'configuration_retention_set',retention={'max_age_minutes':1,'max_count':1,'max_size_bytes':budget,'mode':mode,'expires_at':time.time()+3600})
    actual=command(runtime,identity,'configuration_retention_apply')
    assert set(actual['deleted'])==expected
    assert actual['excluded']==[rows[0]['identity']]
    remaining=command(runtime,identity,'configuration_list')['snapshots']
    assert {row['snapshot_id'] for row in remaining}=={row['identity'] for row in rows}-expected


def test_expired_policy_does_not_cleanup_and_budget_failure_rolls_back(runtime,monkeypatch,tmp_path):
    from truthound.mcp.durable_checkpoint_services import manage,sweep
    from truthound.mcp.checkpoint_services import CheckpointServiceCommand
    from truthound.mcp.policy import MCPError
    adapter,control,_,_,_=runtime
    identity=prepare(runtime,monkeypatch,'runner')
    for n in range(2):command(runtime,identity,'persist_configuration',checkpoint_names=['approved-checkpoint'],key=f'budget-save-{n}')
    command(runtime,identity,'configuration_retention_set',retention={'max_count':1,'expires_at':time.time()+60})
    metadata=adapter._session.resources.get(identity,'monitor').metadata
    store=adapter._session.approvals();old_clock=store.clock
    store.clock=lambda:old_clock()+3600
    try:
        with store.transaction():assert not sweep(adapter._session,metadata)['active']
        assert store.connection.execute('SELECT COUNT(*) FROM checkpoint_service_snapshots').fetchone()[0]==2
    finally:store.clock=old_clock
    args=CheckpointServiceCommand.model_validate(dict(binding_id='service',service_id=identity,
        command='configuration_retention_apply',idempotency_key='budget-retention'))
    with pytest.raises(MCPError,match='resource_exhausted'):
        manage(adapter._session,metadata,args,tmp_path/'bounded',1)
    assert store.connection.execute('SELECT COUNT(*) FROM checkpoint_service_snapshots').fetchone()[0]==2


def test_size_policy_can_remove_new_snapshot_without_claiming_it_is_restorable(runtime,monkeypatch):
    identity=prepare(runtime,monkeypatch,'runner')
    command(runtime,identity,'configuration_retention_set',retention={'max_size_bytes':1,'expires_at':time.time()+3600})
    saved=command(runtime,identity,'persist_configuration',checkpoint_names=['approved-checkpoint'])
    assert saved['retained'] is False
    assert command(runtime,identity,'configuration_list')['total']==0
