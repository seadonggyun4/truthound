"""Real process boundaries preserve claims and pending reservations."""

import json
import os
import subprocess
import sys

import pytest


def child(database, program):
    environment = dict(os.environ)
    from pathlib import Path

    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[4] / "src")
    return subprocess.run(
        [sys.executable, "-c", program, str(database)],
        env=environment,
        capture_output=True,
        text=True,
        timeout=20,
        check=True,
    ).stdout


def test_restart_preserves_pending_and_quarantines_interrupted_attempt(tmp_path):
    database = tmp_path / "journal.sqlite"
    child(
        database,
        """
import os, sys
from truthound.checkpoint.escalation.journal import ExecutionJournal
j = ExecutionJournal(sys.argv[1])
j.enqueue('job:pending', {'target': 2})
j.enqueue('notification:interrupted', {'record_id': 'incident'})
assert j.claim('notification:interrupted')
os._exit(0)
""",
    )
    result = child(
        database,
        """
import json, sys
from truthound.checkpoint.escalation.journal import ExecutionJournal
j = ExecutionJournal(sys.argv[1])
assert not j.claim('notification:interrupted')
print(json.dumps({'pending': j.pending(), 'state': j.state('notification:interrupted')}))
j.close()
""",
    )
    assert json.loads(result) == {
        "pending": [["job:pending", {"target": 2}]],
        "state": "unknown",
    }


def test_concurrent_processes_cannot_execute_the_same_claim(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    database = tmp_path / "claims.sqlite"
    child(
        database,
        """
import sys
from truthound.checkpoint.escalation.journal import ExecutionJournal
j = ExecutionJournal(sys.argv[1])
j.enqueue('job:shared', {'record': 'shared'})
j.close()
""",
    )
    program = """
import sys
from truthound.checkpoint.escalation.journal import ExecutionJournal
j = ExecutionJournal(sys.argv[1])
print(int(j.claim('job:shared')))
j.close()
"""
    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(lambda _: child(database, program).strip(), range(2)))
    assert sorted(outcomes) == ["0", "1"]


@pytest.mark.parametrize("backend", ["asyncio", "apscheduler"])
def test_pending_native_scheduler_job_executes_once_after_process_restart(tmp_path, backend):
    database = tmp_path / "scheduler.sqlite"
    child(
        database,
        f"""
import asyncio, os, sys
from datetime import timedelta
from truthound.checkpoint.escalation.scheduler import create_scheduler, SchedulerConfig, SchedulerType
async def run():
    scheduler = create_scheduler(SchedulerConfig(scheduler_type=SchedulerType({backend!r}), journal_path=sys.argv[1]))
    scheduler.schedule_escalation('record', 'policy', 2, timedelta(seconds=0))
    os._exit(0)
asyncio.run(run())
""",
    )
    program = f"""
import asyncio, json, sys
from truthound.checkpoint.escalation.scheduler import create_scheduler, SchedulerConfig, SchedulerType
async def run():
    calls = []
    finished = asyncio.Event()
    async def execute(job):
        calls.append(job.record_id)
        finished.set()
    scheduler = create_scheduler(SchedulerConfig(scheduler_type=SchedulerType({backend!r}), journal_path=sys.argv[1]), execute)
    scheduler.start()
    if scheduler.get_scheduled_jobs():
        await asyncio.wait_for(finished.wait(), 5)
        await asyncio.sleep(0)
    scheduler.stop()
    print(json.dumps(calls))
asyncio.run(run())
"""
    assert json.loads(child(database, program)) == ["record"]
    assert json.loads(child(database, program)) == []


ENGINE_SETUP = """
import asyncio, json, os, sys
from pathlib import Path
from truthound.checkpoint.escalation.engine import EscalationEngine, EscalationEngineConfig
from truthound.checkpoint.escalation.protocols import EscalationPolicy, EscalationLevel, EscalationTarget
from truthound.checkpoint.escalation.scheduler import SchedulerConfig
path = sys.argv[1]
def engine(handler):
    service = EscalationEngine(EscalationEngineConfig(
        store_type='sqlite', store_config={'database': path},
        scheduler_config=SchedulerConfig(journal_path=path),
        default_notification_handler=handler,
    ))
    service.register_policy(EscalationPolicy(name='policy', levels=[
        EscalationLevel(level=1, targets=[EscalationTarget.user('local')]),
        EscalationLevel(level=2, delay_minutes=60, targets=[EscalationTarget.user('local')]),
    ]))
    return service
"""


def test_actual_notification_crash_is_not_replayed_or_bypassed(tmp_path):
    database = tmp_path / "engine.sqlite"
    child(
        database,
        ENGINE_SETUP
        + """
async def notify(*args):
    Path(path + '.sent').write_text('delivered', encoding='utf-8')
    os._exit(0)
async def run():
    await engine(notify).trigger('incident', policy_name='policy')
asyncio.run(run())
""",
    )
    assert database.with_name("engine.sqlite.sent").read_text() == "delivered"
    result = child(
        database,
        ENGINE_SETUP
        + """
async def notify(*args):
    raise AssertionError('must not redeliver')
async def run():
    service = engine(notify)
    await service.start()
    result = await service.trigger('incident', policy_name='policy')
    assert result.action == 'trigger_existing'
    assert len(service.get_records_for_incident('incident')) == 1
    try:
        await service.escalate(result.record.id)
    except RuntimeError:
        pass
    else:
        raise AssertionError('uncertain notification was bypassed')
    assert service._scheduler.get_scheduled_jobs() == []
    print(json.dumps(service.recovery_status()))
    await service.close()
asyncio.run(run())
""",
    )
    assert json.loads(result)["attempts"][0]["state"] == "unknown"


def test_live_engine_owner_cannot_be_stolen_and_close_releases_it(tmp_path):
    database = tmp_path / "owner.sqlite"
    child(
        database,
        ENGINE_SETUP
        + """
async def notify(*args): return True
async def run():
    first, second = engine(notify), engine(notify)
    original = await first.trigger('incident', policy_name='policy')
    try:
        await second.trigger('incident', policy_name='policy')
    except RuntimeError as error:
        assert 'owner' in str(error)
    else:
        raise AssertionError('second engine acquired a live service')
    await first.close()
    result = await second.trigger('incident', policy_name='policy')
    assert result.record.id == original.record.id
    assert len(second.get_records_for_incident('incident')) == 1
    await second.close()
asyncio.run(run())
""",
    )


def test_failed_notification_prevents_followup_and_preserves_original_error(tmp_path):
    database = tmp_path / "failure.sqlite"
    child(
        database,
        ENGINE_SETUP
        + """
async def run():
    for outcome in ['false', 'exception', 'cancelled']:
        async def notify(*args):
            if outcome == 'exception': raise LookupError('original')
            if outcome == 'cancelled': raise asyncio.CancelledError()
            return False
        service = engine(notify)
        try:
            await service.trigger(outcome, policy_name='policy')
        except BaseException as error:
            expected = {'false': RuntimeError, 'exception': LookupError, 'cancelled': asyncio.CancelledError}[outcome]
            assert isinstance(error, expected)
        else:
            raise AssertionError('failed notification reported success')
        assert service._scheduler.get_scheduled_jobs() == []
        assert all(row['state'] == 'unknown' for row in service.recovery_status()['attempts'])
        await service.close()
asyncio.run(run())
""",
    )


def test_receipt_and_followup_rollback_together_on_store_failure(tmp_path):
    child(
        tmp_path / "rollback.sqlite",
        ENGINE_SETUP
        + """
async def notify(*args): return True
async def run():
    service = engine(notify)
    original = service.store.save
    def fail_final_save(record):
        if record.notification_count:
            raise OSError('synthetic final persistence failure')
        return original(record)
    service.store.save = fail_final_save
    try:
        await service.trigger('incident', policy_name='policy')
    except OSError:
        pass
    else:
        raise AssertionError('final commit failure reported success')
    assert service._notification_journal.pending() == []
    assert service.get_records_for_incident('incident')[0].notification_count == 0
    await service.close()
    recovered = engine(notify)
    await recovered.start()
    assert recovered._scheduler.get_scheduled_jobs() == []
    assert recovered.recovery_status()['attempts'][0]['state'] == 'unknown'
    await recovered.close()
asyncio.run(run())
""",
    )


def test_reschedule_losing_claim_does_not_change_memory_or_durable_time(tmp_path):
    from datetime import datetime, timedelta

    import pytest

    from truthound.checkpoint.escalation.journal import ExecutionJournal
    from truthound.checkpoint.escalation.scheduler import InMemoryScheduler, SchedulerConfig

    scheduler = InMemoryScheduler(SchedulerConfig(journal_path=str(tmp_path / "race.sqlite")))
    job = scheduler.schedule_escalation("record", "policy", 2, timedelta(minutes=10))
    original_time = job.scheduled_at
    competitor = ExecutionJournal(tmp_path / "race.sqlite")
    assert competitor.claim("job:" + job.id)
    with pytest.raises(ValueError, match="no longer pending"):
        scheduler.reschedule(job.id, datetime.now())
    assert job.scheduled_at == original_time
    assert not scheduler._cancel_job(job.id)
    assert competitor.raw_state("job:" + job.id) == "attempting"
    competitor.close()
    scheduler._journal.close()


def test_actual_engine_pending_job_recovers_and_does_not_repeat(tmp_path):
    database = tmp_path / "engine-pending.sqlite"
    child(
        database,
        ENGINE_SETUP
        + """
from datetime import datetime
async def notify(record, level, targets):
    with open(path + '.levels', 'a') as stream: stream.write(str(level.level))
    return True
async def run():
    service = engine(notify)
    await service.trigger('incident', policy_name='policy')
    job = service._scheduler.get_scheduled_jobs()[0]
    service._scheduler.reschedule(job.id, datetime.now())
    os._exit(0)
asyncio.run(run())
""",
    )
    program = (
        ENGINE_SETUP
        + """
async def run():
    done = asyncio.Event()
    async def notify(record, level, targets):
        with open(path + '.levels', 'a') as stream: stream.write(str(level.level))
        done.set()
        return True
    service = engine(notify)
    await service.start()
    if service._notification_journal.pending():
        await asyncio.wait_for(done.wait(), 5)
        await asyncio.sleep(0)
    await service.close()
    assert Path(path + '.levels').read_text() == '12'
asyncio.run(run())
"""
    )
    child(database, program)
    child(database, program)


def test_operator_confirmation_continues_without_redelivery_and_retains_unknown(tmp_path):
    child(
        tmp_path / "operator.sqlite",
        ENGINE_SETUP
        + """
async def notify(*args): return False
async def run():
    service = engine(notify)
    try:
        await service.trigger('incident', policy_name='policy')
    except RuntimeError:
        pass
    attempt = service.recovery_status()['attempts'][0]['identity']
    try:
        await service.recovery_forget([attempt])
    except ValueError:
        pass
    else:
        raise AssertionError('unknown evidence was deleted')
    decision = await service.recovery_resolve(attempt, 'completed')
    assert decision['state'] == 'completed'
    record = service.get_records_for_incident('incident')[0]
    assert record.notification_count == 1
    assert len(service._notification_journal.pending()) == 1
    assert (await service.recovery_forget([attempt]))['removed'] == 1
    await service.close()
asyncio.run(run())
""",
    )


def test_policy_changed_since_reservation_cannot_send_recovered_notification(tmp_path):
    child(
        tmp_path / "policy.sqlite",
        ENGINE_SETUP
        + """
from dataclasses import replace
from datetime import datetime
async def notify(*args): return True
async def run():
    service = engine(notify)
    await service.trigger('incident', policy_name='policy')
    job = service._scheduler.get_scheduled_jobs()[0]
    service._scheduler.reschedule(job.id, datetime.now())
    await service.close()
    async def forbidden(*args): raise AssertionError('changed policy used for recovery')
    service = engine(forbidden)
    service.register_policy(replace(service.get_policy('policy'), enabled=False))
    service._ensure_owner()
    service._scheduler.recover_pending()
    job = service._scheduler.get_scheduled_jobs()[0]
    await service._scheduler._execute_job(job)
    assert service._notification_journal.raw_state('job:' + job.id) == 'unknown'
    await service.close()
asyncio.run(run())
""",
    )


def test_live_process_ownership_blocks_a_second_engine(tmp_path):
    import time
    from concurrent.futures import ThreadPoolExecutor

    database = tmp_path / "two-processes.sqlite"
    ready, release = tmp_path / "ready", tmp_path / "release"
    first_program = (
        ENGINE_SETUP
        + """
async def notify(*args): return True
async def run():
    service = engine(notify)
    await service.trigger('incident', policy_name='policy')
    Path(path).with_name('ready').touch()
    while not Path(path).with_name('release').exists():
        await asyncio.sleep(0.01)
    await service.close()
asyncio.run(run())
"""
    )
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(child, database, first_program)
        try:
            deadline = time.monotonic() + 10
            while not ready.exists() and time.monotonic() < deadline:
                if future.done():
                    future.result()
                time.sleep(0.01)
            assert ready.exists()
            child(
                database,
                ENGINE_SETUP
                + """
async def notify(*args): raise AssertionError('second owner delivered')
async def run():
    service = engine(notify)
    try:
        await service.trigger('incident', policy_name='policy')
    except RuntimeError as error:
        assert 'owner' in str(error)
    else:
        raise AssertionError('live process was displaced')
    await service.close()
asyncio.run(run())
""",
            )
        finally:
            release.touch()
        future.result(timeout=10)


@pytest.mark.parametrize('backend', ['asyncio', 'apscheduler'])
def test_native_engine_stop_drains_backend_notification_before_owner_release(tmp_path, backend):
    program = ENGINE_SETUP.replace(
        'SchedulerConfig(journal_path=path)',
        f'SchedulerConfig(scheduler_type=SchedulerType({backend!r}), journal_path=path)',
    ) + """
from dataclasses import replace
from truthound.checkpoint.escalation.scheduler import SchedulerType
async def run():
    entered = asyncio.Event()
    calls = []
    async def notify(record, level, targets):
        calls.append(level.level)
        if level.level == 2:
            entered.set()
            await asyncio.Event().wait()
        return True
    service = engine(notify)
    policy = service.get_policy('policy')
    service.register_policy(replace(policy, levels=[policy.levels[0], replace(policy.levels[1], delay_minutes=0)]))
    await service.start()
    await service.trigger('incident', policy_name='policy')
    await asyncio.wait_for(entered.wait(), 5)
    await service.stop()
    assert service._scheduler._active_tasks == set()
    assert not any(row['state'] == 'attempting' for row in service.recovery_status()['attempts'])
    assert calls == [1, 2]
    await service.start()
    await service.stop()
    assert calls == [1, 2]
    await service.close()
    replacement = engine(notify)
    await replacement.start()
    await replacement.stop()
    assert calls == [1, 2]
    assert any(row['state'] == 'unknown' for row in replacement.recovery_status()['attempts'])
    await replacement.close()
asyncio.run(run())
"""
    child(tmp_path / 'drain.sqlite', program)
