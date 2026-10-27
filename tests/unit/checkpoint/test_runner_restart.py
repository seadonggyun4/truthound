"""Actual native checkpoints retain clocks and quarantine interrupted actions."""

import json

from tests.unit.checkpoint.escalation.test_journal_restart import child

SETUP = """
import json, os, sys, time
from pathlib import Path
from truthound.checkpoint import Checkpoint
from truthound.checkpoint.runner import CheckpointRunner, RunnerConfig
from truthound.checkpoint.triggers.schedule import ScheduleTrigger
from truthound.checkpoint.actions.base import BaseAction, ActionConfig, ActionResult, ActionStatus
path = sys.argv[1]
class MarkerAction(BaseAction):
    action_type = 'marker'
    @classmethod
    def _default_config(cls): return ActionConfig(name='local', notify_on='always')
    def _execute(self, result):
        marker = Path(path + '.sent')
        with marker.open('a', encoding='utf-8') as stream:
            stream.write('delivered\\n')
            stream.flush()
            os.fsync(stream.fileno())
        if Path(path + '.crash').exists(): os._exit(0)
        return ActionResult(self.name, self.action_type, ActionStatus.SUCCESS)
def runner():
    checkpoint = Checkpoint(name='quality', data_source={'x': [1, None]}, validators=['null'], actions=[MarkerAction()])
    trigger = ScheduleTrigger(name='clock', interval_seconds=60, max_runs=1)
    checkpoint.add_trigger(trigger)
    service = CheckpointRunner(RunnerConfig(journal_path=path, poll_interval_seconds=0.01))
    service.add_checkpoint(checkpoint)
    return service, checkpoint, trigger
"""


def test_interrupted_real_checkpoint_action_is_quarantined_after_restart(tmp_path):
    database = tmp_path / "runner.sqlite"
    database.with_name("runner.sqlite.crash").touch()
    child(
        database,
        SETUP
        + """
service, checkpoint, trigger = runner()
service.start()
assert service.wait_for_completion(timeout=10), 'expected process to exit in the actual action'
raise AssertionError('process did not exit in action')
""",
    )
    database.with_name("runner.sqlite.crash").unlink()
    result = child(
        database,
        SETUP
        + """
service, checkpoint, trigger = runner()
service.start()
service.stop()
service._check_triggers()
assert Path(path + '.sent').read_text().count('delivered') == 1
print(json.dumps(service.recovery_status()))
service.close()
""",
    )
    assert [row["state"] for row in json.loads(result)["attempts"]] == ["unknown"]


def test_completed_trigger_limit_survives_process_restart(tmp_path):
    database = tmp_path / "completed.sqlite"
    child(
        database,
        SETUP
        + """
service, checkpoint, trigger = runner()
service.start()
assert len(service.get_results(timeout=10, max_results=1)) == 1
service.close()
""",
    )
    child(
        database,
        SETUP
        + """
service, checkpoint, trigger = runner()
service.start()
service.stop()
assert trigger.run_count == 1 and trigger.last_run is not None
service._check_triggers()
assert Path(path + '.sent').read_text().count('delivered') == 1
service.close()
""",
    )


def test_unclaimed_reservation_survives_process_loss(tmp_path):
    database = tmp_path / "pending.sqlite"
    child(
        database,
        SETUP
        + """
service, checkpoint, trigger = runner()
service._ensure_owner()
service._durable.reserve(checkpoint, trigger, {'trigger': trigger.name})
os._exit(0)
""",
    )
    child(
        database,
        SETUP
        + """
service, checkpoint, trigger = runner()
service.start()
assert len(service.get_results(timeout=10, max_results=1)) == 1
service.close()
assert trigger.run_count == 1
assert Path(path + '.sent').read_text().count('delivered') == 1
""",
    )


def test_discarding_interrupted_run_consumes_trigger_slot_without_actions(tmp_path):
    database = tmp_path / "discard.sqlite"
    database.with_name("discard.sqlite.crash").touch()
    child(
        database,
        SETUP
        + """
service, _, _ = runner()
service.start()
service.wait_for_completion(timeout=10)
raise AssertionError('action failed to exit')
""",
    )
    database.with_name("discard.sqlite.crash").unlink()
    child(
        database,
        SETUP
        + """
service, checkpoint, trigger = runner()
service._ensure_owner()
identity = service.recovery_status()['attempts'][0]['identity']
service.recovery_resolve(identity, 'cancelled')
service.start()
service.stop()
service._check_triggers()
assert trigger.run_count == 1
assert Path(path + '.sent').read_text().count('delivered') == 1
service.close()
""",
    )


def test_stop_deadline_keeps_ownership_while_native_execution_is_alive(tmp_path):
    import threading

    import pytest

    from truthound.checkpoint import Checkpoint
    from truthound.checkpoint.runner import CheckpointRunner, RunnerConfig
    from truthound.checkpoint.triggers.schedule import ScheduleTrigger

    entered, release = threading.Event(), threading.Event()
    checkpoint = Checkpoint(name="blocked", data_source={"x": [1]})
    checkpoint.add_trigger(ScheduleTrigger(name="clock", interval_seconds=60))

    def block(context=None):
        entered.set()
        assert release.wait(5)
        return object()

    checkpoint.run = block
    config = RunnerConfig(journal_path=str(tmp_path / "blocked.sqlite"), poll_interval_seconds=0.01)
    first = CheckpointRunner(config).add_checkpoint(checkpoint)
    second = CheckpointRunner(config)
    try:
        first.start()
        assert entered.wait(5)
        first.stop(timeout=0.01)
        assert first._thread.is_alive()
        with pytest.raises(RuntimeError, match="quiescent"):
            first.edit_schedule("blocked", "clock", {"interval_seconds": 120})
        with pytest.raises(RuntimeError, match="owner"):
            second.start()
    finally:
        release.set()
        first.close()
        second.close()


def test_cron_and_custom_shared_edits_preserve_history_and_native_evaluation():
    from datetime import datetime

    import pytest

    from truthound.checkpoint import Checkpoint
    from truthound.checkpoint.runner import CheckpointRunner
    from truthound.checkpoint.triggers.event import EventTrigger
    from truthound.checkpoint.triggers.schedule import CronTrigger

    checkpoint = Checkpoint(name="edits", data_source={"x": [1]})
    cron = CronTrigger(name="cron", expression="0 1 * * *")
    event = EventTrigger(name="event", event_type="approved-event")
    checkpoint.add_trigger(cron).add_trigger(event)
    service = CheckpointRunner().add_checkpoint(checkpoint)
    cron.record_run()
    event.record_run()
    assert cron._matches_cron(datetime(2026, 10, 24, 1, 0))
    service.edit_schedule("edits", "cron", {"expression": "0 2 * * *"})
    assert not cron._matches_cron(datetime(2026, 10, 24, 1, 0))
    assert cron._matches_cron(datetime(2026, 10, 24, 2, 0))
    service.edit_schedule("edits", "event", {"max_runs": 1})
    assert not event.should_continue()
    assert cron.run_count == event.run_count == 1
    with pytest.raises(ValueError, match="shared settings"):
        service.edit_schedule("edits", "event", {"event_type": "arbitrary-provider-event"})
    with pytest.raises(ValueError):
        service.edit_schedule("edits", "cron", {"expression": "invalid"})
    assert cron.config.expression == "0 2 * * *"
    service.close()
