"""Native runner reservations and trigger clocks in operator-owned SQLite storage."""

import hashlib
import json
from dataclasses import asdict
from datetime import datetime
from uuid import uuid4

from truthound.checkpoint.escalation.journal import ExecutionJournal


def configuration(trigger):
    def encode(value):
        if isinstance(value, datetime):
            return value.isoformat()
        raise ValueError("Durable trigger configuration must be JSON compatible")

    return hashlib.sha256(
        json.dumps(
            {"type": trigger.trigger_type, "config": asdict(trigger.config)},
            default=encode,
            sort_keys=True,
            allow_nan=False,
        ).encode()
    ).hexdigest()


class DurableRunnerState(ExecutionJournal):
    """Restore clocks, quarantine interrupted actions, never infer their delivery."""

    def __init__(self, path):
        super().__init__(path)
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS runner_triggers "
            "(identity TEXT PRIMARY KEY, configuration TEXT NOT NULL, payload TEXT NOT NULL)"
        )

    @staticmethod
    def key(checkpoint, trigger):
        if not trigger.config.name:
            raise ValueError("Durable triggers require explicit stable names")
        return json.dumps([checkpoint.name, trigger.name])

    def save_trigger(self, checkpoint, trigger):
        self.connection.execute(
            "INSERT OR REPLACE INTO runner_triggers VALUES (?,?,?)",
            (
                self.key(checkpoint, trigger),
                configuration(trigger),
                json.dumps(trigger.export_runtime_state(), sort_keys=True, allow_nan=False),
            ),
        )

    def restore_trigger(self, checkpoint, trigger, *, compatible_configuration=None):
        row = self.connection.execute(
            "SELECT configuration,payload FROM runner_triggers WHERE identity=?",
            (self.key(checkpoint, trigger),),
        ).fetchone()
        if row:
            if row[0] not in {configuration(trigger), compatible_configuration}:
                raise ValueError("Persisted trigger configuration changed")
            trigger.restore_runtime_state(json.loads(row[1]))

    def reserve(self, checkpoint, trigger, context):
        identity = "runner:" + uuid4().hex
        with self.transaction():
            if trigger is not None:
                self.save_trigger(checkpoint, trigger)
            self.enqueue(
                identity,
                {
                    "checkpoint": checkpoint.name,
                    "trigger": trigger.name if trigger else None,
                    "configuration": configuration(trigger) if trigger else None,
                    "context": context or {},
                },
            )
        return identity

    def blocked(self, checkpoint, trigger):
        return (
            self.connection.execute(
                "SELECT 1 FROM execution_attempts WHERE state IN ('unknown','attempting') "
                "AND json_extract(payload,'$.checkpoint')=? "
                "AND json_extract(payload,'$.trigger') IS ? LIMIT 1",
                (checkpoint.name, trigger.name if trigger else None),
            ).fetchone()
            is not None
        )
