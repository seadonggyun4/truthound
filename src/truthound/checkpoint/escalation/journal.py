"""Durable at-most-once execution claims for operator-owned services."""

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from threading import RLock


class ExecutionJournal:
    """Persist intent before effects; interrupted attempts never become pending."""

    def __init__(self, path, *, connection=None):
        if str(path) == ":memory:":
            raise ValueError("Durable execution requires a file-backed journal")
        self.path = str(Path(path).resolve())
        self._lock = RLock()
        self._owns_connection = connection is None
        self._owner = None
        self.connection = connection or sqlite3.connect(
            self.path, isolation_level=None, timeout=5, check_same_thread=False
        )
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS execution_attempts "
            "(identity TEXT PRIMARY KEY, state TEXT NOT NULL, payload TEXT NOT NULL)"
        )

    @contextmanager
    def transaction(self):
        with self._lock:
            # Savepoints compose with the incident store's outer transaction.
            nested = self.connection.in_transaction
            self.connection.execute("SAVEPOINT execution_journal" if nested else "BEGIN IMMEDIATE")
            try:
                yield
                self.connection.execute("RELEASE execution_journal" if nested else "COMMIT")
            except BaseException:
                self.connection.execute("ROLLBACK TO execution_journal" if nested else "ROLLBACK")
                if nested:
                    self.connection.execute("RELEASE execution_journal")
                raise

    def enqueue(self, identity, payload):
        raw = json.dumps(payload, allow_nan=False, sort_keys=True)
        with self.transaction():
            self.connection.execute(
                "INSERT OR IGNORE INTO execution_attempts VALUES (?, 'pending', ?)",
                (identity, raw),
            )
            stored = self.connection.execute(
                "SELECT payload FROM execution_attempts WHERE identity=?", (identity,)
            ).fetchone()[0]
            if stored != raw:
                raise ValueError("execution identity conflict")

    def claim(self, identity):
        with self.transaction():
            return (
                self.connection.execute(
                    "UPDATE execution_attempts SET state='attempting' "
                    "WHERE identity=? AND state='pending'",
                    (identity,),
                ).rowcount
                == 1
            )

    def finish(self, identity, state):
        if state not in {"completed", "unknown", "cancelled"}:
            raise ValueError("invalid execution outcome")
        with self.transaction():
            if (
                self.connection.execute(
                    "UPDATE execution_attempts SET state=? WHERE identity=? AND state='attempting'",
                    (state, identity),
                ).rowcount
                != 1
            ):
                raise ValueError("execution has no active claim")

    def pending(self):
        return [
            (identity, json.loads(payload))
            for identity, payload in self.connection.execute(
                "SELECT identity,payload FROM execution_attempts WHERE state='pending'"
            )
        ]

    def state(self, identity):
        row = self.connection.execute(
            "SELECT state FROM execution_attempts WHERE identity=?", (identity,)
        ).fetchone()
        # A persisted attempt is uncertain to a new owner; it is never replayable.
        return "unknown" if row and row[0] == "attempting" else row[0] if row else None

    def update_pending(self, identity, payload):
        """Update a reservation only while no executor has claimed it."""
        raw = json.dumps(payload, allow_nan=False, sort_keys=True)
        with self.transaction():
            if (
                self.connection.execute(
                    "UPDATE execution_attempts SET payload=? WHERE identity=? AND state='pending'",
                    (raw, identity),
                ).rowcount
                != 1
            ):
                raise ValueError("execution is no longer pending")

    def uncertain(self):
        return [
            identity
            for (identity,) in self.connection.execute(
                "SELECT identity FROM execution_attempts WHERE state IN ('attempting','unknown')"
            )
        ]

    def acquire_owner(self):
        """Hold a process-bound SQLite lock; never steal from a live owner.

        The separate lock database permits journal commits while ownership is
        held. OS handle cleanup releases it even after SIGKILL or os._exit.
        """
        if self._owner is not None:
            return
        owner = sqlite3.connect(
            self.path + ".owner", isolation_level=None, timeout=0, check_same_thread=False
        )
        try:
            owner.execute("BEGIN EXCLUSIVE")
        except BaseException:
            owner.close()
            raise RuntimeError("Durable service already has an owner") from None
        self._owner = owner
        try:
            with self.transaction():
                self.connection.execute(
                    "UPDATE execution_attempts SET state='unknown' WHERE state='attempting'"
                )
        except BaseException:
            self.release_owner()
            raise

    def bind_scope(self, scope):
        """Pin recovery to an operator-supplied authorization generation."""
        self.acquire_owner()
        with self.transaction():
            self.connection.execute(
                "CREATE TABLE IF NOT EXISTS execution_scope (identity INTEGER PRIMARY KEY, scope TEXT NOT NULL)"
            )
            self.connection.execute("INSERT OR IGNORE INTO execution_scope VALUES (1,?)", (scope,))
            stored = self.connection.execute(
                "SELECT scope FROM execution_scope WHERE identity=1"
            ).fetchone()[0]
            if stored != scope:
                raise ValueError("Durable service authorization scope changed")

    def cancel_pending(self, identity):
        """Cancel atomically without creating an ambiguous execution claim."""
        with self.transaction():
            return (
                self.connection.execute(
                    "UPDATE execution_attempts SET state='cancelled' "
                    "WHERE identity=? AND state='pending'",
                    (identity,),
                ).rowcount
                == 1
            )

    def attempts(self, *, prefix="", limit=100, offset=0):
        if not 1 <= limit <= 1001 or offset < 0:
            raise ValueError("Invalid recovery page")
        rows = self.connection.execute(
            "SELECT identity,state,payload FROM execution_attempts "
            "WHERE substr(identity,1,?)=? ORDER BY identity LIMIT ? OFFSET ?",
            (len(prefix), prefix, limit, offset),
        )
        return [(identity, state, json.loads(payload)) for identity, state, payload in rows]

    def resolve_unknown(self, identity, resolution):
        """Record an operator decision without repeating an external effect."""
        if resolution not in {"completed", "cancelled"}:
            raise ValueError("Invalid recovery resolution")
        with self.transaction():
            if (
                self.connection.execute(
                    "UPDATE execution_attempts SET state=? WHERE identity=? AND state='unknown'",
                    (resolution, identity),
                ).rowcount
                != 1
            ):
                raise ValueError("Execution is not an unknown outcome")

    def forget_terminal(self, identities):
        """Explicit bounded retention; unfinished evidence cannot be removed."""
        if not identities or len(identities) > 100 or len(set(identities)) != len(identities):
            raise ValueError("Supply 1 to 100 unique attempt identities")
        with self.transaction():
            for identity in identities:
                if self.raw_state(identity) not in {"completed", "cancelled"}:
                    raise ValueError("Only terminal execution evidence may be removed")
            for identity in identities:
                self.connection.execute(
                    "DELETE FROM execution_attempts WHERE identity=?", (identity,)
                )
        return len(identities)

    def raw_state(self, identity):
        row = self.connection.execute(
            "SELECT state FROM execution_attempts WHERE identity=?", (identity,)
        ).fetchone()
        return row[0] if row else None

    def release_owner(self):
        if self._owner is not None:
            self._owner.close()
            self._owner = None

    def close(self):
        self.release_owner()
        if self._owns_connection:
            self.connection.close()
