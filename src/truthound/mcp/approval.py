"""Operator-signed plans and transactional intent outcomes for workspace effects.

This is local authority, not host authentication or an OS sandbox. No MCP tool
can approve a plan. The operator CLI is a separate trusted channel.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import sqlite3
import stat
import time
from contextlib import contextmanager, suppress
from typing import TYPE_CHECKING

from truthound.mcp.policy import MCPError

if TYPE_CHECKING:
    from pathlib import Path


def canonical(value):
    """Defined local JSON encoding, not an RFC 8785/JCS implementation."""
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    )


def private_read(path: Path, maximum: int):
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    with os.fdopen(os.open(path, flags), "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > maximum:
            raise MCPError("invalid_control_state")
        if os.name != "nt" and (info.st_uid != os.getuid() or info.st_mode & 0o077):
            raise MCPError("invalid_control_state")
        return stream.read(maximum + 1)


class ApprovalStore:
    def __init__(self, directory: Path, *, clock=time.time):
        directory = directory.resolve(strict=True)
        if not directory.is_dir() or (
            os.name != "nt"
            and (directory.stat().st_uid != os.getuid() or directory.stat().st_mode & 0o077)
        ):
            raise MCPError("invalid_control_state")
        self.clock = clock
        self.directory = directory
        key_path, database = directory / ".approval-key", directory / "control.sqlite3"
        if not key_path.exists():
            if database.exists():
                raise MCPError("invalid_control_state")
            try:
                fd = os.open(key_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                with os.fdopen(fd, "wb") as stream:
                    stream.write(secrets.token_bytes(32))
                    stream.flush()
                    os.fsync(stream.fileno())
            except FileExistsError:
                pass
        self._key = private_read(key_path, 32)
        if len(self._key) != 32:
            raise MCPError("invalid_control_state")
        if not database.exists():
            with suppress(FileExistsError):
                os.close(os.open(database, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600))
        private_read(database, 64 * 1024**2)
        self.connection = sqlite3.connect(database, isolation_level=None, timeout=5.0)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.executescript("""
            CREATE TABLE IF NOT EXISTS intents (
                identity TEXT PRIMARY KEY, operation TEXT NOT NULL, digest TEXT NOT NULL,
                request TEXT NOT NULL, state TEXT NOT NULL, result TEXT
            );
            CREATE TABLE IF NOT EXISTS plans (
                identity TEXT PRIMARY KEY, intent TEXT NOT NULL, session TEXT NOT NULL,
                policy TEXT NOT NULL, expires REAL NOT NULL, state TEXT NOT NULL,
                signature TEXT, effects TEXT NOT NULL,
                FOREIGN KEY(intent) REFERENCES intents(identity)
            );
        """)

    def close(self):
        self.connection.close()

    @contextmanager
    def transaction(self):
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            yield
            self.connection.execute("COMMIT")
        except BaseException:
            self.connection.execute("ROLLBACK")
            raise

    def digest(self, value):
        return hmac.new(self._key, canonical(value).encode(), hashlib.sha256).hexdigest()

    def prepare(self, operation, *, key, request, policy, session, effects):
        digest = self.digest(request)
        identity = self.digest({"idempotency_key": key})
        with self.transaction():
            intent = self.connection.execute(
                "SELECT * FROM intents WHERE identity=?", (identity,)
            ).fetchone()
            if intent is not None:
                if intent["operation"] != operation or intent["digest"] != digest:
                    raise MCPError("idempotency_conflict")
                if intent["state"] in {"executing", "outcome_unknown"}:
                    raise MCPError("outcome_unknown")
                if intent["state"] == "succeeded":
                    return {"committed_result": json.loads(intent["result"])}
            else:
                self.connection.execute(
                    "INSERT INTO intents VALUES (?,?,?,?,'pending',NULL)",
                    (identity, operation, digest, canonical(request)),
                )
            plan = self.connection.execute(
                "SELECT * FROM plans WHERE intent=? AND session=? AND policy=? AND state IN ('awaiting_approval','approved','succeeded') AND expires>? ORDER BY rowid DESC LIMIT 1",
                (identity, session, policy, self.clock()),
            ).fetchone()
            if plan is None:
                plan_id = "plan_" + secrets.token_urlsafe(24)
                self.connection.execute(
                    "INSERT INTO plans VALUES (?,?,?,?,?,'awaiting_approval',NULL,?)",
                    (plan_id, identity, session, policy, self.clock() + 300, canonical(effects)),
                )
                plan = self.connection.execute(
                    "SELECT * FROM plans WHERE identity=?", (plan_id,)
                ).fetchone()
        return self.status(plan["identity"], session=session)

    def review(self, identity):
        plan = self.connection.execute(
            "SELECT * FROM plans WHERE identity=?", (identity,)
        ).fetchone()
        if plan is None:
            raise MCPError("invalid_plan")
        intent = self.connection.execute(
            "SELECT * FROM intents WHERE identity=?", (plan["intent"],)
        ).fetchone()
        return {
            "plan": self.status(identity, session=plan["session"]),
            "request": json.loads(intent["request"]),
        }

    def _signature(self, plan, intent):
        return self.digest(
            {
                "id": plan["identity"],
                "intent": plan["intent"],
                "operation": intent["operation"],
                "request": intent["digest"],
                "session": plan["session"],
                "policy": plan["policy"],
                "expires": plan["expires"],
                "effects": json.loads(plan["effects"]),
            }
        )

    def approve(self, identity):
        """Trusted operator API only. Never register this method as an MCP tool."""
        with self.transaction():
            plan = self.connection.execute(
                "SELECT * FROM plans WHERE identity=?", (identity,)
            ).fetchone()
            if (
                plan is None
                or plan["state"] != "awaiting_approval"
                or self.clock() >= plan["expires"]
            ):
                raise MCPError("invalid_plan")
            intent = self.connection.execute(
                "SELECT * FROM intents WHERE identity=?", (plan["intent"],)
            ).fetchone()
            signature = self._signature(plan, intent)
            self.connection.execute(
                "UPDATE plans SET state='approved',signature=? WHERE identity=?",
                (signature, identity),
            )

    def revoke(self, identity):
        with self.transaction():
            plan = self.connection.execute(
                "SELECT state FROM plans WHERE identity=?", (identity,)
            ).fetchone()
            if plan is None or plan["state"] not in {"awaiting_approval", "approved"}:
                raise MCPError("invalid_plan")
            self.connection.execute(
                "UPDATE plans SET state='revoked' WHERE identity=?", (identity,)
            )

    def status(self, identity, *, session):
        plan = self.connection.execute(
            "SELECT * FROM plans WHERE identity=? AND session=?", (identity, session)
        ).fetchone()
        if plan is None:
            raise MCPError("invalid_plan")
        intent = self.connection.execute(
            "SELECT * FROM intents WHERE identity=?", (plan["intent"],)
        ).fetchone()
        state = plan["state"]
        if intent["state"] == "outcome_unknown":
            state = "outcome_unknown"
        elif self.clock() >= plan["expires"] and state in {"awaiting_approval", "approved"}:
            state = "expired"
        return {
            "plan_id": identity,
            "operation_id": intent["operation"],
            "state": state,
            "expires_at": plan["expires"],
            "request_digest": intent["digest"],
            "effects": json.loads(plan["effects"]),
        }

    def begin(self, identity, *, session, policy):
        with self.transaction():
            plan = self.connection.execute(
                "SELECT * FROM plans WHERE identity=? AND session=?", (identity, session)
            ).fetchone()
            if plan is None:
                raise MCPError("invalid_plan")
            intent = self.connection.execute(
                "SELECT * FROM intents WHERE identity=?", (plan["intent"],)
            ).fetchone()
            if plan["policy"] != policy:
                raise MCPError("plan_stale")
            if self.clock() >= plan["expires"]:
                raise MCPError("plan_expired")
            if (
                plan["state"] not in {"approved", "succeeded"}
                or not plan["signature"]
                or not hmac.compare_digest(plan["signature"], self._signature(plan, intent))
            ):
                raise MCPError("approval_required")
            if intent["state"] == "succeeded":
                return json.loads(intent["result"])
            if intent["state"] != "pending":
                raise MCPError("outcome_unknown")
            self.connection.execute(
                "UPDATE intents SET state='executing' WHERE identity=?", (plan["intent"],)
            )
            self.connection.execute(
                "UPDATE plans SET state='executing' WHERE identity=?", (identity,)
            )
            return None

    def finish(self, identity, result):
        with self.transaction():
            plan = self.connection.execute(
                "SELECT * FROM plans WHERE identity=? AND state='executing'", (identity,)
            ).fetchone()
            if plan is None:
                raise MCPError("invalid_plan")
            self.connection.execute(
                "UPDATE intents SET state='succeeded',result=? WHERE identity=?",
                (canonical(result), plan["intent"]),
            )
            self.connection.execute(
                "UPDATE plans SET state='succeeded' WHERE identity=?", (identity,)
            )

    def uncertain(self, identity):
        with self.transaction():
            plan = self.connection.execute(
                "SELECT * FROM plans WHERE identity=?", (identity,)
            ).fetchone()
            if plan is not None:
                self.connection.execute(
                    "UPDATE intents SET state='outcome_unknown' WHERE identity=? AND state='executing'",
                    (plan["intent"],),
                )
                self.connection.execute(
                    "UPDATE plans SET state='outcome_unknown' WHERE identity=? AND state='executing'",
                    (identity,),
                )
