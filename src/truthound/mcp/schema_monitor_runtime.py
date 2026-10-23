"""Native watcher supervision. No transport URL or credentials enter the wire."""

import asyncio
import json
import threading

from truthound.mcp.encoding import encode
from truthound.mcp.policy import MCPError
from truthound.mcp.validator_wire import _known_type


class MonitorRuntime:
    def __init__(self, source, notifier, *, mode, interval, max_events, budget, guard):
        from truthound.profiler.evolution.breaking_alerts import AlertNotifier
        from truthound.profiler.evolution.watcher import SchemaSource

        if not isinstance(source, SchemaSource) or (
            notifier is not None and not isinstance(notifier, AlertNotifier)
        ):
            raise MCPError("invalid_binding")
        self.lock = threading.RLock()
        self.error = None
        self.stopped = False
        self.polls = 0
        self.events = 0
        self.sent = 0
        self.mode = mode
        self.loop = None
        self.thread = None
        self.guard = guard
        self.budget = budget
        self.max_events = max_events
        self.interval = interval
        history_prefix = "truthound.profiler.evolution.history."
        watch_prefix = "truthound.profiler.evolution.watcher."
        alert_prefix = "truthound.profiler.evolution.breaking_alerts."
        self.history = _known_type(history_prefix + "InMemorySchemaHistory")(
            max_versions=max_events
        )
        manager = _known_type(history_prefix + "SchemaHistory")(self.history)
        runtime = self

        class CheckedNotifier:
            def send(self, alert):
                guard()
                if notifier.send(alert) is not True:
                    raise MCPError("notification_delivery_unknown")
                runtime.sent += 1
                return True

        self.alerts = _known_type(alert_prefix + "BreakingChangeAlertManager")(
            notifiers=[CheckedNotifier()] if notifier is not None else []
        )
        handlers = [
            _known_type(watch_prefix + "HistoryEventHandler")(manager),
            _known_type(watch_prefix + "AlertingEventHandler")(self.alerts),
        ]
        sync_class = _known_type(watch_prefix + "SchemaWatcher")
        async_class = _known_type(watch_prefix + "AsyncSchemaWatcher")

        class SyncWatcher(sync_class):
            def check_now(self):
                with runtime.lock:
                    if not runtime.before():
                        return []
                    try:
                        return runtime.after(super().check_now())
                    except Exception:
                        runtime.error = "monitor_execution_or_delivery_unknown"
                        return []

        class AsyncWatcher(async_class):
            async def check_now(self):
                with runtime.lock:
                    if not runtime.before():
                        return []
                    try:
                        return runtime.after(await super().check_now())
                    except Exception:
                        runtime.error = "monitor_execution_or_delivery_unknown"
                        return []

        self.watcher = SyncWatcher() if mode == "sync" else AsyncWatcher()
        guard()
        self.watcher.add_source(source)
        for handler in handlers:
            self.watcher.add_handler(handler if mode == "sync" else handler.handle)
        self.check_budget()

    def before(self):
        if self.error or self.stopped:
            return False
        try:
            self.guard()
            if self.events >= self.max_events:
                raise MCPError("resource_exhausted")
            self.check_budget()
            return True
        except Exception:
            self.error = "monitor_authority_or_budget_failed"
            return False

    def after(self, events):
        self.events += len(events)
        self.polls += 1
        self.check_budget()
        return events

    def snapshot(self):
        with self.lock:
            if not self.stopped and hasattr(self, "watcher"):
                if self.mode == "sync":
                    thread = self.watcher._watch_thread
                    ended = thread is not None and not thread.is_alive()
                else:
                    task = self.watcher._task
                    ended = task is not None and task.done()
                if ended:
                    self.error = self.error or "monitor_poller_terminated"
            return {
                "state": "failed" if self.error else "stopped" if self.stopped else "running",
                "error": self.error,
                "mode": self.mode,
                "polls": self.polls,
                "events": self.events,
                "notifications_sent": self.sent,
                "versions": [item.to_dict() for item in self.history.list_versions()],
                "alerts": [item.to_dict() for item in self.alerts.get_alert_history()],
            }

    def check_budget(self):
        if len(json.dumps(encode(self.snapshot()), allow_nan=False).encode()) > self.budget:
            raise MCPError("resource_exhausted")

    def start(self):
        if self.mode == "sync":
            self.watcher.start(self.interval)
        else:
            self.loop = asyncio.new_event_loop()
            self.thread = threading.Thread(target=self.loop.run_forever, daemon=True)
            self.thread.start()
            asyncio.run_coroutine_threadsafe(self.watcher.start(self.interval), self.loop).result()

    def stop(self):
        if self.stopped:
            return
        # Never join the poller while holding its state lock.
        with self.lock:
            self.stopped = True
        if self.mode == "sync":
            self.watcher.stop()
            if self.watcher._watch_thread and self.watcher._watch_thread.is_alive():
                raise MCPError("monitor_stop_incomplete")
        elif self.loop is not None:
            asyncio.run_coroutine_threadsafe(self.watcher.stop(), self.loop).result()
            self.loop.call_soon_threadsafe(self.loop.stop)
            self.thread.join()
            self.loop.close()
