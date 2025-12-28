"""Task detail view.

Renders task metrics for display in various formats.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from truthound.checkpoint.monitoring.protocols import (
    QueueMetrics,
    WorkerMetrics,
    TaskMetrics,
)
from truthound.checkpoint.monitoring.views.base import BaseView


class TaskDetailView(BaseView):
    """View for rendering task details.

    Example:
        >>> view = TaskDetailView()
        >>>
        >>> # Render single task
        >>> output = view.render(task_metrics)
        >>>
        >>> # Render for CLI
        >>> print(view.format_for_cli(task_metrics))
        Task: task-abc123
        ├── Checkpoint: daily_validation
        ├── State: running [>]
        ├── Queue: default
        ├── Worker: worker-1
        ├── Wait Time: 150ms
        └── Running For: 2.5s
    """

    def __init__(self, name: str = "task_detail") -> None:
        """Initialize task detail view."""
        super().__init__(name=name)

    def render(
        self,
        metrics: QueueMetrics | WorkerMetrics | TaskMetrics,
    ) -> dict[str, Any]:
        """Render task metrics as dictionary."""
        if not isinstance(metrics, TaskMetrics):
            return {"error": "Expected TaskMetrics"}

        return {
            "task_id": metrics.task_id,
            "checkpoint_name": metrics.checkpoint_name,
            "state": metrics.state,
            "queue_name": metrics.queue_name,
            "worker_id": metrics.worker_id,
            "timing": {
                "submitted_at": metrics.submitted_at.isoformat(),
                "started_at": metrics.started_at.isoformat() if metrics.started_at else None,
                "completed_at": metrics.completed_at.isoformat() if metrics.completed_at else None,
                "wait_time_ms": metrics.wait_time_ms,
                "execution_time_ms": metrics.execution_time_ms,
                "total_time_ms": metrics.total_time_ms,
            },
            "retries": metrics.retries,
            "error": metrics.error,
        }

    def render_summary(
        self,
        queue_metrics: list[QueueMetrics],
        worker_metrics: list[WorkerMetrics],
    ) -> dict[str, Any]:
        """Render task summary (placeholder - tasks are typically listed individually)."""
        return {
            "note": "Task view is typically used for individual tasks",
            "queues": len(queue_metrics),
            "workers": len(worker_metrics),
        }

    def render_task_list(
        self,
        tasks: list[TaskMetrics],
    ) -> dict[str, Any]:
        """Render a list of tasks."""
        by_state: dict[str, list[dict]] = {}

        for task in tasks:
            rendered = self.render(task)
            state = task.state
            if state not in by_state:
                by_state[state] = []
            by_state[state].append(rendered)

        return {
            "tasks": [self.render(t) for t in tasks],
            "by_state": by_state,
            "totals": {
                "total": len(tasks),
                **{state: len(tasks) for state, tasks in by_state.items()},
            },
        }

    def format_for_cli(
        self,
        metrics: QueueMetrics | WorkerMetrics | TaskMetrics,
    ) -> str:
        """Format task metrics for CLI display."""
        if not isinstance(metrics, TaskMetrics):
            return "Error: Expected TaskMetrics"

        state_icon = self._status_emoji(metrics.state)

        lines = [
            f"Task: {metrics.task_id}",
            f"├── Checkpoint: {metrics.checkpoint_name}",
            f"├── State: {metrics.state} {state_icon}",
            f"├── Queue: {metrics.queue_name}",
        ]

        if metrics.worker_id:
            lines.append(f"├── Worker: {metrics.worker_id}")

        lines.append(f"├── Submitted: {self._format_time(metrics.submitted_at)}")

        if metrics.started_at:
            lines.append(f"├── Started: {self._format_time(metrics.started_at)}")
            if metrics.wait_time_ms:
                lines.append(f"├── Wait Time: {self._format_duration(metrics.wait_time_ms)}")

        if metrics.completed_at:
            lines.append(f"├── Completed: {self._format_time(metrics.completed_at)}")
            if metrics.execution_time_ms:
                lines.append(f"├── Exec Time: {self._format_duration(metrics.execution_time_ms)}")
        elif metrics.started_at:
            running_for = (datetime.now() - metrics.started_at).total_seconds() * 1000
            lines.append(f"├── Running For: {self._format_duration(running_for)}")

        if metrics.retries > 0:
            lines.append(f"├── Retries: {metrics.retries}")

        if metrics.error:
            lines.append(f"└── Error: {metrics.error[:50]}...")
        else:
            # Remove the last ├── and replace with └──
            if lines:
                lines[-1] = lines[-1].replace("├──", "└──")

        return "\n".join(lines)

    def _format_time(self, dt: datetime) -> str:
        """Format datetime for display."""
        return dt.strftime("%H:%M:%S")

    def format_task_list_for_cli(
        self,
        tasks: list[TaskMetrics],
        show_completed: bool = False,
    ) -> str:
        """Format task list for CLI display."""
        if not tasks:
            return "No tasks found."

        # Filter out completed tasks unless requested
        if not show_completed:
            tasks = [t for t in tasks if t.state not in ("succeeded", "failed", "cancelled")]

        if not tasks:
            return "No active tasks."

        lines = ["Active Tasks", "=" * 70]

        # Group by state
        by_state: dict[str, list[TaskMetrics]] = {}
        for task in tasks:
            if task.state not in by_state:
                by_state[task.state] = []
            by_state[task.state].append(task)

        for state, state_tasks in by_state.items():
            icon = self._status_emoji(state)
            lines.append(f"\n{icon} {state.upper()} ({len(state_tasks)})")
            lines.append("-" * 40)

            for task in state_tasks[:10]:  # Limit to 10 per state
                age = self._format_age(task)
                line = (
                    f"  {task.task_id[:16]:16s} "
                    f"{task.checkpoint_name[:20]:20s} "
                    f"{task.queue_name:10s} "
                    f"{age:8s}"
                )
                lines.append(line)

            if len(state_tasks) > 10:
                lines.append(f"  ... and {len(state_tasks) - 10} more")

        return "\n".join(lines)

    def _format_age(self, task: TaskMetrics) -> str:
        """Format task age."""
        if task.completed_at:
            return "done"

        if task.started_at:
            elapsed = (datetime.now() - task.started_at).total_seconds()
            return self._format_duration(elapsed * 1000)

        elapsed = (datetime.now() - task.submitted_at).total_seconds()
        return f"wait {self._format_duration(elapsed * 1000)}"

    def format_for_table(
        self,
        tasks: list[TaskMetrics],
    ) -> list[list[str]]:
        """Format task list as table rows.

        Returns:
            List of rows, first row is headers.
        """
        headers = [
            "Task ID", "Checkpoint", "State", "Queue",
            "Worker", "Wait", "Exec", "Retries"
        ]

        rows = [headers]
        for task in tasks:
            row = [
                task.task_id[:16],
                task.checkpoint_name[:20],
                task.state,
                task.queue_name,
                task.worker_id or "-",
                self._format_duration(task.wait_time_ms or 0),
                self._format_duration(task.execution_time_ms or 0),
                str(task.retries),
            ]
            rows.append(row)

        return rows
