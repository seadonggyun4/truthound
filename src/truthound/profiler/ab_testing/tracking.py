"""Experiment tracking and storage.

Provides persistence and tracking for A/B experiments.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from truthound.profiler.ab_testing.base import (
    ExperimentConfig,
    ExperimentResult,
    ExperimentStatus,
)


logger = logging.getLogger(__name__)


class ExperimentStore:
    """Storage backend for experiment data.

    Abstract base for different storage implementations.
    """

    def save_experiment(self, experiment_id: str, data: Dict[str, Any]) -> None:
        """Save experiment data."""
        raise NotImplementedError

    def load_experiment(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Load experiment data."""
        raise NotImplementedError

    def list_experiments(self) -> List[str]:
        """List all experiment IDs."""
        raise NotImplementedError

    def delete_experiment(self, experiment_id: str) -> bool:
        """Delete experiment data."""
        raise NotImplementedError


class FileExperimentStore(ExperimentStore):
    """File-based experiment storage.

    Stores experiments as JSON files in a directory.
    """

    def __init__(self, directory: str | Path):
        """Initialize store.

        Args:
            directory: Directory to store experiments
        """
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def _get_path(self, experiment_id: str) -> Path:
        return self.directory / f"{experiment_id}.json"

    def save_experiment(self, experiment_id: str, data: Dict[str, Any]) -> None:
        """Save experiment to file."""
        path = self._get_path(experiment_id)

        # Convert datetime objects to strings
        def convert_datetime(obj: Any) -> Any:
            if isinstance(obj, datetime):
                return obj.isoformat()
            elif isinstance(obj, dict):
                return {k: convert_datetime(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_datetime(v) for v in obj]
            return obj

        data = convert_datetime(data)

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        logger.debug(f"Saved experiment {experiment_id} to {path}")

    def load_experiment(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Load experiment from file."""
        path = self._get_path(experiment_id)

        if not path.exists():
            return None

        with open(path) as f:
            return json.load(f)

    def list_experiments(self) -> List[str]:
        """List all experiment IDs."""
        return [
            p.stem for p in self.directory.glob("*.json")
        ]

    def delete_experiment(self, experiment_id: str) -> bool:
        """Delete experiment file."""
        path = self._get_path(experiment_id)

        if path.exists():
            path.unlink()
            return True
        return False


class MemoryExperimentStore(ExperimentStore):
    """In-memory experiment storage.

    Useful for testing and short-lived experiments.
    """

    def __init__(self):
        self._data: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

    def save_experiment(self, experiment_id: str, data: Dict[str, Any]) -> None:
        with self._lock:
            self._data[experiment_id] = data

    def load_experiment(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self._data.get(experiment_id)

    def list_experiments(self) -> List[str]:
        with self._lock:
            return list(self._data.keys())

    def delete_experiment(self, experiment_id: str) -> bool:
        with self._lock:
            if experiment_id in self._data:
                del self._data[experiment_id]
                return True
            return False

    def clear(self) -> None:
        """Clear all experiments."""
        with self._lock:
            self._data.clear()


class ExperimentTracker:
    """Track and manage A/B experiments.

    Provides lifecycle management, result storage, and querying.

    Example:
        tracker = ExperimentTracker()

        # Register experiment
        tracker.register(experiment)

        # Update status
        tracker.update_status(experiment.experiment_id, ExperimentStatus.RUNNING)

        # Save result
        tracker.save_result(result)

        # Query experiments
        completed = tracker.get_experiments(status=ExperimentStatus.COMPLETED)
    """

    _instance: Optional["ExperimentTracker"] = None
    _lock: threading.Lock = threading.Lock()

    # Default storage directory
    DEFAULT_STORAGE_DIR = Path.home() / ".truthound" / "experiments"

    def __new__(cls, store: Optional[ExperimentStore] = None) -> "ExperimentTracker":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self, store: Optional[ExperimentStore] = None):
        if self._initialized:
            return

        if store is None:
            store = FileExperimentStore(self.DEFAULT_STORAGE_DIR)

        self._store = store
        self._active_experiments: Dict[str, Dict[str, Any]] = {}
        self._initialized = True

    def register(self, config: ExperimentConfig, experiment_id: str) -> None:
        """Register a new experiment.

        Args:
            config: Experiment configuration
            experiment_id: Unique experiment ID
        """
        data = {
            "experiment_id": experiment_id,
            "config": config.to_dict(),
            "status": ExperimentStatus.DRAFT.value,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "results": None,
        }

        self._active_experiments[experiment_id] = data
        self._store.save_experiment(experiment_id, data)

        logger.info(f"Registered experiment: {experiment_id}")

    def update_status(
        self,
        experiment_id: str,
        status: ExperimentStatus,
    ) -> None:
        """Update experiment status.

        Args:
            experiment_id: Experiment ID
            status: New status
        """
        data = self._get_data(experiment_id)
        if data:
            data["status"] = status.value
            data["updated_at"] = datetime.now().isoformat()
            self._store.save_experiment(experiment_id, data)
            logger.debug(f"Updated experiment {experiment_id} status to {status.value}")

    def save_result(self, result: ExperimentResult) -> None:
        """Save experiment result.

        Args:
            result: Experiment result
        """
        data = self._get_data(result.experiment_id)
        if data:
            data["results"] = result.to_dict()
            data["status"] = result.status.value
            data["updated_at"] = datetime.now().isoformat()
            self._store.save_experiment(result.experiment_id, data)
            logger.info(f"Saved result for experiment {result.experiment_id}")
        else:
            # Create new entry
            data = {
                "experiment_id": result.experiment_id,
                "config": {},
                "status": result.status.value,
                "created_at": result.start_time.isoformat(),
                "updated_at": datetime.now().isoformat(),
                "results": result.to_dict(),
            }
            self._store.save_experiment(result.experiment_id, data)

    def get_experiment(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Get experiment data.

        Args:
            experiment_id: Experiment ID

        Returns:
            Experiment data or None
        """
        return self._get_data(experiment_id)

    def get_result(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Get experiment result.

        Args:
            experiment_id: Experiment ID

        Returns:
            Result data or None
        """
        data = self._get_data(experiment_id)
        return data.get("results") if data else None

    def get_experiments(
        self,
        status: Optional[ExperimentStatus] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Get experiments, optionally filtered by status.

        Args:
            status: Filter by status
            limit: Maximum results

        Returns:
            List of experiment data
        """
        experiments = []

        for exp_id in self._store.list_experiments():
            data = self._get_data(exp_id)
            if data:
                if status is None or data.get("status") == status.value:
                    experiments.append(data)
                    if len(experiments) >= limit:
                        break

        return experiments

    def delete_experiment(self, experiment_id: str) -> bool:
        """Delete an experiment.

        Args:
            experiment_id: Experiment ID

        Returns:
            True if deleted
        """
        if experiment_id in self._active_experiments:
            del self._active_experiments[experiment_id]

        return self._store.delete_experiment(experiment_id)

    def _get_data(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Get experiment data from cache or store."""
        if experiment_id in self._active_experiments:
            return self._active_experiments[experiment_id]

        data = self._store.load_experiment(experiment_id)
        if data:
            self._active_experiments[experiment_id] = data
        return data

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all experiments.

        Returns:
            Summary statistics
        """
        all_experiments = self.get_experiments(limit=10000)

        status_counts = {}
        for exp in all_experiments:
            status = exp.get("status", "unknown")
            status_counts[status] = status_counts.get(status, 0) + 1

        winners = {}
        for exp in all_experiments:
            result = exp.get("results", {})
            winner = result.get("winner")
            if winner:
                winners[winner] = winners.get(winner, 0) + 1

        return {
            "total_experiments": len(all_experiments),
            "status_counts": status_counts,
            "winner_counts": winners,
        }


def get_tracker(store: Optional[ExperimentStore] = None) -> ExperimentTracker:
    """Get the global experiment tracker.

    Args:
        store: Optional custom store

    Returns:
        ExperimentTracker singleton
    """
    return ExperimentTracker(store)
