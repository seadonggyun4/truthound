"""Truthound 3.0 project context and zero-config workspace support."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from truthound.cache import get_data_fingerprint, get_source_key
from truthound.core.contracts import MetricRepository
from truthound.core.results import ValidationRunResult
from truthound.schema import Schema, learn


def _json_default(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _detect_project_root(start_path: Path) -> Path:
    start_path = start_path.resolve()
    for candidate in (start_path, *start_path.parents):
        if (candidate / ".truthound").exists():
            return candidate
        if (candidate / "truthound.yaml").exists():
            return candidate
        if (candidate / ".git").exists():
            return candidate
    return start_path


@dataclass(frozen=True)
class TruthoundContextConfig:
    """Resolved zero-config workspace defaults."""

    persist_runs: bool = True
    persist_docs: bool = True
    auto_create_baseline: bool = True
    auto_create_workspace: bool = True
    default_result_format: str = "summary"
    docs_theme: str = "professional"

    @classmethod
    def from_project_file(cls, root: Path) -> "TruthoundContextConfig":
        config_path = root / "truthound.yaml"
        if not config_path.exists():
            return cls()

        try:
            payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        except Exception:
            return cls()

        if not isinstance(payload, dict):
            return cls()

        context_data = payload.get("context", payload)
        if not isinstance(context_data, dict):
            return cls()

        return cls(
            persist_runs=bool(context_data.get("persist_runs", True)),
            persist_docs=bool(context_data.get("persist_docs", True)),
            auto_create_baseline=bool(context_data.get("auto_create_baseline", True)),
            auto_create_workspace=bool(context_data.get("auto_create_workspace", True)),
            default_result_format=str(context_data.get("default_result_format", "summary")),
            docs_theme=str(context_data.get("docs_theme", "professional")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "persist_runs": self.persist_runs,
            "persist_docs": self.persist_docs,
            "auto_create_baseline": self.auto_create_baseline,
            "auto_create_workspace": self.auto_create_workspace,
            "default_result_format": self.default_result_format,
            "docs_theme": self.docs_theme,
        }


class FileMetricRepository(MetricRepository):
    """Small file-backed metric repository for zero-config history."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._data = self._load()

    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {}
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            return payload if isinstance(payload, dict) else {}
        except Exception:
            quarantine = self.path.with_suffix(".corrupt.json")
            try:
                self.path.replace(quarantine)
            except Exception:
                pass
            return {}

    def _save(self) -> None:
        self.path.write_text(
            json.dumps(self._data, indent=2, default=_json_default),
            encoding="utf-8",
        )

    def get(self, key: str) -> Any | None:
        return self._data.get(key)

    def put(self, key: str, value: Any) -> None:
        self._data[key] = value
        self._save()

    def append_history(self, key: str, value: Any) -> None:
        history = self._data.setdefault(key, [])
        if not isinstance(history, list):
            history = []
            self._data[key] = history
        history.append(value)
        self._save()


@dataclass
class TruthoundContext:
    """Auto-discovered project context for Truthound 3.0."""

    root_dir: Path
    config: TruthoundContextConfig = field(default_factory=TruthoundContextConfig)
    workspace_dir: Path = field(init=False)
    catalog_dir: Path = field(init=False)
    baselines_dir: Path = field(init=False)
    runs_dir: Path = field(init=False)
    docs_dir: Path = field(init=False)
    plugins_dir: Path = field(init=False)
    metric_repository: FileMetricRepository = field(init=False)
    _plugin_manager: Any | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.root_dir = self.root_dir.resolve()
        self.workspace_dir = self.root_dir / ".truthound"
        self.catalog_dir = self.workspace_dir / "catalog"
        self.baselines_dir = self.workspace_dir / "baselines"
        self.runs_dir = self.workspace_dir / "runs"
        self.docs_dir = self.workspace_dir / "docs"
        self.plugins_dir = self.workspace_dir / "plugins"
        self.metric_repository = FileMetricRepository(
            self.baselines_dir / "metric-history.json"
        )
        if self.config.auto_create_workspace:
            self.ensure_workspace()

    @property
    def config_path(self) -> Path:
        return self.workspace_dir / "config.yaml"

    @property
    def catalog_index_path(self) -> Path:
        return self.catalog_dir / "assets.json"

    @property
    def baseline_index_path(self) -> Path:
        return self.baselines_dir / "index.json"

    def ensure_workspace(self) -> None:
        for path in (
            self.workspace_dir,
            self.catalog_dir,
            self.baselines_dir,
            self.runs_dir,
            self.docs_dir,
            self.plugins_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

        if not self.config_path.exists():
            self.config_path.write_text(
                yaml.safe_dump(
                    {
                        "version": 3,
                        "context": self.config.to_dict(),
                    },
                    sort_keys=False,
                    allow_unicode=True,
                ),
                encoding="utf-8",
            )

        if not self.catalog_index_path.exists():
            self.catalog_index_path.write_text("{}", encoding="utf-8")
        if not self.baseline_index_path.exists():
            self.baseline_index_path.write_text("{}", encoding="utf-8")

    def _read_json_dict(self, path: Path) -> dict[str, Any]:
        if not path.exists():
            return {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            return payload if isinstance(payload, dict) else {}
        except Exception:
            quarantine = path.with_suffix(f"{path.suffix}.corrupt")
            try:
                path.replace(quarantine)
            except Exception:
                pass
            return {}

    def _write_json(self, path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, indent=2, default=_json_default),
            encoding="utf-8",
        )

    def resolve_source_key(self, data: Any = None, source: Any = None) -> str:
        if source is not None:
            source_name = getattr(source, "name", None)
            if source_name:
                return f"source:{source_name}"
            return f"source:{type(source).__name__}"
        if data is None:
            return "unknown"
        return get_source_key(data)

    def resolve_fingerprint(self, data: Any = None, source: Any = None) -> str:
        if source is not None:
            source_name = getattr(source, "name", type(source).__name__)
            content = f"{source_name}:{getattr(source, 'row_count', 0)}:{getattr(source, 'columns', ())}"
            return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]
        return get_data_fingerprint(data)

    def track_asset(self, data: Any = None, source: Any = None) -> dict[str, Any]:
        source_key = self.resolve_source_key(data=data, source=source)
        fingerprint = self.resolve_fingerprint(data=data, source=source)
        catalog = self._read_json_dict(self.catalog_index_path)
        entry = catalog.get(source_key, {})
        entry.update(
            {
                "fingerprint": fingerprint,
                "updated_at": datetime.now().isoformat(),
            }
        )
        catalog[source_key] = entry
        self._write_json(self.catalog_index_path, catalog)
        return entry

    def get_or_create_schema_baseline(
        self,
        *,
        data: Any = None,
        source: Any = None,
        infer_constraints: bool = True,
    ) -> tuple[Schema, bool]:
        """Load an existing baseline schema or create one on first use."""

        self.ensure_workspace()
        source_key = self.resolve_source_key(data=data, source=source)
        fingerprint = self.resolve_fingerprint(data=data, source=source)
        index = self._read_json_dict(self.baseline_index_path)
        entry = index.get(source_key)

        if isinstance(entry, dict):
            schema_file = entry.get("schema_file")
            if schema_file:
                schema_path = self.baselines_dir / str(schema_file)
                if schema_path.exists():
                    try:
                        schema = Schema.load(schema_path)
                        entry["last_seen_at"] = datetime.now().isoformat()
                        entry["last_fingerprint"] = fingerprint
                        index[source_key] = entry
                        self._write_json(self.baseline_index_path, index)
                        return schema, False
                    except Exception:
                        quarantine = schema_path.with_suffix(".corrupt.yaml")
                        try:
                            schema_path.replace(quarantine)
                        except Exception:
                            pass

        if not self.config.auto_create_baseline:
            raise FileNotFoundError(
                f"No baseline found for source '{source_key}' and auto baseline creation is disabled."
            )

        schema = learn(data=data, source=source, infer_constraints=infer_constraints)
        schema_name = hashlib.sha256(source_key.encode("utf-8")).hexdigest()[:16]
        schema_path = self.baselines_dir / f"{schema_name}.schema.yaml"
        schema.save(schema_path)
        index[source_key] = {
            "schema_file": schema_path.name,
            "created_at": datetime.now().isoformat(),
            "last_seen_at": datetime.now().isoformat(),
            "last_fingerprint": fingerprint,
            "row_count": schema.row_count,
            "column_count": len(schema.columns),
        }
        self._write_json(self.baseline_index_path, index)
        self.track_asset(data=data, source=source)
        return schema, True

    def persist_run(self, run_result: ValidationRunResult) -> Path:
        self.ensure_workspace()
        output = self.runs_dir / f"{run_result.run_id}.json"
        output.write_text(run_result.to_json(), encoding="utf-8")
        self.metric_repository.append_history(
            self.resolve_source_key(run_result.source),
            {
                "run_id": run_result.run_id,
                "run_time": run_result.run_time.isoformat(),
                "suite_name": run_result.suite_name,
                "issue_count": len(run_result.issues),
                "execution_issue_count": len(run_result.execution_issues),
                "status": "failure" if run_result.has_failures else "success",
            },
        )
        return output

    def persist_docs(self, run_result: ValidationRunResult) -> Path:
        self.ensure_workspace()
        html = run_result.build_docs(theme=self.config.docs_theme)
        output = self.docs_dir / f"{run_result.run_id}.html"
        output.write_text(html, encoding="utf-8")
        return output

    def get_plugin_manager(self) -> Any:
        if self._plugin_manager is None:
            from truthound.plugins.manager import PluginManager

            self._plugin_manager = PluginManager()
        return self._plugin_manager

    @classmethod
    def discover(cls, start_path: str | Path | None = None) -> "TruthoundContext":
        start = Path(start_path or Path.cwd())
        root = _detect_project_root(start if start.is_dir() else start.parent)
        return cls(root_dir=root, config=TruthoundContextConfig.from_project_file(root))


_CONTEXT_CACHE: dict[Path, TruthoundContext] = {}


def get_context(start_path: str | Path | None = None) -> TruthoundContext:
    """Return the auto-discovered Truthound project context."""

    start = Path(start_path or Path.cwd())
    root = _detect_project_root(start if start.is_dir() else start.parent)
    cached = _CONTEXT_CACHE.get(root)
    if cached is not None:
        return cached
    context = TruthoundContext.discover(root)
    _CONTEXT_CACHE[root] = context
    return context

