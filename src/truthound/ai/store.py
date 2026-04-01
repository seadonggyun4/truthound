"""Lazy workspace layout and artifact store for ``truthound.ai``."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from truthound._ai_contract import (
    AI_ANALYSES_DIRNAME,
    AI_APPROVALS_DIRNAME,
    AI_APPROVAL_LOG_FILENAME,
    AI_PROPOSALS_DIRNAME,
    AI_ROOT_DIRNAME,
    is_valid_analysis_artifact_id,
    is_valid_proposal_artifact_id,
)
from truthound._ai_redaction import SummaryOnlyRedactor
from truthound.ai.models import ApprovalLogEvent, RunAnalysisArtifact, SuiteProposalArtifact

if TYPE_CHECKING:
    from truthound.context import TruthoundContext


@dataclass(frozen=True)
class AIWorkspaceLayout:
    """Resolved AI artifact layout under ``.truthound/ai``."""

    workspace_dir: Path

    @property
    def root_dir(self) -> Path:
        return self.workspace_dir / AI_ROOT_DIRNAME

    @property
    def proposals_dir(self) -> Path:
        return self.root_dir / AI_PROPOSALS_DIRNAME

    @property
    def analyses_dir(self) -> Path:
        return self.root_dir / AI_ANALYSES_DIRNAME

    @property
    def approvals_dir(self) -> Path:
        return self.root_dir / AI_APPROVALS_DIRNAME

    @property
    def approval_log_path(self) -> Path:
        return self.approvals_dir / AI_APPROVAL_LOG_FILENAME

    def ensure_workspace(self) -> None:
        for path in (self.root_dir, self.proposals_dir, self.analyses_dir, self.approvals_dir):
            path.mkdir(parents=True, exist_ok=True)

    def proposal_path(self, artifact_id: str) -> Path:
        if not is_valid_proposal_artifact_id(artifact_id):
            raise ValueError(f"Invalid suite proposal artifact_id: {artifact_id}")
        return self._resolve_safe_path(self.proposals_dir / f"{artifact_id}.json")

    def analysis_path(self, artifact_id: str) -> Path:
        if not is_valid_analysis_artifact_id(artifact_id):
            raise ValueError(f"Invalid run analysis artifact_id: {artifact_id}")
        return self._resolve_safe_path(self.analyses_dir / f"{artifact_id}.json")

    def _resolve_safe_path(self, path: Path) -> Path:
        root = self.root_dir.resolve()
        resolved = path.resolve()
        if not resolved.is_relative_to(root):
            raise ValueError(f"AI artifact path escapes workspace root: {path}")
        return path


class AIArtifactStore:
    """Persist and read AI artifacts without mutating the base workspace contract."""

    def __init__(self, context: "TruthoundContext") -> None:
        self._context = context
        self.layout = AIWorkspaceLayout(context.workspace_dir)
        self._redactor = SummaryOnlyRedactor()

    def write_proposal(self, artifact: SuiteProposalArtifact) -> Path:
        path = self.layout.proposal_path(artifact.artifact_id)
        return self._write_json_artifact(path, artifact)

    def write_analysis(self, artifact: RunAnalysisArtifact) -> Path:
        path = self.layout.analysis_path(artifact.artifact_id)
        return self._write_json_artifact(path, artifact)

    def read_proposal(self, artifact_id: str) -> SuiteProposalArtifact:
        return SuiteProposalArtifact.model_validate_json(
            self.layout.proposal_path(artifact_id).read_text(encoding="utf-8")
        )

    def read_analysis(self, artifact_id: str) -> RunAnalysisArtifact:
        return RunAnalysisArtifact.model_validate_json(
            self.layout.analysis_path(artifact_id).read_text(encoding="utf-8")
        )

    def list_proposals(self) -> list[SuiteProposalArtifact]:
        if not self.layout.proposals_dir.exists():
            return []
        artifacts = [
            SuiteProposalArtifact.model_validate_json(path.read_text(encoding="utf-8"))
            for path in sorted(self.layout.proposals_dir.glob("*.json"))
        ]
        return sorted(artifacts, key=lambda item: item.created_at, reverse=True)

    def list_analyses(self) -> list[RunAnalysisArtifact]:
        if not self.layout.analyses_dir.exists():
            return []
        artifacts = [
            RunAnalysisArtifact.model_validate_json(path.read_text(encoding="utf-8"))
            for path in sorted(self.layout.analyses_dir.glob("*.json"))
        ]
        return sorted(artifacts, key=lambda item: item.created_at, reverse=True)

    def append_approval(self, event: ApprovalLogEvent) -> Path:
        payload = event.model_dump(mode="json")
        self._redactor.assert_safe(payload, label="approval event")
        self.layout.ensure_workspace()
        self.layout.approval_log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.layout.approval_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, separators=(",", ":"), ensure_ascii=False))
            handle.write("\n")
        return self.layout.approval_log_path

    def list_approval_events(self, *, proposal_id: str | None = None) -> list[ApprovalLogEvent]:
        if not self.layout.approval_log_path.exists():
            return []
        events: list[ApprovalLogEvent] = []
        for line in self.layout.approval_log_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            event = ApprovalLogEvent.model_validate_json(line)
            if proposal_id is not None and event.proposal_id != proposal_id:
                continue
            events.append(event)
        return events

    def _write_json_artifact(self, path: Path, artifact: SuiteProposalArtifact | RunAnalysisArtifact) -> Path:
        payload = artifact.model_dump(mode="json")
        self._redactor.assert_safe(payload, label=f"{artifact.artifact_type} artifact")
        self.layout.ensure_workspace()
        path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return path


__all__ = [
    "AIArtifactStore",
    "AIWorkspaceLayout",
]
