"""Read-only run analysis helpers for ``truthound.ai``."""

from __future__ import annotations

from truthound._ai_contract import analysis_artifact_id_for_run
from truthound.ai.models import RunAnalysisArtifact
from truthound.ai.store import AIArtifactStore
from truthound.context import TruthoundContext, get_context


def list_analyses(
    *,
    context: TruthoundContext | None = None,
    source_key: str | None = None,
    run_id: str | None = None,
    limit: int | None = None,
) -> list[RunAnalysisArtifact]:
    active_context = context or get_context()
    analyses = AIArtifactStore(active_context).list_analyses()
    if source_key is not None:
        analyses = [item for item in analyses if item.source_key == source_key]
    if run_id is not None:
        analyses = [item for item in analyses if item.run_id == run_id]
    if limit is not None:
        analyses = analyses[: max(0, int(limit))]
    return analyses


def show_analysis(
    run_id_or_artifact_id: str,
    *,
    context: TruthoundContext | None = None,
) -> RunAnalysisArtifact:
    active_context = context or get_context()
    artifact_id = (
        run_id_or_artifact_id
        if run_id_or_artifact_id.startswith("run-analysis-")
        else analysis_artifact_id_for_run(run_id_or_artifact_id)
    )
    return AIArtifactStore(active_context).read_analysis(artifact_id)


__all__ = [
    "list_analyses",
    "show_analysis",
]
