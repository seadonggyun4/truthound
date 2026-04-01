from __future__ import annotations

import os
from pathlib import Path

import pytest

pytest.importorskip("pydantic")


def _write_result_if_requested(result) -> None:
    result_path = os.getenv("TRUTHOUND_AI_SMOKE_RESULT_PATH")
    if not result_path:
        return
    path = Path(result_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(result.model_dump_json(indent=2), encoding="utf-8")


@pytest.mark.integration
def test_openai_live_smoke_end_to_end():
    from truthound.ai import run_openai_smoke

    if os.getenv("TRUTHOUND_AI_RUN_LIVE_SMOKE") != "1":
        pytest.skip("TRUTHOUND_AI_RUN_LIVE_SMOKE=1 is required for the live OpenAI smoke test.")
    if not os.getenv("OPENAI_API_KEY", "").strip():
        pytest.skip("OPENAI_API_KEY is required for the live OpenAI smoke test.")
    if not (
        os.getenv("TRUTHOUND_AI_SMOKE_MODEL", "").strip()
        or os.getenv("TRUTHOUND_AI_MODEL", "").strip()
    ):
        pytest.skip("TRUTHOUND_AI_SMOKE_MODEL or TRUTHOUND_AI_MODEL is required for the live smoke test.")

    keep_workspace = os.getenv("TRUTHOUND_AI_SMOKE_KEEP_WORKSPACE") == "1"
    result = run_openai_smoke(keep_workspace=keep_workspace)
    _write_result_if_requested(result)

    assert result.success is True, result.model_dump_json(indent=2)
    assert result.compile_status in {"ready", "partial"}
    assert result.compiled_check_count >= 1
    assert result.artifact_id is not None
    assert result.proposal_path is not None
