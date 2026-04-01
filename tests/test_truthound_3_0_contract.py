from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

import truthound as th
from truthound.cli import app
from truthound.context import TruthoundContext, TruthoundContextConfig
from truthound.core.results import ValidationRunResult

pytestmark = pytest.mark.contract


def test_get_context_creates_zero_config_workspace(tmp_path: Path):
    context = th.get_context(tmp_path)

    assert context.workspace_dir == tmp_path / ".truthound"
    assert context.config_path.exists()
    assert context.catalog_dir.exists()
    assert context.baselines_dir.exists()
    assert context.runs_dir.exists()
    assert context.docs_dir.exists()
    assert context.plugins_dir.exists()
    assert not (context.workspace_dir / "ai").exists()


def test_check_uses_context_and_persists_zero_config_artifacts(tmp_path: Path):
    context = TruthoundContext(tmp_path)

    run_result = th.check(
        {"customer_id": [1, 2, 2], "email": ["a@example.com", None, "c@example.com"]},
        context=context,
    )

    assert isinstance(run_result, ValidationRunResult)
    assert run_result.metadata["context_root"] == str(tmp_path)
    assert run_result.metadata["context_source_key"] == "dict:customer_id:email"
    assert run_result.metadata["context_history_key"] == "dict:customer_id:email"
    assert isinstance(run_result.metadata["context_source_fingerprint"], str)

    run_artifact = Path(run_result.metadata["context_run_artifact"])
    docs_artifact = Path(run_result.metadata["context_docs_artifact"])

    assert run_artifact.exists()
    assert docs_artifact.exists()
    assert run_artifact.parent == context.runs_dir
    assert docs_artifact.parent == context.docs_dir
    assert not (context.workspace_dir / "ai").exists()

    baseline_index = json.loads(context.baseline_index_path.read_text(encoding="utf-8"))
    assert baseline_index


def test_metric_history_is_bounded_for_repeated_zero_config_runs(tmp_path: Path):
    context = TruthoundContext(
        tmp_path,
        config=TruthoundContextConfig(max_metric_history_entries=3),
    )

    last_run = None
    for _ in range(5):
        last_run = th.check(
            {"customer_id": [1, 2, 2], "email": ["a@example.com", None, "c@example.com"]},
            context=context,
        )

    history_payload = json.loads(
        (context.baselines_dir / "metric-history.json").read_text(encoding="utf-8")
    )
    assert last_run is not None
    assert len(history_payload) == 1
    history_key, source_history = next(iter(history_payload.items()))

    assert history_key == "dict:customer_id:email"
    assert len(source_history) == 3
    assert all("run_id" in entry for entry in source_history)


def test_validation_run_result_helpers_write_and_build_docs(tmp_path: Path):
    context = TruthoundContext(tmp_path)
    run_result = th.check({"id": [1, 2], "status": ["active", "inactive"]}, context=context)

    json_path = tmp_path / "validation-run.json"
    written_path = run_result.write(str(json_path))
    docs_html = run_result.build_docs(title="Validation Overview")

    assert Path(written_path).exists()
    written_payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert written_payload["result"]["run_id"] == run_result.run_id
    assert "Validation Overview" in docs_html
    assert run_result.source in docs_html


def test_doctor_migrate_2_to_3_reports_removed_surfaces(tmp_path: Path):
    project_file = tmp_path / "legacy_usage.py"
    project_file.write_text(
        "\n".join(
            [
                "from truthound import compare",
                "from truthound.report import Report",
                "",
                "def f(checkpoint_result):",
                "    return checkpoint_result.validation_result",
            ]
        ),
        encoding="utf-8",
    )

    runner = CliRunner()
    result = runner.invoke(app, ["doctor", str(tmp_path), "--migrate-2to3"])

    assert result.exit_code == 1
    assert "root-compare-import" in result.output
    assert "legacy-report-import" in result.output
    assert "legacy-checkpoint-field" in result.output


def test_doctor_workspace_reports_healthy_zero_config_layout(tmp_path: Path):
    context = TruthoundContext(tmp_path)
    th.check({"id": [1, 2], "status": ["active", "inactive"]}, context=context)

    runner = CliRunner()
    result = runner.invoke(app, ["doctor", str(tmp_path), "--workspace"])

    assert result.exit_code == 0
    assert "found no structural issues" in result.output


def test_doctor_workspace_reports_invalid_baseline_index(tmp_path: Path):
    context = TruthoundContext(tmp_path)
    context.baseline_index_path.write_text("{not-json", encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(app, ["doctor", str(tmp_path), "--workspace"])

    assert result.exit_code == 1
    assert "baseline-index-invalid" in result.output


def test_doctor_workspace_json_reports_missing_schema_file(tmp_path: Path):
    context = TruthoundContext(tmp_path)
    context.baseline_index_path.write_text(
        json.dumps(
            {
                "source:dict": {
                    "schema_file": "missing.schema.yaml",
                    "created_at": "2026-03-20T00:00:00",
                }
            }
        ),
        encoding="utf-8",
    )

    runner = CliRunner()
    result = runner.invoke(
        app,
        ["doctor", str(tmp_path), "--workspace", "--format", "json"],
    )

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["workspace_dir"] == str(tmp_path / ".truthound")
    assert any(issue["rule_id"] == "baseline-entry-missing-schema" for issue in payload["issues"])
