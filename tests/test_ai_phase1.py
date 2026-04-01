from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest
from typer.testing import CliRunner

from truthound.cli import app
from truthound.context import TruthoundContext

pytest.importorskip("pydantic")

pytestmark = pytest.mark.contract


def _make_diff_preview():
    from truthound.ai import (
        SuiteCheckSnapshot,
        ValidationSuiteDiffCounts,
        ValidationSuiteDiffPreview,
        ValidationSuiteSnapshot,
    )

    current_check = SuiteCheckSnapshot(
        check_id="null",
        check_key="null||{}",
        validator_name="null",
        category="completeness",
        columns=[],
        params={},
        tags=["completeness"],
        rationale="Always validate completeness across discovered columns.",
        origin="current",
    )
    added_check = SuiteCheckSnapshot(
        check_id="unique|order_id|{}",
        check_key="unique|order_id|{}",
        validator_name="unique",
        category="uniqueness",
        columns=["order_id"],
        params={},
        tags=["uniqueness"],
        rationale="Order ids should remain unique.",
        origin="proposal",
    )
    return ValidationSuiteDiffPreview(
        current_suite=ValidationSuiteSnapshot(
            suite_name="truthound-auto-suite",
            check_count=1,
            schema_check_present=False,
            evidence_mode="summary",
            min_severity=None,
            checks=[current_check],
        ),
        proposed_suite=ValidationSuiteSnapshot(
            suite_name="truthound-auto-suite",
            check_count=2,
            schema_check_present=False,
            evidence_mode="summary",
            min_severity=None,
            checks=[current_check, added_check],
        ),
        added=[added_check],
        counts=ValidationSuiteDiffCounts(
            added=1,
            already_present=0,
            conflicts=0,
            rejected=0,
        ),
    )


class FakeProvider:
    provider_name = "fake-openai"
    api_key_env = None
    supports_structured_outputs = True
    default_model_name = "gpt-fake"

    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload
        self.requests = []

    def generate_structured(self, request):
        from truthound.ai import StructuredProviderResponse

        self.requests.append(request)
        return StructuredProviderResponse(
            provider_name=self.provider_name,
            model_name=request.model_name,
            output_text=json.dumps(self.payload, ensure_ascii=False),
            parsed_output=self.payload,
            usage={"prompt_tokens": 11, "completion_tokens": 22, "total_tokens": 33},
            finish_reason="stop",
        )


def _write_orders_csv(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "order_id,refund_rate,status,customer_email,sku_code,order_url,support_phone,uuid_value",
                "1,0.10,pending,alice@example.com,SKU-001,https://example.com/a,+82-10-1111-2222,550e8400-e29b-41d4-a716-446655440000",
                "2,0.25,approved,bob@example.com,SKU-002,https://example.com/b,+82-10-3333-4444,550e8400-e29b-41d4-a716-446655440001",
                "3,0.00,pending,charlie@example.com,SKU-003,https://example.com/c,+82-10-5555-6666,550e8400-e29b-41d4-a716-446655440002",
            ]
        ),
        encoding="utf-8",
    )


def _snapshot_core_state(context: TruthoundContext) -> dict[str, str]:
    snapshot: dict[str, str] = {}
    for directory in (context.catalog_dir, context.baselines_dir, context.runs_dir, context.docs_dir):
        for path in sorted(directory.rglob("*")):
            if path.is_file():
                snapshot[str(path.relative_to(context.workspace_dir))] = path.read_text(encoding="utf-8")
    return snapshot


def _make_cli_artifact(tmp_path: Path):
    from truthound.ai import CompiledProposalCheck, InputRef, SuiteProposalArtifact

    return SuiteProposalArtifact(
        source_key="source:orders",
        input_refs=[
            InputRef(
                kind="schema_summary",
                ref="schema-summary:source:orders",
                hash="schemahash01",
                redacted=True,
                metadata={"column_count": 8, "observed_count": 3},
            )
        ],
        model_provider="fake-openai",
        model_name="gpt-fake",
        prompt_hash="prompt-hash-cli",
        created_by="phase1-test",
        workspace_root=str(tmp_path),
        summary="Order proposals emphasize identifier stability and refund bounds.",
        rationale="These checks cover the aggregate risks mentioned by the operator.",
        checks=[
            CompiledProposalCheck(
                check_key='unique|order_id|{}',
                validator_name="unique",
                category="uniqueness",
                columns=["order_id"],
                params={},
                rationale="Order ids should remain unique.",
            )
        ],
        risks=["Thresholds may need tuning during seasonal periods."],
        compile_status="ready",
        diff_preview=_make_diff_preview(),
        compiled_check_count=1,
        rejected_check_count=0,
        compiler_errors=[],
    )


def test_suggest_suite_compiles_curated_intents_and_keeps_core_state_read_only(tmp_path: Path):
    from truthound.ai import suggest_suite

    data_path = tmp_path / "orders.csv"
    _write_orders_csv(data_path)
    context = TruthoundContext(tmp_path)
    before = _snapshot_core_state(context)

    provider = FakeProvider(
        {
            "summary": "Compile aggregate controls for order identifiers, refund signals, and basic string quality.",
            "rationale": "The operator asked for identifier stability, aggregate bounds, and business-status guardrails.",
            "proposed_checks": [
                {"intent": "null", "columns": [], "params": {}, "rationale": "Retain global null coverage."},
                {"intent": "not_null", "columns": ["customer_email"], "params": {}, "rationale": "Customer email should be present."},
                {"intent": "completeness_ratio", "columns": ["status"], "params": {"min_ratio": 0.95}, "rationale": "Status should stay mostly populated."},
                {"intent": "unique", "columns": ["order_id"], "params": {}, "rationale": "Order ids should remain unique."},
                {"intent": "unique_ratio", "columns": ["customer_email"], "params": {"min_ratio": 0.95}, "rationale": "Customer email should stay almost unique."},
                {"intent": "between", "columns": ["refund_rate"], "params": {"min_value": 0, "max_value": 1}, "rationale": "Refund rate should stay between zero and one."},
                {"intent": "in_set", "columns": ["status"], "params": {"allowed_values": ["pending", "approved"]}, "rationale": "Status should stay in the approved business states."},
                {"intent": "length", "columns": ["sku_code"], "params": {"min_length": 7, "max_length": 7}, "rationale": "Sku codes follow a fixed-width token."},
                {"intent": "format", "columns": ["customer_email"], "params": {"format": "email"}, "rationale": "Customer email should follow email format."},
                {"intent": "regex", "columns": ["sku_code"], "params": {"pattern": "^SKU-[0-9]{3}$"}, "rationale": "Sku codes should follow the SKU-000 pattern."},
                {"intent": "mean_between", "columns": ["refund_rate"], "params": {"min_value": 0, "max_value": 1}, "rationale": "Mean refund rate should stay bounded."},
                {"intent": "sum_between", "columns": ["refund_rate"], "params": {"min_value": 0, "max_value": 3}, "rationale": "Aggregate refund rate should stay within a conservative bound."},
            ],
            "risks": ["Thresholds may need tuning when traffic patterns change."],
            "rejected_requests": [],
        }
    )

    artifact = suggest_suite(
        prompt="Keep order ids unique and refund metrics bounded.",
        data=str(data_path),
        context=context,
        provider=provider,
        sample_size=50,
    )

    assert str(artifact.compile_status) == "ready"
    assert artifact.compiled_check_count == 12
    assert artifact.rejected_check_count == 0
    assert artifact.existing_suite_summary is not None
    assert artifact.existing_suite_summary["check_count"] >= 1
    assert artifact.diff_preview.current_suite.check_count == artifact.existing_suite_summary["check_count"]
    assert artifact.diff_preview.proposed_suite.check_count == (
        artifact.diff_preview.current_suite.check_count + artifact.diff_preview.counts.added
    )
    assert artifact.diff_preview.counts.already_present >= 1
    assert {check.validator_name for check in artifact.checks} >= {"unique", "between", "regex", "mean_between", "sum_between"}
    assert provider.requests
    request = provider.requests[0]
    assert "alice@example.com" not in request.user_prompt
    assert "sample_values" not in request.user_prompt
    assert _snapshot_core_state(context) == before


def test_suggest_suite_records_partial_compile_and_rejected_items(tmp_path: Path):
    from truthound.ai import suggest_suite

    data_path = tmp_path / "orders.csv"
    _write_orders_csv(data_path)
    context = TruthoundContext(tmp_path)
    provider = FakeProvider(
        {
            "summary": "Compile a conservative reviewable proposal.",
            "rationale": "One request is supported and two should be rejected safely.",
            "proposed_checks": [
                {"intent": "unique", "columns": ["order_id"], "params": {}, "rationale": "Order ids should remain unique."},
                {"intent": "unsupported_magic", "columns": ["order_id"], "params": {}, "rationale": "This intent does not exist."},
                {"intent": "regex", "columns": ["sku_code"], "params": {"pattern": "(a+)+"}, "rationale": "Unsafe regex should be rejected."},
            ],
            "risks": [],
            "rejected_requests": ["Skip automatic apply"],
        }
    )

    artifact = suggest_suite(
        prompt="Keep order ids unique.",
        data=str(data_path),
        context=context,
        provider=provider,
    )

    assert str(artifact.compile_status) == "partial"
    assert artifact.compiled_check_count == 1
    assert artifact.rejected_check_count == 3
    assert artifact.diff_preview.counts.added == 1
    assert artifact.diff_preview.counts.already_present == 0
    reasons = {item.reason for item in artifact.rejected_items}
    assert any("unsupported intent" in reason for reason in reasons)
    assert any("regex pattern is not allowed" in reason for reason in reasons)
    assert any(item.source == "model" for item in artifact.rejected_items)


def test_formal_suite_diff_marks_conflicts_against_current_suite():
    from truthound.ai import CompiledProposalCheck
    from truthound.ai.suite_diff import build_formal_suite_diff
    from truthound.core.suite import ValidationSuite

    current_suite = ValidationSuite(
        name="current-suite",
        checks=(
            CompiledProposalCheck(
                check_key="between|refund_rate|{\"inclusive\":true,\"max_value\":1,\"min_value\":0}",
                validator_name="between",
                category="distribution",
                columns=["refund_rate"],
                params={"min_value": 0, "max_value": 1, "inclusive": True},
                rationale="Current refund bounds.",
            ).to_check_spec(),
        ),
    )
    proposed_check = CompiledProposalCheck(
        check_key="between|refund_rate|{\"inclusive\":true,\"max_value\":0.5,\"min_value\":0}",
        validator_name="between",
        category="distribution",
        columns=["refund_rate"],
        params={"min_value": 0, "max_value": 0.5, "inclusive": True},
        rationale="Proposed tighter refund bounds.",
    )

    result = build_formal_suite_diff(
        current_suite=current_suite,
        compiled_checks=[proposed_check],
        rejected_items=[],
    )

    assert len(result.compiled_checks) == 1
    assert result.diff_preview.counts.conflicts == 1
    assert result.diff_preview.proposed_suite.check_count == result.diff_preview.current_suite.check_count
    assert result.diff_preview.conflicts[0].proposed.check_key == proposed_check.check_key
    assert result.diff_preview.conflicts[0].existing.check_key != proposed_check.check_key


def test_suggest_suite_persists_rejected_artifact_for_malformed_provider_output(tmp_path: Path):
    from truthound.ai import AIArtifactStore, suggest_suite

    data_path = tmp_path / "orders.csv"
    _write_orders_csv(data_path)
    context = TruthoundContext(tmp_path)
    provider = FakeProvider(
        {
            "summary": "This payload is incomplete and should be rejected.",
        }
    )

    artifact = suggest_suite(
        prompt="Compile a proposal.",
        data=str(data_path),
        context=context,
        provider=provider,
    )

    assert str(artifact.compile_status) == "rejected"
    assert artifact.compiler_errors == ["provider_output_validation_failed"]
    stored = AIArtifactStore(context).read_proposal(artifact.artifact_id)
    assert str(stored.compile_status) == "rejected"


def test_legacy_phase1_v1_proposal_is_upgraded_on_read_and_doctor_accepts_it(tmp_path: Path):
    from truthound.ai import AIArtifactStore
    from truthound._ai_contract import TRUTHOUND_AI_PROPOSAL_COMPILER_VERSION_V1

    context = TruthoundContext(tmp_path)
    proposal_dir = context.workspace_dir / "ai" / "proposals"
    proposal_dir.mkdir(parents=True)
    (context.workspace_dir / "ai" / "analyses").mkdir(parents=True)
    (context.workspace_dir / "ai" / "approvals").mkdir(parents=True)
    proposal_id = "suite-proposal-20260401120000-abcdef"
    proposal_path = proposal_dir / f"{proposal_id}.json"
    proposal_path.write_text(
        json.dumps(
            {
                "schema_version": "1",
                "artifact_id": proposal_id,
                "artifact_type": "suite_proposal",
                "source_key": "source:orders",
                "input_refs": [],
                "model_provider": "openai",
                "model_name": "gpt-4o-mini",
                "prompt_hash": "legacy-hash-001",
                "compiler_version": TRUTHOUND_AI_PROPOSAL_COMPILER_VERSION_V1,
                "approval_status": "pending",
                "approved_by": None,
                "approved_at": None,
                "redaction_policy": {
                    "mode": "summary_only",
                    "raw_samples_allowed": False,
                    "pii_literals_allowed": False,
                },
                "created_at": "2026-04-01T00:00:00+00:00",
                "created_by": "phase1-test",
                "workspace_root": str(tmp_path),
                "target_type": "validation_suite",
                "summary": "Legacy proposal artifact.",
                "rationale": "Legacy diff preview shape should still read.",
                "checks": [
                    {
                        "check_key": "unique|order_id|{}",
                        "validator_name": "unique",
                        "category": "uniqueness",
                        "columns": ["order_id"],
                        "params": {},
                        "rationale": "Order ids should remain unique.",
                    }
                ],
                "risks": [],
                "compile_status": "ready",
                "diff_preview": {
                    "added": [{"check_key": "unique|order_id|{}"}],
                    "already_present": [],
                    "conflicts": [],
                    "rejected": [],
                    "counts": {"added": 1, "already_present": 0, "conflicts": 0, "rejected": 0},
                },
                "existing_suite_summary": {
                    "suite_name": "truthound-auto-suite",
                    "check_count": 0,
                    "checks": [],
                },
                "compiled_check_count": 1,
                "rejected_check_count": 0,
                "compiler_errors": [],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    artifact = AIArtifactStore(context).read_proposal(proposal_id)
    assert artifact.compiler_version == TRUTHOUND_AI_PROPOSAL_COMPILER_VERSION_V1
    assert artifact.diff_preview.current_suite.check_count == 0
    assert artifact.diff_preview.proposed_suite.check_count == 1
    assert artifact.diff_preview.counts.added == 1

    runner = CliRunner()
    result = runner.invoke(app, ["doctor", str(tmp_path), "--workspace"])
    assert result.exit_code == 0
    assert "found no structural issues" in result.output


def test_openai_provider_parses_chat_completion_response(monkeypatch):
    from truthound.ai import ProviderConfig, StructuredProviderRequest
    from truthound.ai.providers import OpenAIStructuredProvider

    def fake_create(**kwargs):
        message = types.SimpleNamespace(
            content='{"summary":"Compiled aggregate checks.","rationale":"Safe output.","proposed_checks":[],"risks":[],"rejected_requests":[]}'
        )
        choice = types.SimpleNamespace(message=message, finish_reason="stop")
        usage = types.SimpleNamespace(prompt_tokens=10, completion_tokens=12, total_tokens=22)
        return types.SimpleNamespace(choices=[choice], usage=usage)

    class FakeOpenAIClient:
        def __init__(self, **kwargs):
            self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=fake_create))

    fake_module = types.SimpleNamespace(OpenAI=FakeOpenAIClient)
    monkeypatch.setitem(sys.modules, "openai", fake_module)

    provider = OpenAIStructuredProvider(
        ProviderConfig(provider_name="openai", model_name="gpt-test"),
    )
    request = StructuredProviderRequest(
        provider_name="openai",
        model_name="gpt-test",
        system_prompt="Return one JSON object with safe aggregate content only.",
        user_prompt="operator_request: compile aggregate checks; source_key: source:orders",
        response_format_name="suite_proposal",
    )

    response = provider.generate_structured(request)

    assert response.provider_name == "openai"
    assert response.parsed_output["summary"] == "Compiled aggregate checks."
    assert response.usage == {"prompt_tokens": 10, "completion_tokens": 12, "total_tokens": 22}


def test_ai_cli_suggest_list_and_show_surface(monkeypatch, tmp_path: Path):
    import truthound.ai as ai_namespace
    from truthound.ai import AIArtifactStore

    data_path = tmp_path / "orders.csv"
    _write_orders_csv(data_path)
    context = TruthoundContext(tmp_path)
    runner = CliRunner()

    def fake_suggest_suite(*args, **kwargs):
        artifact = _make_cli_artifact(tmp_path)
        AIArtifactStore(context).write_proposal(artifact)
        return artifact

    monkeypatch.setattr(ai_namespace, "suggest_suite", fake_suggest_suite)
    monkeypatch.setattr("truthound.cli_modules.ai.get_context", lambda: context)

    suggest_result = runner.invoke(
        app,
        ["ai", "suggest-suite", str(data_path), "--prompt", "keep order ids unique", "--json"],
    )
    assert suggest_result.exit_code == 0
    suggest_payload = json.loads(suggest_result.output)
    assert suggest_payload["artifact_id"].startswith("suite-proposal-")

    list_result = runner.invoke(app, ["ai", "proposals", "list", "--json"])
    assert list_result.exit_code == 0
    list_payload = json.loads(list_result.output)
    assert len(list_payload) == 1

    show_result = runner.invoke(
        app,
        ["ai", "proposals", "show", list_payload[0]["artifact_id"], "--json"],
    )
    assert show_result.exit_code == 0
    show_payload = json.loads(show_result.output)
    assert show_payload["compiled_check_count"] == 1
    assert show_payload["diff_preview"]["current_suite"]["check_count"] == 1
    assert show_payload["diff_preview"]["proposed_suite"]["check_count"] == 2

    list_text_result = runner.invoke(app, ["ai", "proposals", "list"])
    assert list_text_result.exit_code == 0
    assert "added=1" in list_text_result.output
    assert "already_present=0" in list_text_result.output
    assert "conflicts=0" in list_text_result.output

    show_text_result = runner.invoke(app, ["ai", "proposals", "show", list_payload[0]["artifact_id"]])
    assert show_text_result.exit_code == 0
    assert "added_count: 1" in show_text_result.output
    assert "already_present_count: 0" in show_text_result.output
    assert "conflict_count: 0" in show_text_result.output
