"""AI CLI commands for suite proposal generation and review."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Optional

import typer

from truthound.cli_modules.common.datasource import (
    ConnectionOpt,
    QueryOpt,
    SourceConfigOpt,
    SourceNameOpt,
    TableOpt,
    resolve_datasource,
)
from truthound.cli_modules.common.errors import DependencyError, error_boundary
from truthound.context import get_context

app = typer.Typer(
    name="ai",
    help="AI-assisted review commands",
    no_args_is_help=True,
)
smoke_app = typer.Typer(
    name="smoke",
    help="Live smoke verification commands",
    no_args_is_help=True,
)
analyses_app = typer.Typer(
    name="analyses",
    help="Read-only run analysis review commands",
    no_args_is_help=True,
)
proposals_app = typer.Typer(
    name="proposals",
    help="Proposal review and lifecycle commands",
    no_args_is_help=True,
)


def register_commands(parent_app: typer.Typer) -> None:
    parent_app.add_typer(app, name="ai")


def _require_ai_namespace():
    try:
        import truthound.ai as ai_namespace
    except ImportError as exc:  # pragma: no cover - exercised in CLI import contract tests
        raise DependencyError(
            "truthound[ai]",
            install_command="pip install truthound[ai]",
            hint=str(exc),
        ) from exc
    return ai_namespace


def _actor_ref(ai_namespace, *, actor_id: str, actor_name: str):
    return ai_namespace.ActorRef(
        actor_id=actor_id,
        actor_name=actor_name,
    )


@app.command("suggest-suite")
@error_boundary
def suggest_suite_cmd(
    prompt: Annotated[
        str,
        typer.Option("--prompt", help="Natural-language prompt describing the desired checks"),
    ],
    file: Annotated[
        Optional[Path],
        typer.Argument(help="Path to the data file (CSV, JSON, Parquet, NDJSON)"),
    ] = None,
    connection: ConnectionOpt = None,
    table: TableOpt = None,
    query: QueryOpt = None,
    source_config: SourceConfigOpt = None,
    source_name: SourceNameOpt = None,
    provider: Annotated[
        Optional[str],
        typer.Option("--provider", help="AI provider name (Phase 1 supports openai)"),
    ] = None,
    model: Annotated[
        Optional[str],
        typer.Option("--model", help="AI model name"),
    ] = None,
    sample_size: Annotated[
        int,
        typer.Option("--sample-size", help="Local summary budget used for prompt compilation"),
    ] = 1000,
    redact: Annotated[
        str,
        typer.Option("--redact", help="Redaction mode (Phase 1 supports summary_only)"),
    ] = "summary_only",
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Output the full proposal artifact as JSON"),
    ] = False,
) -> None:
    ai_namespace = _require_ai_namespace()
    data_path, source = resolve_datasource(
        file=file,
        connection=connection,
        table=table,
        query=query,
        source_config=source_config,
        source_name=source_name,
    )

    provider_config = None
    if provider:
        provider_config = ai_namespace.ProviderConfig(
            provider_name=provider,
            model_name=model,
        )

    artifact = ai_namespace.suggest_suite(
        prompt=prompt,
        data=data_path,
        source=source,
        context=get_context(),
        provider=provider_config,
        model=model,
        sample_size=sample_size,
        redact=redact,
    )
    if json_output:
        typer.echo(artifact.model_dump_json(indent=2))
        return

    typer.echo(f"artifact_id: {artifact.artifact_id}")
    typer.echo(f"compile_status: {artifact.compile_status}")
    typer.echo(f"compiled_check_count: {artifact.compiled_check_count}")
    typer.echo(f"rejected_check_count: {artifact.rejected_check_count}")
    typer.echo(f"added_count: {artifact.diff_preview.counts.added}")
    typer.echo(f"already_present_count: {artifact.diff_preview.counts.already_present}")
    typer.echo(f"conflict_count: {artifact.diff_preview.counts.conflicts}")


@app.command("explain-run")
@error_boundary
def explain_run_cmd(
    run_id: Annotated[
        str,
        typer.Option("--run-id", help="Persisted ValidationRunResult run identifier"),
    ],
    include_history: Annotated[
        bool,
        typer.Option("--history/--no-history", help="Include read-only metric history context"),
    ] = True,
    provider: Annotated[
        Optional[str],
        typer.Option("--provider", help="AI provider name (Phase 2 supports openai)"),
    ] = None,
    model: Annotated[
        Optional[str],
        typer.Option("--model", help="AI model name"),
    ] = None,
    redact: Annotated[
        str,
        typer.Option("--redact", help="Redaction mode (Phase 2 supports summary_only)"),
    ] = "summary_only",
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Output the full analysis artifact as JSON"),
    ] = False,
) -> None:
    ai_namespace = _require_ai_namespace()
    provider_config = None
    if provider:
        provider_config = ai_namespace.ProviderConfig(
            provider_name=provider,
            model_name=model,
        )

    artifact = ai_namespace.explain_run(
        run_id=run_id,
        context=get_context(),
        include_history=include_history,
        provider=provider_config,
        model=model,
        redact=redact,
    )
    if json_output:
        typer.echo(artifact.model_dump_json(indent=2))
        return

    typer.echo(f"artifact_id: {artifact.artifact_id}")
    typer.echo(f"run_id: {artifact.run_id}")
    typer.echo(f"failed_check_count: {len(artifact.failed_checks)}")
    typer.echo(f"top_column_count: {len(artifact.top_columns)}")
    typer.echo(f"evidence_ref_count: {len(artifact.evidence_refs)}")


@smoke_app.command("openai")
@error_boundary
def openai_smoke_cmd(
    model: Annotated[
        Optional[str],
        typer.Option("--model", help="OpenAI model name for the live smoke run"),
    ] = None,
    base_url: Annotated[
        Optional[str],
        typer.Option("--base-url", help="Optional OpenAI-compatible base URL override"),
    ] = None,
    timeout_seconds: Annotated[
        Optional[float],
        typer.Option("--timeout-seconds", help="Optional timeout override in seconds"),
    ] = None,
    keep_workspace: Annotated[
        bool,
        typer.Option("--keep-workspace", help="Retain the temporary smoke workspace on success"),
    ] = False,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Output the full smoke result as JSON"),
    ] = False,
) -> None:
    ai_namespace = _require_ai_namespace()
    result = ai_namespace.run_openai_smoke(
        model=model,
        base_url=base_url,
        timeout_seconds=timeout_seconds,
        keep_workspace=keep_workspace,
    )
    if json_output:
        typer.echo(result.model_dump_json(indent=2))
        if not result.success:
            raise typer.Exit(1)
        return

    typer.echo(f"success: {result.success}")
    typer.echo(f"provider_name: {result.provider_name}")
    typer.echo(f"model_name: {result.model_name or 'unknown'}")
    typer.echo(f"artifact_id: {result.artifact_id or 'none'}")
    typer.echo(f"compile_status: {result.compile_status or 'none'}")
    typer.echo(
        f"compiled_check_count: {result.compiled_check_count} | "
        f"rejected_check_count: {result.rejected_check_count}"
    )
    if result.proposal_path:
        typer.echo(f"proposal_path: {result.proposal_path}")
    typer.echo(f"workspace_retained: {result.workspace_retained}")
    if result.workspace_retained and result.workspace_dir:
        typer.echo(f"workspace_dir: {result.workspace_dir}")
    if not result.success:
        typer.echo(f"failure_stage: {result.failure_stage or 'unknown'}")
        typer.echo(f"error_message: {result.error_message or 'unknown error'}")
        raise typer.Exit(1)


@smoke_app.command("openai-explain-run")
@error_boundary
def openai_explain_run_smoke_cmd(
    model: Annotated[
        Optional[str],
        typer.Option("--model", help="OpenAI model name for the live explain-run smoke"),
    ] = None,
    base_url: Annotated[
        Optional[str],
        typer.Option("--base-url", help="Optional OpenAI-compatible base URL override"),
    ] = None,
    timeout_seconds: Annotated[
        Optional[float],
        typer.Option("--timeout-seconds", help="Optional timeout override in seconds"),
    ] = None,
    keep_workspace: Annotated[
        bool,
        typer.Option("--keep-workspace", help="Retain the temporary explain-run smoke workspace on success"),
    ] = False,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Output the full explain-run smoke result as JSON"),
    ] = False,
) -> None:
    ai_namespace = _require_ai_namespace()
    result = ai_namespace.run_openai_explain_run_smoke(
        model=model,
        base_url=base_url,
        timeout_seconds=timeout_seconds,
        keep_workspace=keep_workspace,
    )
    if json_output:
        typer.echo(result.model_dump_json(indent=2))
        if not result.success:
            raise typer.Exit(1)
        return

    typer.echo(f"success: {result.success}")
    typer.echo(f"provider_name: {result.provider_name}")
    typer.echo(f"model_name: {result.model_name or 'unknown'}")
    typer.echo(f"run_id: {result.run_id or 'none'}")
    typer.echo(f"artifact_id: {result.artifact_id or 'none'}")
    typer.echo(
        f"failed_check_count: {result.failed_check_count} | "
        f"top_column_count: {result.top_column_count} | "
        f"evidence_ref_count: {result.evidence_ref_count}"
    )
    if result.analysis_path:
        typer.echo(f"analysis_path: {result.analysis_path}")
    typer.echo(f"workspace_retained: {result.workspace_retained}")
    if result.workspace_retained and result.workspace_dir:
        typer.echo(f"workspace_dir: {result.workspace_dir}")
    if not result.success:
        typer.echo(f"failure_stage: {result.failure_stage or 'unknown'}")
        typer.echo(f"error_message: {result.error_message or 'unknown error'}")
        raise typer.Exit(1)


@proposals_app.command("list")
@error_boundary
def list_proposals_cmd(
    source_key: Annotated[
        Optional[str],
        typer.Option("--source-key", help="Filter proposals by source key"),
    ] = None,
    compile_status: Annotated[
        Optional[str],
        typer.Option("--compile-status", help="Filter proposals by compile status"),
    ] = None,
    limit: Annotated[
        Optional[int],
        typer.Option("--limit", help="Maximum number of proposals to show"),
    ] = None,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Output the proposal list as JSON"),
    ] = False,
) -> None:
    ai_namespace = _require_ai_namespace()
    proposals = ai_namespace.list_proposals(
        context=get_context(),
        source_key=source_key,
        compile_status=compile_status,
        limit=limit,
    )
    if json_output:
        typer.echo(
            json.dumps(
                [item.model_dump(mode="json") for item in proposals],
                indent=2,
                ensure_ascii=False,
            )
        )
        return

    if not proposals:
        typer.echo("No AI proposals found.")
        return

    for item in proposals:
        typer.echo(
            " | ".join(
                [
                    item.artifact_id,
                    f"approval={item.approval_status}",
                    f"status={item.compile_status}",
                    f"compiled={item.compiled_check_count}",
                    f"rejected={item.rejected_check_count}",
                    f"added={item.diff_preview.counts.added}",
                    f"already_present={item.diff_preview.counts.already_present}",
                    f"conflicts={item.diff_preview.counts.conflicts}",
                ]
            )
        )


@proposals_app.command("show")
@error_boundary
def show_proposal_cmd(
    proposal_id: Annotated[str, typer.Argument(help="Proposal artifact identifier")],
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Output the proposal artifact as JSON"),
    ] = False,
) -> None:
    ai_namespace = _require_ai_namespace()
    artifact = ai_namespace.show_proposal(proposal_id, context=get_context())
    if json_output:
        typer.echo(artifact.model_dump_json(indent=2))
        return

    typer.echo(f"artifact_id: {artifact.artifact_id}")
    typer.echo(f"approval_status: {artifact.approval_status}")
    typer.echo(f"compile_status: {artifact.compile_status}")
    typer.echo(f"summary: {artifact.summary}")
    typer.echo(f"compiled_check_count: {artifact.compiled_check_count}")
    typer.echo(f"rejected_check_count: {artifact.rejected_check_count}")
    typer.echo(f"added_count: {artifact.diff_preview.counts.added}")
    typer.echo(f"already_present_count: {artifact.diff_preview.counts.already_present}")
    typer.echo(f"conflict_count: {artifact.diff_preview.counts.conflicts}")


@proposals_app.command("approve")
@error_boundary
def approve_proposal_cmd(
    proposal_id: Annotated[str, typer.Argument(help="Proposal artifact identifier")],
    actor_id: Annotated[
        str,
        typer.Option("--actor-id", help="Actor identifier performing the review action"),
    ],
    actor_name: Annotated[
        str,
        typer.Option("--actor-name", help="Actor display name performing the review action"),
    ],
    comment: Annotated[
        str,
        typer.Option("--comment", help="Approval comment stored in the review history"),
    ],
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Output the decision result as JSON"),
    ] = False,
) -> None:
    ai_namespace = _require_ai_namespace()
    result = ai_namespace.approve_proposal(
        proposal_id,
        actor=_actor_ref(ai_namespace, actor_id=actor_id, actor_name=actor_name),
        comment=comment,
        context=get_context(),
    )
    if json_output:
        typer.echo(result.model_dump_json(indent=2))
        return

    typer.echo(f"proposal_id: {result.proposal.artifact_id}")
    typer.echo(f"approval_status: {result.proposal.approval_status}")
    typer.echo(f"changed: {result.changed}")
    typer.echo(f"event_id: {result.event.event_id if result.event else 'none'}")


@proposals_app.command("reject")
@error_boundary
def reject_proposal_cmd(
    proposal_id: Annotated[str, typer.Argument(help="Proposal artifact identifier")],
    actor_id: Annotated[
        str,
        typer.Option("--actor-id", help="Actor identifier performing the review action"),
    ],
    actor_name: Annotated[
        str,
        typer.Option("--actor-name", help="Actor display name performing the review action"),
    ],
    comment: Annotated[
        str,
        typer.Option("--comment", help="Rejection comment stored in the review history"),
    ],
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Output the decision result as JSON"),
    ] = False,
) -> None:
    ai_namespace = _require_ai_namespace()
    result = ai_namespace.reject_proposal(
        proposal_id,
        actor=_actor_ref(ai_namespace, actor_id=actor_id, actor_name=actor_name),
        comment=comment,
        context=get_context(),
    )
    if json_output:
        typer.echo(result.model_dump_json(indent=2))
        return

    typer.echo(f"proposal_id: {result.proposal.artifact_id}")
    typer.echo(f"approval_status: {result.proposal.approval_status}")
    typer.echo(f"changed: {result.changed}")
    typer.echo(f"event_id: {result.event.event_id if result.event else 'none'}")


@proposals_app.command("apply")
@error_boundary
def apply_proposal_cmd(
    proposal_id: Annotated[str, typer.Argument(help="Proposal artifact identifier")],
    actor_id: Annotated[
        str,
        typer.Option("--actor-id", help="Actor identifier performing the apply action"),
    ],
    actor_name: Annotated[
        str,
        typer.Option("--actor-name", help="Actor display name performing the apply action"),
    ],
    comment: Annotated[
        Optional[str],
        typer.Option("--comment", help="Optional apply comment stored in the review history"),
    ] = None,
    yes: Annotated[
        bool,
        typer.Option("--yes", help="Apply without interactive confirmation"),
    ] = False,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Output the apply result as JSON"),
    ] = False,
) -> None:
    ai_namespace = _require_ai_namespace()
    if not yes:
        typer.confirm(
            f"Apply proposal {proposal_id} to the active validation suite?",
            abort=True,
        )

    result = ai_namespace.apply_proposal(
        proposal_id,
        actor=_actor_ref(ai_namespace, actor_id=actor_id, actor_name=actor_name),
        comment=comment,
        context=get_context(),
    )
    if json_output:
        typer.echo(result.model_dump_json(indent=2))
        return

    typer.echo(f"proposal_id: {result.proposal.artifact_id}")
    typer.echo(f"approval_status: {result.proposal.approval_status}")
    typer.echo(f"changed: {result.changed}")
    typer.echo(f"target: {result.target}")
    typer.echo(f"applied_check_count: {result.applied_check_count}")
    typer.echo(f"event_id: {result.event.event_id if result.event else 'none'}")


@proposals_app.command("history")
@error_boundary
def proposal_history_cmd(
    proposal_id: Annotated[str, typer.Argument(help="Proposal artifact identifier")],
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Output proposal approval history as JSON"),
    ] = False,
) -> None:
    ai_namespace = _require_ai_namespace()
    events = ai_namespace.list_proposal_approval_events(
        proposal_id,
        context=get_context(),
    )
    if json_output:
        typer.echo(
            json.dumps(
                [item.model_dump(mode="json") for item in events],
                indent=2,
                ensure_ascii=False,
            )
        )
        return

    if not events:
        typer.echo("No proposal approval history found.")
        return

    for event in events:
        typer.echo(
            " | ".join(
                [
                    event.event_id,
                    f"action={event.action}",
                    f"actor={event.actor_name}",
                    f"acted_at={event.acted_at.isoformat()}",
                    f"diff_hash={event.diff_hash[:12]}",
                ]
            )
        )


@analyses_app.command("list")
@error_boundary
def list_analyses_cmd(
    source_key: Annotated[
        Optional[str],
        typer.Option("--source-key", help="Filter analyses by source key"),
    ] = None,
    run_id: Annotated[
        Optional[str],
        typer.Option("--run-id", help="Filter analyses by run id"),
    ] = None,
    limit: Annotated[
        Optional[int],
        typer.Option("--limit", help="Maximum number of analyses to show"),
    ] = None,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Output the analysis list as JSON"),
    ] = False,
) -> None:
    ai_namespace = _require_ai_namespace()
    analyses = ai_namespace.list_analyses(
        context=get_context(),
        source_key=source_key,
        run_id=run_id,
        limit=limit,
    )
    if json_output:
        typer.echo(
            json.dumps(
                [item.model_dump(mode="json") for item in analyses],
                indent=2,
                ensure_ascii=False,
            )
        )
        return

    if not analyses:
        typer.echo("No AI run analyses found.")
        return

    for item in analyses:
        typer.echo(
            " | ".join(
                [
                    item.artifact_id,
                    f"run_id={item.run_id}",
                    f"failed_checks={len(item.failed_checks)}",
                    f"top_columns={len(item.top_columns)}",
                    f"evidence_refs={len(item.evidence_refs)}",
                ]
            )
        )


@analyses_app.command("show")
@error_boundary
def show_analysis_cmd(
    run_id_or_artifact_id: Annotated[str, typer.Argument(help="Run id or analysis artifact identifier")],
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Output the run analysis artifact as JSON"),
    ] = False,
) -> None:
    ai_namespace = _require_ai_namespace()
    artifact = ai_namespace.show_analysis(run_id_or_artifact_id, context=get_context())
    if json_output:
        typer.echo(artifact.model_dump_json(indent=2))
        return

    typer.echo(f"artifact_id: {artifact.artifact_id}")
    typer.echo(f"run_id: {artifact.run_id}")
    typer.echo(f"summary: {artifact.summary}")
    typer.echo(f"failed_check_count: {len(artifact.failed_checks)}")
    typer.echo(f"top_column_count: {len(artifact.top_columns)}")
    typer.echo(f"evidence_ref_count: {len(artifact.evidence_refs)}")


app.add_typer(analyses_app, name="analyses")
app.add_typer(smoke_app, name="smoke")
app.add_typer(proposals_app, name="proposals")


__all__ = [
    "app",
    "analyses_app",
    "explain_run_cmd",
    "list_analyses_cmd",
    "show_analysis_cmd",
    "openai_explain_run_smoke_cmd",
    "openai_smoke_cmd",
    "proposal_history_cmd",
    "proposals_app",
    "approve_proposal_cmd",
    "apply_proposal_cmd",
    "register_commands",
    "reject_proposal_cmd",
    "smoke_app",
    "show_proposal_cmd",
    "list_proposals_cmd",
    "suggest_suite_cmd",
]
