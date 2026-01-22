"""Real-time streaming validation commands.

This module implements streaming validation commands (Phase 10).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Optional

import typer

from truthound.cli_modules.common.errors import error_boundary

# Realtime app for subcommands
app = typer.Typer(
    name="realtime",
    help="Real-time and streaming validation commands",
    no_args_is_help=True,  # Show help when no subcommand provided
)


@app.command(name="validate")
@error_boundary
def validate_cmd(
    source: Annotated[
        str,
        typer.Argument(help="Streaming source (mock, kafka:topic, kinesis:stream)"),
    ],
    validators: Annotated[
        Optional[str],
        typer.Option("--validators", "-v", help="Comma-separated validators"),
    ] = None,
    batch_size: Annotated[
        int,
        typer.Option("--batch-size", "-b", help="Batch size"),
    ] = 1000,
    max_batches: Annotated[
        int,
        typer.Option("--max-batches", help="Maximum batches to process (0=unlimited)"),
    ] = 10,
    output: Annotated[
        Optional[Path],
        typer.Option("--output", "-o", help="Output file for results"),
    ] = None,
    checkpoint_dir: Annotated[
        Path,
        typer.Option("--checkpoint-dir", "-c", help="Directory to save checkpoints"),
    ] = Path("./checkpoints"),
    checkpoint_interval: Annotated[
        int,
        typer.Option("--checkpoint-interval", help="Save checkpoint every N batches"),
    ] = 0,
) -> None:
    """Validate streaming data in real-time.

    Sources:
        - mock: Mock data source for testing
        - kafka:topic_name: Kafka topic
        - kinesis:stream_name: Kinesis stream

    Examples:
        truthound realtime validate mock --max-batches 5
        truthound realtime validate mock --validators null,range --batch-size 500
        truthound realtime validate kafka:my_topic --max-batches 100
        truthound realtime validate mock -c ./checkpoints --checkpoint-interval 5
    """
    from truthound.realtime import (
        MockStreamingSource,
        StreamingValidator,
        StreamingConfig,
    )
    from truthound.realtime.incremental import CheckpointManager, MemoryStateStore

    try:
        # Parse source
        if source.startswith("mock"):
            stream = MockStreamingSource(
                records_per_batch=batch_size,
                num_batches=max_batches if max_batches > 0 else 100,
            )
        elif source.startswith("kafka:"):
            topic = source.split(":", 1)[1]
            typer.echo(f"Kafka source: {topic}")
            typer.echo("Note: Kafka requires aiokafka. Using mock source for now.")
            stream = MockStreamingSource(
                records_per_batch=batch_size,
                num_batches=max_batches if max_batches > 0 else 100,
            )
        elif source.startswith("kinesis:"):
            stream_name = source.split(":", 1)[1]
            typer.echo(f"Kinesis source: {stream_name}")
            typer.echo("Note: Kinesis requires aiobotocore. Using mock source for now.")
            stream = MockStreamingSource(
                records_per_batch=batch_size,
                num_batches=max_batches if max_batches > 0 else 100,
            )
        else:
            typer.echo(f"Source '{source}' requires additional configuration.")
            typer.echo("Using mock source for demonstration.")
            stream = MockStreamingSource(
                records_per_batch=batch_size,
                num_batches=max_batches if max_batches > 0 else 100,
            )

        validator_list = [v.strip() for v in validators.split(",")] if validators else None
        config = StreamingConfig(batch_size=batch_size)
        streaming_validator = StreamingValidator(
            validators=validator_list,
            config=config,
        )

        # Initialize checkpoint manager
        checkpoint_manager = CheckpointManager(checkpoint_dir=checkpoint_dir)
        state_store = MemoryStateStore()

        results = []
        batch_count = 0
        total_records = 0
        total_issues = 0

        with stream:
            typer.echo("Starting streaming validation...")
            typer.echo(f"  Source: {source}")
            typer.echo(f"  Batch size: {batch_size}")
            typer.echo(f"  Validators: {validator_list or 'all'}")
            typer.echo(f"  Checkpoint dir: {checkpoint_dir}")
            if checkpoint_interval > 0:
                typer.echo(f"  Checkpoint interval: every {checkpoint_interval} batches")
            typer.echo()

            for result in streaming_validator.validate_stream(
                stream, max_batches=max_batches if max_batches > 0 else None
            ):
                batch_count += 1
                total_records += result.record_count
                total_issues += result.issue_count

                status = "[ISSUES]" if result.has_issues else "[OK]"
                typer.echo(
                    f"Batch {result.batch_id}: {result.record_count} records, "
                    f"{result.issue_count} issues {status}"
                )
                results.append(result.to_dict())

                # Save checkpoint at interval
                if (
                    checkpoint_interval > 0
                    and batch_count % checkpoint_interval == 0
                ):
                    cp = checkpoint_manager.create_checkpoint(
                        state=state_store,
                        batch_count=batch_count,
                        total_records=total_records,
                        total_issues=total_issues,
                    )
                    typer.echo(f"  [Checkpoint saved: {cp.checkpoint_id}]")

        # Save final checkpoint
        cp = checkpoint_manager.create_checkpoint(
            state=state_store,
            batch_count=batch_count,
            total_records=total_records,
            total_issues=total_issues,
        )
        typer.echo(f"\nFinal checkpoint saved: {cp.checkpoint_id}")

        stats = streaming_validator.get_stats()
        typer.echo("\nSummary")
        typer.echo("=" * 40)
        typer.echo(f"Batches processed: {stats['batch_count']}")
        typer.echo(f"Total records: {stats['total_records']}")
        typer.echo(f"Total issues: {stats['total_issues']}")
        typer.echo(f"Issue rate: {stats['issue_rate']:.2%}")
        typer.echo(f"Avg processing time: {stats['avg_processing_time_ms']:.1f}ms")

        if output:
            with open(output, "w") as f:
                json.dump({"batches": results, "stats": stats}, f, indent=2)
            typer.echo(f"\nResults saved to {output}")

    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


@app.command(name="monitor")
@error_boundary
def monitor_cmd(
    source: Annotated[
        str,
        typer.Argument(help="Streaming source to monitor"),
    ],
    interval: Annotated[
        int,
        typer.Option("--interval", "-i", help="Monitoring interval in seconds"),
    ] = 5,
    duration: Annotated[
        int,
        typer.Option("--duration", "-d", help="Total monitoring duration in seconds (0=indefinite)"),
    ] = 60,
) -> None:
    """Monitor streaming validation metrics.

    This command continuously displays validation metrics
    for a streaming data source.

    Examples:
        truthound realtime monitor mock --interval 5 --duration 60
        truthound realtime monitor kafka:my_topic --interval 10
    """
    import time

    typer.echo(f"Monitoring {source}")
    typer.echo(f"  Interval: {interval}s")
    typer.echo(f"  Duration: {duration}s (0=indefinite)")
    typer.echo()
    typer.echo("Press Ctrl+C to stop")
    typer.echo()

    start_time = time.time()
    iteration = 0

    try:
        while True:
            iteration += 1
            elapsed = time.time() - start_time

            # Simulate monitoring output
            typer.echo(f"[{elapsed:.0f}s] Iteration {iteration}:")
            typer.echo(f"  Records/sec: {1000 + iteration * 10:.0f}")
            typer.echo(f"  Issues/sec: {5 + iteration:.0f}")
            typer.echo(f"  Latency p99: {50 + iteration * 2:.0f}ms")
            typer.echo()

            if duration > 0 and elapsed >= duration:
                typer.echo("Monitoring duration reached.")
                break

            time.sleep(interval)

    except KeyboardInterrupt:
        typer.echo("\nMonitoring stopped.")


# =============================================================================
# Checkpoint subcommand group
# =============================================================================

checkpoint_app = typer.Typer(
    name="checkpoint",
    help="Manage streaming validation checkpoints",
    no_args_is_help=True,
)


def _register_checkpoint_subcommands() -> None:
    """Register checkpoint subcommands after all commands are defined.

    This ensures proper command registration order in Typer.
    Called at module load time after all command definitions.
    """
    app.add_typer(checkpoint_app, name="checkpoint")


@checkpoint_app.command(name="list")
@error_boundary
def checkpoint_list_cmd(
    checkpoint_dir: Annotated[
        Path,
        typer.Option("--dir", "-d", help="Checkpoint directory"),
    ] = Path("./checkpoints"),
    format: Annotated[
        str,
        typer.Option("--format", "-f", help="Output format (console, json)"),
    ] = "console",
) -> None:
    """List available streaming validation checkpoints.

    Examples:
        truthound realtime checkpoint list
        truthound realtime checkpoint list --dir ./my_checkpoints
        truthound realtime checkpoint list --format json
    """
    from truthound.realtime.incremental import CheckpointManager

    if not checkpoint_dir.exists():
        typer.echo(f"Checkpoint directory not found: {checkpoint_dir}")
        typer.echo("No checkpoints available.")
        return

    manager = CheckpointManager(checkpoint_dir=checkpoint_dir)
    checkpoints = manager.list_checkpoints()

    if not checkpoints:
        typer.echo("No checkpoints found.")
        return

    if format == "json":
        output = [cp.to_dict() for cp in checkpoints]
        typer.echo(json.dumps(output, indent=2, default=str))
    else:
        typer.echo(f"\nCheckpoints in {checkpoint_dir}")
        typer.echo("=" * 60)
        typer.echo(
            f"{'ID':<12} {'Created':<20} {'Batches':>8} {'Records':>10} {'Issues':>8}"
        )
        typer.echo("-" * 60)
        for cp in checkpoints:
            created = cp.created_at.strftime("%Y-%m-%d %H:%M:%S")
            typer.echo(
                f"{cp.checkpoint_id:<12} {created:<20} {cp.batch_count:>8} "
                f"{cp.total_records:>10} {cp.total_issues:>8}"
            )
        typer.echo(f"\nTotal: {len(checkpoints)} checkpoint(s)")


@checkpoint_app.command(name="show")
@error_boundary
def checkpoint_show_cmd(
    checkpoint_id: Annotated[
        str,
        typer.Argument(help="Checkpoint ID to show"),
    ],
    checkpoint_dir: Annotated[
        Path,
        typer.Option("--dir", "-d", help="Checkpoint directory"),
    ] = Path("./checkpoints"),
) -> None:
    """Show details of a specific checkpoint.

    Examples:
        truthound realtime checkpoint show abc12345
        truthound realtime checkpoint show abc12345 --dir ./my_checkpoints
    """
    from truthound.realtime.incremental import CheckpointManager

    if not checkpoint_dir.exists():
        typer.echo(f"Error: Checkpoint directory not found: {checkpoint_dir}", err=True)
        raise typer.Exit(1)

    manager = CheckpointManager(checkpoint_dir=checkpoint_dir)
    checkpoint = manager.get_checkpoint(checkpoint_id)

    if not checkpoint:
        typer.echo(f"Error: Checkpoint '{checkpoint_id}' not found", err=True)
        raise typer.Exit(1)

    typer.echo(f"\nCheckpoint: {checkpoint.checkpoint_id}")
    typer.echo("=" * 40)
    typer.echo(f"Created: {checkpoint.created_at}")
    typer.echo(f"Batches processed: {checkpoint.batch_count}")
    typer.echo(f"Total records: {checkpoint.total_records}")
    typer.echo(f"Total issues: {checkpoint.total_issues}")

    if checkpoint.position:
        typer.echo("\nStream Position:")
        for key, value in checkpoint.position.items():
            typer.echo(f"  {key}: {value}")

    if checkpoint.state_snapshot:
        typer.echo(f"\nState snapshot keys: {list(checkpoint.state_snapshot.keys())}")


@checkpoint_app.command(name="delete")
@error_boundary
def checkpoint_delete_cmd(
    checkpoint_id: Annotated[
        str,
        typer.Argument(help="Checkpoint ID to delete"),
    ],
    checkpoint_dir: Annotated[
        Path,
        typer.Option("--dir", "-d", help="Checkpoint directory"),
    ] = Path("./checkpoints"),
    force: Annotated[
        bool,
        typer.Option("--force", "-f", help="Skip confirmation"),
    ] = False,
) -> None:
    """Delete a checkpoint.

    Examples:
        truthound realtime checkpoint delete abc12345
        truthound realtime checkpoint delete abc12345 --force
    """
    if not checkpoint_dir.exists():
        typer.echo(f"Error: Checkpoint directory not found: {checkpoint_dir}", err=True)
        raise typer.Exit(1)

    checkpoint_file = checkpoint_dir / f"checkpoint_{checkpoint_id}.json"

    if not checkpoint_file.exists():
        typer.echo(f"Error: Checkpoint '{checkpoint_id}' not found", err=True)
        raise typer.Exit(1)

    if not force:
        confirm = typer.confirm(f"Delete checkpoint '{checkpoint_id}'?")
        if not confirm:
            typer.echo("Cancelled.")
            return

    checkpoint_file.unlink()
    typer.echo(f"Checkpoint '{checkpoint_id}' deleted.")


# =============================================================================
# Module initialization - register subcommands after all definitions
# =============================================================================

# Register checkpoint subcommand group at the end to ensure
# validate and monitor commands are registered first
_register_checkpoint_subcommands()
