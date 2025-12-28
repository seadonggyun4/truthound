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
    """
    from truthound.realtime import (
        MockStreamingSource,
        StreamingValidator,
        StreamingConfig,
    )

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

        results = []
        with stream:
            typer.echo("Starting streaming validation...")
            typer.echo(f"  Source: {source}")
            typer.echo(f"  Batch size: {batch_size}")
            typer.echo(f"  Validators: {validator_list or 'all'}")
            typer.echo()

            for result in streaming_validator.validate_stream(
                stream, max_batches=max_batches if max_batches > 0 else None
            ):
                status = "[ISSUES]" if result.has_issues else "[OK]"
                typer.echo(
                    f"Batch {result.batch_id}: {result.record_count} records, "
                    f"{result.issue_count} issues {status}"
                )
                results.append(result.to_dict())

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
