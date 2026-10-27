"""Lazy MCP subcommands; importing the ordinary CLI never requires the SDK."""

from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer(help="Local Model Context Protocol integration")


@app.command()
def serve(
    allow_dir: Annotated[list[Path], typer.Option("--allow-dir")],
    max_file_bytes: Annotated[int, typer.Option(min=1, max=1024**3)] = 100 * 1024 * 1024,
    max_items: Annotated[int, typer.Option(min=1, max=1000)] = 100,
    timeout_seconds: Annotated[int, typer.Option(min=1, max=300)] = 30,
    profile: Annotated[str, typer.Option("--profile")] = "summary",
    allow_sensitive_results: Annotated[bool, typer.Option("--allow-sensitive-results")] = False,
    workspace_dir: Annotated[list[Path] | None, typer.Option("--workspace-dir")] = None,
    control_dir: Annotated[Path | None, typer.Option("--control-dir")] = None,
    max_handles: Annotated[int, typer.Option(min=1, max=1024)] = 128,
    max_session_bytes: Annotated[int, typer.Option(min=1024, max=8 * 1024**3)] = 1024**3,
    handle_ttl_seconds: Annotated[int, typer.Option(min=1, max=3600)] = 900,
) -> None:
    """Serve Core tools over stdio; explicit file access is required."""
    import asyncio

    from truthound.mcp.policy import MCPPolicy
    from truthound.mcp.server import serve as run

    try:
        policy = MCPPolicy(
            tuple(allow_dir),
            max_file_bytes=max_file_bytes,
            max_items=max_items,
            timeout_seconds=timeout_seconds,
            profile=profile,
            allow_sensitive_results=allow_sensitive_results,
            workspace_directories=tuple(workspace_dir or ()),
            control_directory=control_dir,
            max_handles=max_handles,
            max_session_bytes=max_session_bytes,
            handle_ttl_seconds=handle_ttl_seconds,
        )
        asyncio.run(run(policy))
    except ModuleNotFoundError as exc:
        if exc.name and (exc.name == "mcp" or exc.name.startswith("mcp.")):
            typer.echo('Install: pip install "truthound[mcp]"', err=True)
            raise typer.Exit(2) from None
        raise
    except (ValueError, OSError):
        typer.echo("Invalid MCP policy", err=True)
        raise typer.Exit(2) from None


@app.command()
def bundle(
    output: Annotated[Path, typer.Option("--output")],
    wheel: Annotated[Path | None, typer.Option("--wheel")] = None,
    profile: Annotated[str, typer.Option("--profile")] = "summary",
) -> None:
    """Build a version-matched Claude Desktop bundle."""
    from truthound.mcp.bundle import build_bundle

    typer.echo(str(build_bundle(output, wheel=wheel, profile=profile)))


def register_commands(parent_app: typer.Typer) -> None:
    parent_app.add_typer(app, name="mcp")
