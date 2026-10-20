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
) -> None:
    """Build a version-matched Claude Desktop bundle."""
    from truthound.mcp.bundle import build_bundle

    typer.echo(str(build_bundle(output, wheel=wheel)))


def register_commands(parent_app: typer.Typer) -> None:
    parent_app.add_typer(app, name="mcp")
