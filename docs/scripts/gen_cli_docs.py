"""Generate CLI documentation from command help text.

This script extracts help information from Typer commands and
generates Markdown documentation.
"""

from pathlib import Path
import subprocess
import re
import mkdocs_gen_files


def get_command_help(command: list[str]) -> str:
    """Get help text for a CLI command.

    Args:
        command: Command parts (e.g., ["truthound", "check", "--help"])

    Returns:
        Help text output
    """
    try:
        result = subprocess.run(
            ["python", "-m", "truthound.cli"] + command[1:] + ["--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.stdout
    except Exception as e:
        return f"Error getting help: {e}"


def parse_help_to_markdown(command: str, help_text: str) -> str:
    """Parse Typer help text into Markdown format.

    Args:
        command: The command name
        help_text: Raw help text from --help

    Returns:
        Markdown formatted documentation
    """
    lines = help_text.strip().split("\n")

    md = f"### `{command}`\n\n"

    # Extract description (usually first paragraph after Usage:)
    in_description = False
    description_lines = []

    for i, line in enumerate(lines):
        if line.strip().startswith("Usage:"):
            in_description = True
            continue
        if in_description:
            if line.strip() and not line.strip().startswith(("Options:", "Arguments:", "Commands:")):
                description_lines.append(line.strip())
            else:
                break

    if description_lines:
        md += " ".join(description_lines) + "\n\n"

    # Extract usage
    usage_match = re.search(r"Usage: (.+)", help_text)
    if usage_match:
        md += f"```bash\n{usage_match.group(1)}\n```\n\n"

    # Extract options
    if "Options:" in help_text:
        options_start = help_text.index("Options:")
        options_section = help_text[options_start:]

        # Find end of options section
        next_section = re.search(r"\n[A-Z][a-z]+:", options_section[8:])
        if next_section:
            options_section = options_section[: 8 + next_section.start()]

        md += "**Options:**\n\n"
        md += "| Option | Description |\n"
        md += "|--------|-------------|\n"

        # Parse option lines
        option_pattern = r"^\s+(--\S+(?:\s+-\S)?)\s+(.+)$"
        for line in options_section.split("\n"):
            match = re.match(r"^\s+(--[\w-]+(?:,\s*-\w)?)\s+(.+)$", line)
            if match:
                option = match.group(1).strip()
                desc = match.group(2).strip()
                md += f"| `{option}` | {desc} |\n"

        md += "\n"

    return md


# CLI commands to document
CLI_COMMANDS = [
    # Core
    ["truthound", "learn"],
    ["truthound", "check"],
    ["truthound", "scan"],
    ["truthound", "mask"],
    ["truthound", "profile"],
    ["truthound", "compare"],
    # Profiling
    ["truthound", "auto-profile"],
    ["truthound", "generate-suite"],
    ["truthound", "quick-suite"],
    ["truthound", "list-formats"],
    ["truthound", "list-presets"],
    ["truthound", "list-categories"],
    # Checkpoint
    ["truthound", "checkpoint", "run"],
    ["truthound", "checkpoint", "list"],
    ["truthound", "checkpoint", "validate"],
    ["truthound", "checkpoint", "init"],
    # Docs
    ["truthound", "docs", "generate"],
    ["truthound", "docs", "themes"],
    ["truthound", "dashboard"],
    # ML
    ["truthound", "ml", "anomaly"],
    ["truthound", "ml", "drift"],
    ["truthound", "ml", "learn-rules"],
    # Lineage
    ["truthound", "lineage", "show"],
    ["truthound", "lineage", "impact"],
    # Realtime
    ["truthound", "realtime", "validate"],
    # Benchmark
    ["truthound", "benchmark", "run"],
    ["truthound", "benchmark", "list"],
    ["truthound", "benchmark", "compare"],
    # Plugin
    ["truthound", "plugin", "list"],
    ["truthound", "plugin", "install"],
    ["truthound", "plugin", "info"],
    # New
    ["truthound", "new", "validator"],
    ["truthound", "new", "reporter"],
    ["truthound", "new", "plugin"],
]


def generate_cli_reference() -> str:
    """Generate complete CLI reference documentation.

    Returns:
        Markdown content for CLI reference
    """
    content = """# CLI Reference (Auto-Generated)

This page is auto-generated from CLI help text.

!!! note
    For the complete CLI reference with examples, see the
    [CLI Reference](cli-reference.md) page.

## Commands

"""

    # Group commands by prefix
    groups = {}
    for cmd in CLI_COMMANDS:
        if len(cmd) >= 2:
            prefix = cmd[1]
            if prefix not in groups:
                groups[prefix] = []
            groups[prefix].append(cmd)
        else:
            if "root" not in groups:
                groups["root"] = []
            groups["root"].append(cmd)

    # Note: In actual build, we would call get_command_help
    # For now, just generate placeholder structure
    for group, commands in sorted(groups.items()):
        content += f"## {group.title()} Commands\n\n"

        for cmd in commands:
            cmd_str = " ".join(cmd)
            content += f"### `{cmd_str}`\n\n"
            content += f"Run `{cmd_str} --help` for usage information.\n\n"

    return content


# Generate the auto CLI reference (placeholder)
with mkdocs_gen_files.open("user-guide/cli-auto-reference.md", "w") as f:
    f.write(generate_cli_reference())
