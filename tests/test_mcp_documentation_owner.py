"""Native documentation files, failure evidence and confinement at the MCP owner."""

import hashlib

import pytest

from truthound.docs import DocConfig, DocumentationGenerator
from truthound.mcp.broker_worker import NativeServiceResult
from truthound.mcp.documentation_services import DocumentationJob, DocumentationRuntime
from truthound.mcp.documentation_worker import DocumentationOwner
from truthound.mcp.policy import MCPError


@pytest.mark.asyncio
@pytest.mark.parametrize("format", ["mkdocs", "sphinx", "html", "json"])
async def test_native_generated_files_retained_verified_and_released(tmp_path, format):
    source = tmp_path / "example.py"
    source.write_text(
        '"""Synthetic API."""\ndef add(a: int, b: int) -> int:\n    """Add inputs."""\n    return a + b\n'
    )
    forbidden = tmp_path / "must-not-write"
    generator = DocumentationGenerator(DocConfig(format=format, output_dir=str(forbidden)))
    owner = DocumentationOwner(
        DocumentationRuntime({"local": DocumentationJob(generator, source)}), {}
    )
    try:
        result = await owner.execute({"command": "generate", "job_id": "local"}, 1024 * 1024)
        assert result["result"]["success"] and result["result"]["functions_found"] == 1
        assert not forbidden.exists()
        assert result["result"]["files"]
        for file in result["result"]["files"]:
            content = await owner.execute(
                {"command": "read", "result_id": result["result_id"], "file_id": file["file_id"]},
                1024 * 1024,
            )
            assert hashlib.sha256(content.content).hexdigest() == file["sha256"]
            assert len(content.content) == file["bytes"]
            assert not file["name"].startswith("/")
        await owner.execute({"command": "release", "result_id": result["result_id"]}, 1024 * 1024)
        with pytest.raises(MCPError):
            await owner.execute(
                {"command": "result", "result_id": result["result_id"]}, 1024 * 1024
            )
    finally:
        await owner.close()


@pytest.mark.asyncio
async def test_partial_native_render_failure_retains_files_without_private_error(tmp_path):
    from truthound.docs.renderer import MkDocsRenderer

    class PartialRenderer(MkDocsRenderer):
        def write(self, info, output_dir=None):
            from pathlib import Path

            target = Path(output_dir)
            target.mkdir()
            (target / "partial.md").write_text("partial evidence")
            raise RuntimeError("private-renderer-secret")

    source = tmp_path / "example.py"
    source.write_text('"""Synthetic API."""')
    generator = DocumentationGenerator(renderer=PartialRenderer())
    owner = DocumentationOwner(
        DocumentationRuntime({"local": DocumentationJob(generator, source)}), {}
    )
    try:
        result = await owner.execute({"command": "generate", "job_id": "local"}, 1024 * 1024)
        assert isinstance(result, NativeServiceResult)
        assert result.failure_code == "native_documentation_incomplete"
        assert result.payload["result"]["errors"] == ["native_documentation_failed"]
        assert len(result.payload["result"]["files"]) == 1
        assert "private-renderer-secret" not in str(result.payload)
    finally:
        await owner.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", ["symlink", "budget"])
async def test_generated_files_cannot_follow_links_or_exceed_retained_budget(tmp_path, fault):
    from truthound.docs.renderer import MkDocsRenderer

    private = tmp_path / "private.txt"
    private.write_text("private-file-content")

    class InvalidRenderer(MkDocsRenderer):
        def write(self, info, output_dir=None):
            from pathlib import Path

            target = Path(output_dir)
            target.mkdir()
            if fault == "symlink":
                (target / "linked.txt").symlink_to(private)
            else:
                (target / "large.txt").write_bytes(b"x" * 4096)
            return []

    source = tmp_path / "example.py"
    source.write_text('"""Synthetic API."""')
    generator = DocumentationGenerator(renderer=InvalidRenderer())
    owner = DocumentationOwner(
        DocumentationRuntime({"local": DocumentationJob(generator, source)}), {}
    )
    try:
        with pytest.raises(MCPError):
            await owner.execute({"command": "generate", "job_id": "local"}, 1024)
        assert not owner.results
        assert private.read_text() == "private-file-content"
    finally:
        await owner.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["module", "info"])
async def test_pinned_module_and_extracted_info_use_native_generation(tmp_path, monkeypatch, kind):
    from truthound.docs.extractor import APIExtractor

    source = tmp_path / "synthetic_documented_api.py"
    source.write_text(
        '"""Synthetic API."""\ndef operation(value: int) -> int:\n    """Return value."""\n    return value\n'
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    target = (
        "synthetic_documented_api"
        if kind == "module"
        else APIExtractor().extract_module(file_path=source)
    )
    owner = DocumentationOwner(
        DocumentationRuntime({"local": DocumentationJob(DocumentationGenerator(), target, kind)}),
        {},
    )
    try:
        result = await owner.execute({"command": "generate", "job_id": "local"}, 1024 * 1024)
        assert result["result"]["success"] and result["result"]["modules_processed"] == 1
        assert result["result"]["files"]
    finally:
        await owner.close()


@pytest.mark.asyncio
async def test_cancelled_documentation_joins_renderer_and_removes_owned_directory(tmp_path):
    import asyncio
    import threading
    from pathlib import Path

    from truthound.docs.renderer import MkDocsRenderer

    entered, release, finished = threading.Event(), threading.Event(), threading.Event()
    directories = []

    class WaitingRenderer(MkDocsRenderer):
        def write(self, info, output_dir=None):
            directories.append(Path(output_dir).parent)
            entered.set()
            try:
                if not release.wait(5):
                    raise TimeoutError("synthetic release missing")
                return super().write(info, output_dir)
            finally:
                finished.set()

    source = tmp_path / "example.py"
    source.write_text('"""Synthetic API."""')
    owner = DocumentationOwner(
        DocumentationRuntime(
            {"local": DocumentationJob(DocumentationGenerator(renderer=WaitingRenderer()), source)}
        ),
        {},
    )
    task = asyncio.create_task(
        owner.execute({"command": "generate", "job_id": "local"}, 1024 * 1024)
    )
    try:
        async with asyncio.timeout(5):
            while not entered.is_set():
                await asyncio.sleep(0.01)
        asyncio.get_running_loop().call_later(0.05, release.set)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert finished.is_set()
        assert directories and all(not directory.exists() for directory in directories)
        await owner.close()
        assert not owner.results
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        await owner.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("family", ["validator", "reporter", "plugin"])
async def test_pinned_native_scaffolds_return_hashable_source_files(family):
    from truthound.cli_modules.scaffolding import (
        PluginScaffold,
        ReporterScaffold,
        ScaffoldConfig,
        ScaffoldType,
        ValidatorScaffold,
    )

    factory = {
        "validator": ValidatorScaffold,
        "reporter": ReporterScaffold,
        "plugin": PluginScaffold,
    }[family]
    config = ScaffoldConfig(
        name="synthetic_component",
        scaffold_type=ScaffoldType[family.upper()],
        template_variant="full",
    )
    owner = DocumentationOwner(
        DocumentationRuntime({"local": DocumentationJob(factory(), config, "scaffold")}), {}
    )
    try:
        generated = await owner.execute({"command": "generate", "job_id": "local"}, 1024 * 1024)
        assert generated["result"]["success"]
        python_files = [
            entry for entry in generated["result"]["files"] if entry["name"].endswith(".py")
        ]
        assert python_files
        for entry in python_files:
            artifact = await owner.execute(
                {
                    "command": "read",
                    "result_id": generated["result_id"],
                    "file_id": entry["file_id"],
                },
                1024 * 1024,
            )
            compile(artifact.content, entry["name"], "exec")
    finally:
        await owner.close()


@pytest.mark.asyncio
async def test_strict_native_docstring_violations_are_failed_artifacts(tmp_path):
    import json

    source = tmp_path / "example.py"
    source.write_text("def undocumented(value):\n    return value\n")
    owner = DocumentationOwner(
        DocumentationRuntime(
            {"local": DocumentationJob(DocumentationGenerator(), source, "validate", strict=True)}
        ),
        {},
    )
    try:
        failed = await owner.execute({"command": "generate", "job_id": "local"}, 1024 * 1024)
        assert isinstance(failed, NativeServiceResult)
        manifest = failed.payload["result"]
        assert not manifest["success"] and manifest["errors"]
        artifact = await owner.execute(
            {
                "command": "read",
                "result_id": failed.payload["result_id"],
                "file_id": manifest["files"][0]["file_id"],
            },
            1024 * 1024,
        )
        details = json.loads(artifact.content)
        assert details["errors"] and not details["warnings"]
    finally:
        await owner.close()
