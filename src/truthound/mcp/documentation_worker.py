"""Owned documentation generation; only pinned sources and retained file IDs on wire."""

import asyncio
import hashlib
import json
from contextlib import suppress
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

from truthound.mcp.broker_worker import NativeServiceBytes, NativeServiceResult
from truthound.mcp.documentation_services import DocumentationJob, DocumentationRuntime
from truthound.mcp.policy import MCPError


class DocumentationOwner:
    def __init__(self, runtime, request):
        from truthound.cli_modules.scaffolding.base import BaseScaffold, ScaffoldConfig
        from truthound.docs import DocumentationGenerator

        if not isinstance(runtime, DocumentationRuntime) or any(
            not isinstance(job, DocumentationJob)
            or not isinstance(
                job.generator, BaseScaffold if job.kind == "scaffold" else DocumentationGenerator
            )
            or job.kind not in {"source", "module", "info", "scaffold", "validate"}
            or type(job.strict) is not bool
            or (job.strict and job.kind != "validate")
            or (job.kind == "scaffold" and not isinstance(job.source, ScaffoldConfig))
            for job in runtime.jobs.values()
        ):
            raise MCPError("invalid_binding")
        self.jobs = {
            key: replace(job, source=deepcopy(job.source)) for key, job in runtime.jobs.items()
        }
        self.results = {}
        self._pending = None
        self._closed = False
        self.shutdown_requested = False

    @staticmethod
    def selected(mapping, key):
        try:
            return mapping[key]
        except KeyError:
            raise MCPError("invalid_handle") from None

    def generate(self, job, budget):
        from truthound.docs.generator import GenerationResult

        with TemporaryDirectory(prefix="truthound-api-docs-") as directory:
            root = Path(directory).resolve()
            output = root / "docs"
            try:
                if job.kind == "scaffold":
                    generated = job.generator.generate(
                        replace(deepcopy(job.source), output_dir=output)
                    )
                    # Preflight every file before any native write. Installed generators
                    # remain trusted code, but their declared artifact paths are bounded.
                    seen = set()
                    for entry in generated.files:
                        relative = Path(entry.path)
                        if (
                            relative.is_absolute()
                            or ".." in relative.parts
                            or relative in seen
                            or not relative.parts
                        ):
                            raise ValueError("Invalid scaffold artifact path")
                        seen.add(relative)
                    generated.write_files(output)
                    result = GenerationResult(
                        success=generated.success,
                        errors=generated.errors,
                        warnings=generated.warnings,
                    )
                elif job.kind == "validate":
                    from truthound.docs.generator import validate_docstrings

                    config = job.generator.config
                    issues = validate_docstrings(
                        job.source,
                        strict=job.strict,
                        include_private=config.include_private,
                        include_magic=config.include_magic,
                        docstring_parser=job.generator.extractor.docstring_parser,
                    )
                    output.mkdir()
                    (output / "docstrings.json").write_text(
                        json.dumps(issues, allow_nan=False), encoding="utf-8"
                    )
                    result = GenerationResult(
                        success=not issues["errors"],
                        errors=issues["errors"],
                        warnings=issues["warnings"],
                    )
                elif job.kind == "module":
                    output.mkdir()
                    job.generator.generate_module(job.source, output / "module.txt")
                    result = GenerationResult(modules_processed=1)
                elif job.kind == "info":
                    result = job.generator.generate_from_info(job.source, output)
                else:
                    result = job.generator.generate(job.source, output)
            except Exception:
                result = GenerationResult(success=False, errors=["native_documentation_failed"])
            if not isinstance(result, GenerationResult):
                raise MCPError("unsupported_value")
            # MkDocs writes its configuration beside docs/, within this owned root.
            files, manifest, size = {}, [], 0
            for path in sorted(root.rglob("*")):
                if path.is_symlink() or not path.resolve().is_relative_to(root):
                    raise MCPError("unsupported_value")
                if path.is_dir():
                    continue
                if not path.is_file() or len(files) >= 256:
                    raise MCPError("resource_exhausted")
                with path.open("rb") as stream:
                    content = stream.read(budget - size + 1)
                size += len(content)
                if size > budget:
                    raise MCPError("resource_exhausted")
                identity = "file_" + uuid4().hex
                files[identity] = content
                manifest.append(
                    {
                        "file_id": identity,
                        "name": path.relative_to(root).as_posix(),
                        "bytes": len(content),
                        "sha256": hashlib.sha256(content).hexdigest(),
                        "executable": bool(path.stat().st_mode & 0o111),
                    }
                )
            payload = {
                "success": result.success and bool(files),
                "modules_processed": result.modules_processed,
                "classes_found": result.classes_found,
                "functions_found": result.functions_found,
                "errors": ["native_documentation_failed"] if result.errors else [],
                "warning_count": len(result.warnings),
                "files": manifest,
            }
            size += len(json.dumps(payload, allow_nan=False).encode())
            if size > budget:
                raise MCPError("resource_exhausted")
            return payload, files, size

    def execute_native(self, args, budget):
        command = args["command"]
        if command == "inspect":
            return {"jobs": sorted(self.jobs), "results": list(self.results)}
        if command == "shutdown":
            self.shutdown_requested = True
            return {"closed": True}
        if command == "generate":
            if len(self.results) >= 16:
                raise MCPError("resource_exhausted")
            remaining = budget - sum(item[2] for item in self.results.values())
            if remaining <= 0:
                raise MCPError("resource_exhausted")
            value = self.generate(self.selected(self.jobs, args["job_id"]), remaining)
            identity = "documentation_" + uuid4().hex
            self.results[identity] = value
            payload = {"result_id": identity, "result": value[0]}
            return (
                payload
                if value[0]["success"]
                else NativeServiceResult(payload, "native_documentation_incomplete")
            )
        result = self.selected(self.results, args["result_id"])
        if command == "result":
            return {"result_id": args["result_id"], "result": result[0]}
        if command == "read":
            return NativeServiceBytes(self.selected(result[1], args["file_id"]))
        if command == "release":
            del self.results[args["result_id"]]
            return {"released": True}
        raise MCPError("invalid_arguments")

    async def execute(self, args, budget):
        if self._closed or self._pending is not None:
            raise MCPError("invalid_handle")
        task = asyncio.create_task(asyncio.to_thread(self.execute_native, args, budget))
        self._pending = task
        task.add_done_callback(lambda done: None if done.cancelled() else done.exception())
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            with suppress(Exception):
                await asyncio.shield(task)
            raise
        finally:
            if task.done():
                self._pending = None

    async def close(self):
        self._closed = True
        if self._pending is not None:
            with suppress(Exception):
                await asyncio.shield(self._pending)
            self._pending = None
        self.results.clear()


if __name__ == "__main__":
    from truthound.mcp.broker_worker import main

    main(DocumentationOwner)
