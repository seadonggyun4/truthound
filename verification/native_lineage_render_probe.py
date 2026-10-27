"""Actual installed SDK, Graphviz executable and isolated MCP lineage artifacts."""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import importlib.metadata
import json
import struct
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def verify():
    import truthound

    package_root = Path(truthound.__file__).resolve().parent
    sources = {
        path.relative_to(package_root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(package_root.rglob("*"))
        if path.is_file() and path.suffix in {".py", ".json"}
    }
    report = {
        "truthound": importlib.metadata.version("truthound"),
        "graphviz_sdk": importlib.metadata.version("graphviz"),
        "dot": subprocess.run(
            ["dot", "-V"], capture_output=True, text=True, check=True
        ).stderr.strip(),
        "package_source_sha256": hashlib.sha256(
            json.dumps(sources, sort_keys=True).encode()
        ).hexdigest(),
        "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "artifacts": [],
    }
    payload = '</script><img src=x onerror="synthetic_untrusted()"><script>'
    with (
        tempfile.TemporaryDirectory(prefix="truthound-lineage-private-") as private,
        tempfile.TemporaryDirectory(prefix="truthound-lineage-native-") as directory,
    ):
        import zlib

        private_png = Path(private) / "synthetic-private.png"

        def chunk(kind, data):
            return (
                struct.pack(">I", len(data))
                + kind
                + data
                + struct.pack(">I", zlib.crc32(kind + data))
            )

        private_png.write_bytes(
            b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(b"\0\xff\0\0"))
            + chunk(b"IEND", b"")
        )
        literal_image_label = f'x\\", image="{private_png}", label="x'
        parameters = StdioServerParameters(
            command=sys.executable,
            args=[
                "-I",
                "-m",
                "truthound.mcp",
                "serve",
                "--profile",
                "core",
                "--allow-dir",
                directory,
                "--allow-sensitive-results",
            ],
        )
        async with stdio_client(parameters) as (read, write), ClientSession(read, write) as client:
            await client.initialize()

            async def call(operation, **arguments):
                result = await client.call_tool(operation, arguments)
                assert not result.isError, result.structuredContent
                envelope = result.structuredContent
                assert envelope["execution_status"] == "succeeded", envelope
                return envelope["data"]

            async def content(identity):
                chunks, offset = [], 0
                while offset is not None:
                    page = await call(
                        "artifacts_read", resource_id=identity, offset=offset, limit=32768
                    )
                    chunks.append(base64.b64decode(page["base64"]))
                    offset = page["next_offset"]
                return b"".join(chunks)

            graph = await call(
                "lineage_create",
                nodes=[
                    {
                        "id": identifier,
                        "name": literal_image_label if identifier == "graph" else payload,
                        "node_type": "table",
                    }
                    for identifier in ("a-b", "a_b", "graph")
                ],
                edges=[{"source": "a-b", "target": "a_b", "edge_type": "derived_from"}],
            )
            for output_format, media_type in [("svg", "image/svg+xml"), ("png", "image/png")]:
                sizes = []
                for width, height in [(96, 64), (640, 480), (640, 32)]:
                    artifact = await call(
                        "lineage_render",
                        graph_id=graph["resource_id"],
                        renderer="graphviz",
                        output_format=output_format,
                        settings={"width": width, "height": height, "include_metadata": False},
                    )
                    assert artifact["media_type"] == media_type
                    data = await content(artifact["resource_id"])
                    proof = {
                        "format": output_format,
                        "bytes": len(data),
                        "sha256": hashlib.sha256(data).hexdigest(),
                        "bounds": [width, height],
                    }
                    if output_format == "svg":
                        root = ET.fromstring(data)
                        nodes = [
                            element for element in root.iter() if element.get("class") == "node"
                        ]
                        edges = [
                            element for element in root.iter() if element.get("class") == "edge"
                        ]
                        assert len(nodes) == 3 and len(edges) == 1
                        assert not any(element.tag.endswith("image") for element in root.iter())
                        proof.update(nodes=len(nodes), edges=len(edges))
                    else:
                        assert data.startswith(b"\x89PNG\r\n\x1a\n")
                        pixels = struct.unpack(">II", data[16:24])
                        assert 0 < pixels[0] <= width + 1 and 0 < pixels[1] <= height + 1, pixels
                        sizes.append(pixels)
                        proof["pixels"] = pixels
                    report["artifacts"].append(proof)
                    await call("resources_release", resource_id=artifact["resource_id"])
                if output_format == "png":
                    assert sizes[0] != sizes[1] and sizes[1] != sizes[2], sizes
            await call("resources_release", resource_id=graph["resource_id"])
    cases = web_cases()
    report["status"] = "passed"
    return report, cases


def web_cases():
    from truthound.lineage import EdgeType, LineageEdge, LineageGraph, LineageNode, NodeType
    from truthound.lineage.visualization import MermaidRenderer

    cases = []
    payload = '</script><img src=x onerror="synthetic_untrusted()"><script>'
    for label in [
        "same",
        'Quoted "node" [value] (x) | a; #hash',
        payload,
        "한글 노드",
        "line1\nline2",
    ]:
        graph = LineageGraph()
        for identity in ("a-b", "a_b", "end", "graph"):
            graph.add_node(LineageNode(id=identity, name=label, node_type=NodeType.TABLE))
        for source, target in [("a-b", "a_b"), ("a_b", "end"), ("end", "graph")]:
            graph.add_edge(
                LineageEdge(source=source, target=target, edge_type=EdgeType.DERIVED_FROM)
            )
        cases.append(MermaidRenderer().render(graph))
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--web-cases", type=Path)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        report, cases = asyncio.run(verify())
    except BaseException as error:
        args.output.write_text(
            json.dumps({"status": "failed", "exception_type": type(error).__name__}) + "\n"
        )
        raise
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if args.web_cases:
        args.web_cases.write_text(json.dumps(cases, ensure_ascii=False) + "\n")
    print(json.dumps({"status": report["status"], "artifacts": len(report["artifacts"])}))


if __name__ == "__main__":
    main()
