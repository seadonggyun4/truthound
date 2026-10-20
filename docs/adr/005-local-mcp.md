# ADR 005: Optional local MCP adapter

Status: proposed implementation for 3.2.0. No remote service is introduced.

## Decision

Keep the validation kernel independent of MCP. An outer transport adapter uses
the official Python SDK, a transport-independent local file policy, a disposable
Core worker and explicit result projections. Distribution is an optional PyPI
extra and a generated MCPB wrapper over the same version.

The first surface has three tools with one file input each. Validation uses
explicit built-in checks and a private context; profile and PII calls reuse the
public APIs. Unknown input properties, arbitrary execution and remote sources
are not part of this version. Additional tools require their own contracts.

## Sources examined and application

This is a bounded engineering review of nine primary references, not an
exhaustive systematic review. Full paid books and paper evaluations were not
reproduced. References were accessed on 2026-10-20.

| Reference and reading scope | Design consequence | Limit |
| --- | --- | --- |
| [MCP tools specification](https://modelcontextprotocol.io/specification/2025-11-25/server/tools), structured results/errors excerpts | Versioned structured content and distinct execution/quality failures | Annotations are hints, not enforcement |
| [Official Python SDK](https://github.com/modelcontextprotocol/python-sdk), server/client API examples | Reuse transport and schema validation; exercise a real SDK client | Pin supported 1.x API; do not promise every future protocol version |
| [MCPB manifest](https://github.com/modelcontextprotocol/mcpb/blob/main/MANIFEST.md), UV and user configuration sections | Explicit directory picker; pinned Truthound version; native dependency installation through UV | Actual host/platform behavior needs separate validation |
| [Parnas, 1972](https://www.cs.lafayette.edu/~gexia/cs301/resources/parnas.html), decomposition discussion | Isolate decisions likely to change: protocol, policy, Core and packaging | Design principle, not performance evidence |
| [Saltzer and Schroeder, 1975](https://www.cs.virginia.edu/~evans/cs551/saltzer/), protection principles | Deny absent roots; mediate each request; minimize exposed authority | Application checks are not an OS security boundary |
| [Percival and Gregory, Architecture Patterns with Python, chapter 2](https://www.cosmicpython.com/book/chapter_02_repository), dependency inversion section | Dependencies point toward existing Core; optional transport remains outside | The chapter concerns persistence; applying it to MCP is an engineering inference |
| [MCP Safety Audit](https://arxiv.org/abs/2504.03767), abstract only | Do not interpret metadata as instructions or expose unrestricted tools | No reproduction or claim of solving prompt injection |
| [MCP cancellation](https://modelcontextprotocol.io/specification/2025-11-25/basic/utilities/cancellation), full utility page | Propagate request cancellation to owned work and free resources | Cancellation races are allowed; not an unconditional protocol guarantee |
| [AnyIO cancellation](https://anyio.readthedocs.io/en/stable/cancellation.html), task-group, deadline and shielded-finalization sections; installed subprocess implementation | Await a structured child directly, rather than abandoning a synchronous background thread | Normal OS process termination, not kernel-hang or sandbox protection |

## Alternatives and tradeoffs

A bespoke JSON-RPC implementation would duplicate protocol maintenance. A
remote service would add identity, hosting and dataset-transfer requirements.
Running Core directly in the event loop would block discovery and cancellation.
The disposable worker costs startup/copy time but supports bounded worker
timeouts and avoids project-state discovery. Copy size does not bound decompressed
memory use. The application does not claim adversarial filesystem race immunity.

Summary projections reduce accidental sample disclosure but retain column names
and aggregate statistics. Those may be sensitive, and the client controls model
provider transmission. Tool annotations describe intended behavior only.

## Verification

The stdio handler directly awaits the service. AnyIO's subprocess context kills
and reaps the worker on cancellation or deadline; only then does the service
remove its private input and release the busy lock. SDK connection shutdown
cancels in-flight handlers. Real wire cancellation, stdin EOF and timeout tests
observe owned PIDs and temporary directories within a declared five-second
test grace; cancellation additionally proves a subsequent normal call succeeds.
This is not a universal wall-clock bound for pathological OS/filesystem hangs.
Initialization uses the SDK factory so configured safety instructions reach clients.

The mandatory MCP lane rejects absent SDKs, empty collection, skipped tests and
missing required test files. Core source changes trigger it. Fresh wheel/sdist
and extracted UV launcher tests assert the intended Python minor inside each
installed runtime and emit actual Python/SDK/Polars versions and artifact hashes.

Use synthetic Core parity and real stdio initialization/list/call/error tests,
path traversal/symlink/size/timeout failure tests, package inventory inspection,
a fresh wheel and source install, official MCPB manifest validation and an
extracted bundle launch. GUI installation and directory approval remain separate.
