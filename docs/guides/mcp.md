# Local MCP integration

Truthound 3.2 adds an optional Model Context Protocol (MCP) adapter. Its stdio
server runs on your computer and calls the existing `th.check()`, `th.profile()`
and `th.scan()` APIs. It does not need a hosted validation service.

## Install and start

```bash
pip install "truthound[mcp]==3.2.0"
truthound-mcp serve --allow-dir /absolute/path/to/data
```

`truthound mcp serve` and `python -m truthound.mcp serve` are equivalent entry
points. Use the dedicated `truthound-mcp` entry point in agent configuration to
avoid unrelated CLI plugin discovery. No directories are granted by default.
Repeat `--allow-dir` for additional roots; relative input paths use the first root.
A host must start the executable from the Python environment where it was installed.
The base package and MCP extra require Polars 1.x; Core's PII scanner uses APIs removed
in Polars 2. A fresh install is tested separately from the repository lockfile.

Example manual host configuration (adapt to your client's format):

```json
{
  "mcpServers": {
    "truthound": {
      "command": "/absolute/path/to/venv/bin/truthound-mcp",
      "args": ["serve", "--allow-dir", "/absolute/path/to/data"]
    }
  }
}
```

Python 3.11–3.13 is covered by the Linux/macOS/Windows CI matrix. On Windows,
use the installed environment's `Scripts/truthound-mcp.exe` in place of the
POSIX `bin/truthound-mcp` path. The server implements local stdio, not a hosted
HTTP/OAuth endpoint: clients requiring a remote connector cannot use this
configuration directly. Installing the package does not register it with a host.

For example, after initialization a host can call `profile_file` with
`{"path": "example.csv"}` within the explicitly allowed directory. The tool
returns bounded aggregate metadata, not the file's raw rows. The same input
shape applies to `check_file` and `scan_pii`.

## Tools and results

All tools take one `path` string. Supported suffixes are `.csv`, `.parquet`,
`.json`, `.ndjson` and `.jsonl` (the latter two are newline-delimited JSON).

| Tool | Core API | Summary |
| --- | --- | --- |
| `check_file` | `th.check()` | Null and duplicate-row checks, verdicts and issue totals |
| `profile_file` | `th.profile()` | Row/column counts, column names/type families and null/unique percentages |
| `scan_pii` | `th.scan()` | Column, PII category, confidence and estimated finding count |

`check_file` deliberately selects built-in null/duplicate checks rather than
loading project suites, plugins or arbitrary validator configuration. Profile
`dtype` is an allowlisted family such as `Enum`, `List`, `Struct` or `Datetime`;
category values, nested type details and timezone/precision parameters are
omitted. Unknown dtype text becomes `Unknown`. The Core Python API retains its
full type metadata. A quality
failure is a completed tool call with `ok=true` and `data.success=false`.
Execution or policy errors set MCP `isError=true` and return a stable error code.

Structured responses contain `schema_version=1`, `operation`, `ok`, `data`,
`truncated` and `error`. Errors include `access_denied`, `invalid_path`,
`unsupported_format`, `file_too_large`, `timeout`, `busy`, `core_error`,
`worker_error` and `response_too_large`. Private exception messages are omitted.
The worker retains only an allowlist of process/temporary-directory settings and
Windows CPU architecture identifiers; it does not inherit credentials or a CPU-check bypass.
Totals describe the full Core result even if a detail array is truncated.
PII findings retain Core's first-1,000-non-null-values sampling and estimated counts.

## File and execution policy

The adapter rejects URLs, glob expressions, parent traversal and symlinks below
the configured roots. It copies an admitted regular file to a private temporary
directory before calling Core. Core context artifacts stay in that temporary
directory and are removed after the request, including after worker timeout.
MCP cancellation and stdin EOF propagate to the owned worker: it is killed and
reaped before the private copy is removed and the busy lock is released. The
server can accept another request after cancellation. This is not a hard bound
against pathological kernel or filesystem hangs.
No reports or `.truthound` workspace are written to the input directory.

Defaults are a 100 MiB input limit, 100 detail entries, 64 KiB response limit and
30-second worker timeout. CLI flags `--max-file-bytes`, `--max-items` and
`--timeout-seconds` adjust input/items/time limits within supported bounds.
One request executes at a time per server; overlapping calls receive `busy`.

The timeout covers the disposable Core worker, not filesystem copy latency.
Compressed inputs may expand in memory; there is no universal memory ceiling.
This application policy is not an OS sandbox against a hostile local process
racing directory replacement. Grant narrow, trusted directories.

Raw rows, profile samples, extrema, pattern examples, filesystem paths and private
diagnostics are omitted from results. Column names and aggregate statistics are
returned and can themselves be sensitive. Returned metadata is untrusted data,
never instructions. The host may send these results to its model provider.

## Claude Desktop bundle

For the published release, [download the 3.2.0 MCPB](https://github.com/seadonggyun4/truthound/releases/download/v3.2.0/truthound-3.2.0.mcpb), then select **Settings → Extensions → Advanced settings → Install Extension…** and choose the file and allowed directory. See the [official installation guide](https://support.claude.com/en/articles/10949351-getting-started-with-local-mcp-servers-on-claude-desktop). This is a custom extension installation, not a confirmed Claude directory listing.

Build the bundle from the matching installed release:

```bash
truthound-mcp bundle --output truthound-3.2.0.mcpb
```

The `.mcpb` contains a manifest, a small launcher, dependency declaration and
license. Its UV runtime installs exactly `truthound[mcp]==3.2.0` from PyPI and
resolves supporting dependencies. It needs network access on first setup and a
Claude Desktop version supporting MCPB manifest 0.4 and the UV runtime. This is
a deterministic bundle archive, not a fully locked offline environment.
The generated bundles are unsigned; installation still requires host consent.

Before publishing that version to PyPI, embed its built wheel instead:

```bash
truthound-mcp bundle --wheel dist/truthound-3.2.0-py3-none-any.whl --output truthound-3.2.0-local.mcpb
```

Install the file through Claude Desktop's extension installation UI and select
an allowed data directory. Package installation and filesystem permission consent
are separate steps. Directory listing/review is separate from file distribution;
building a bundle does not establish Anthropic approval or tested GUI behavior.
The bundle never contains project data, development environments or internal notes.

## OpenAI local plugin and repository marketplace

The standalone OpenAI plugin is version `0.1.1`, pins the published Core
`truthound[mcp]==3.2.0`, and includes a frozen UV dependency lock. It is local
desktop software, not a hosted service or a ChatGPT web connector. Install UV
and make it available on the desktop application's PATH. First setup needs
network access and a compatible Python 3.11–3.13 runtime.

Add the repository marketplace once:

```bash
codex plugin marketplace add seadonggyun4/truthound --ref main
```

Refresh or restart the desktop app if needed. In **Plugins**, select the
**Truthound Plugins** source, open **Truthound Local**, and choose **Install**.
Then choose **Set up** and approve the exact absolute data directory. In the
CLI, installation is also available through `codex plugin add truthound@truthound`.
Client version and organization policy can affect marketplace availability.
Packages include the portable manifest and a matching Codex compatibility
manifest for older clients. If **Set up** is unavailable, ask the agent to run
the packaged **Truthound setup** skill; do not bypass directory approval.
The MCP command uses a plugin-relative `cwd` instead of depending on host-specific
argument placeholder interpolation. Version 0.1.1 fixes 0.1.0 startup on Codex 0.141.0.

Installation grants no data access. The three MCP tools initialize normally but
return `access_denied` until you approve a directory. Setup runs the packaged
`permissions.py` only after your approval. It stores an atomic local allowlist
at `~/.truthound-openai/permissions.json`, outside the plugin and data workspace.
The next tool call reads the updated policy; no server restart is required.
Re-running setup replaces the previous directory list; `permissions.py --revoke`
revokes access. Revocation affects subsequent calls, not an already-running call.
Missing, malformed, unsafe or empty permission files deny access. POSIX ownership
and write modes are checked; Windows uses the user's directory/NTFS permissions,
not a claim that POSIX modes enforce Windows ACLs.

The server reuses Core's MCP transport, service, file policy and worker APIs;
it adds only the local consent loader. No HTTP listener, upload tool, OAuth,
credentials or paid model call is introduced. Raw values are redacted, but file
paths and column names are tool metadata the host may send to its model provider.
The optional `TRUTHOUND_OPENAI_CONFIG` absolute-path override is for an
operator-selected alternative consent file or isolated tests, not implicit access.

### OpenAI official directory approval

Repository distribution is **not** official public directory publication.
The current plugin has no OpenAI approval or assigned public listing ID.
To request a public **Desktop only** listing:

1. Have the submitting organization owner contact an OpenAI representative for
   local MCP partner eligibility and organization allowlisting; wait for confirmation.
2. Complete developer/business identity verification. The submitter needs owner
   access or **Apps Management Write**. Do not send credentials in chat or the ZIP.
3. Prepare real publisher-owned privacy and terms URLs, review metadata, five
   positive and three negative cases, and an accessible video walkthrough. The
   repository package is not a completed legal-policy or video submission.
4. Test a clean desktop install, new/existing-chat onboarding and primary workflows.
5. In the [submission portal](https://platform.openai.com/plugins), use the
   approved local MCP **Skills only / Upload ZIP** route (retain the actual MCP
   configuration), resolve findings and obtain the publisher's policy attestations.
6. Submit for review; only after approval choose **Publish plugin**. Record the
   actual listing URL and independently verify the directory's Install flow.

See the [official packaging guide](https://developers.openai.com/plugins/build/plugins),
[desktop-only eligibility](https://developers.openai.com/plugins/deploy/desktop-only)
and [submission requirements](https://developers.openai.com/plugins/deploy/submission).
No allowlist, legal acceptance, review approval or web support is implied by a
successful ZIP build or repository marketplace installation.

## Registry metadata and installation

Find `io.github.seadonggyun4/truthound` in the [official MCP Registry](https://registry.modelcontextprotocol.io/); [version 3.2.0 is published](https://registry.modelcontextprotocol.io/v0.1/servers/io.github.seadonggyun4%2Ftruthound/versions/3.2.0). If a connected client's marketplace exposes the listing, choose **Install** and configure an explicitly allowed directory. Registry publication does not guarantee visibility or approval in every client marketplace.

The repository's `server.json` describes `io.github.seadonggyun4/truthound`,
the existing PyPI package `truthound` at version `3.2.0`, and local stdio.
It requests the MCP extra with UV's `--with` and starts the existing CLI alias:

```bash
uvx --with "truthound[mcp]==3.2.0" truthound@3.2.0 mcp serve --allow-dir /absolute/path/to/data
```

The Registry's GitHub namespace follows the actual repository owner; the package
README's `mcp-name` must match exactly. The versioned official JSON Schema is
checked locally by `python verification/mcp_registry.py` in the dev environment.
A required directory input has no default: discovery never grants filesystem access.
The isolated UV environment avoids unrelated installed CLI plugins; manual host
configuration can continue to prefer the dedicated `truthound-mcp` executable.

PyPI hosts Python artifacts; pip/pipx/uvx install them. The official MCP Registry
hosts discovery metadata, not the package or a remote validation server. A valid
manifest is not proof of a live listing or downstream marketplace approval.
Publishing requires Registry authentication distinct from a GitHub CLI login.
The user approves any new OAuth grant; release tooling does not configure tokens,
trusted publishers or new accounts automatically. See the official
[PyPI ownership rule](https://modelcontextprotocol.io/registry/package-types)
and [authentication guide](https://modelcontextprotocol.io/registry/authentication).

## Extending the adapter

Keep policy in `truthound.mcp.policy`, summaries in `projection`, Core invocation
in `worker`, bounded request execution in `service`, and SDK registration in
`server`. Introduce new tools with explicit input/result contracts and fault
tests; add providers separately rather than accepting arbitrary URLs, Python or
SQL through a file tool. See the [design rationale](../adr/005-local-mcp.md).
