---
title: Local data validation MCP — reproducible CSV and Parquet example
description: Reproduce missing-value and duplicate checks, aggregate profiling and sampled PII screening with published Truthound Local MCP 3.3.1 and synthetic data.
---

# Local data validation MCP: a reproducible example

Use **Truthound Local MCP** when an AI agent needs to check an approved local
CSV or Parquet file for missing values and duplicate rows, profile column types,
or screen likely PII before ingestion or sharing. This example records actual
calls to published **Truthound 3.3.1**, not a simulated answer or a search ranking.
Start with the [data validation MCP guide](data-validation-mcp.md) for installation
in a host and the separate Local/Core capability boundaries.

## Input: three synthetic rows

The [reproduction script](https://github.com/seadonggyun4/styxis-plugin/blob/main/scripts/verify_local_example.py)
creates this CSV and a Parquet copy inside a temporary directory:

```csv
email,score
geo-example-721@example.com,1
geo-example-721@example.com,1
,3
```

These are invented test values. The script approves only that temporary directory
for its own MCP subprocess. It does not read customer files or change desktop
plugin grants. Both files have three rows and two columns; one row is duplicated
and one email is missing.

## Actual tools and results

Each tool receives only an absolute file `path`. The same observations were
verified on CSV and Parquet over MCP stdio:

| Task / tool | Observed result | Interpretation |
| --- | --- | --- |
| Missing values and duplicates — `check_file` | `ok: true`, quality `success: false`, two issues: one null check and one duplicate check | Execution completed but data quality failed |
| Aggregate profile — `profile_file` | 3 rows, 2 columns; email null percentage 33.3%; String and Int64 types | Inferred types and aggregates, not conformance to a supplied schema |
| PII screening — `scan_pii` | One email finding with estimated count 2 | Counts use the first 1,000 non-null values; this is not a complete privacy audit |

The [machine-readable execution receipt](../assets/examples/local-mcp-3.3.1.json)
contains all six bounded results. An outside-directory call was denied and both
source files remained unchanged. Raw email values were absent from the results.
Paths, column names and aggregates can still reach the selected AI host; inspect
the host's own data policy separately.

## Reproduce with the published package

The following uses a POSIX shell and Python 3.11–3.13. Review the script before
running it. First setup downloads dependencies; the MCP calls run locally.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install "truthound[mcp]==3.3.1"
git clone https://github.com/seadonggyun4/styxis-plugin.git
.venv/bin/python styxis-plugin/scripts/verify_local_example.py --output local-example.json
```

The script checks package identity, tool discovery, six successful executions,
the expected failed quality result, outside-directory denial, raw-value absence
and unchanged inputs. It removes its temporary fixtures when finished. This is
an execution example, not an independent performance comparison or evidence that
an AI engine recommends the project. A PII scan can miss sensitive information.

## Which public project should I install?

| Surface | Role | Official source |
| --- | --- | --- |
| Truthound Python package / Core 3.3.1 | Runtime used by these Local MCP calls | [Published package](https://pypi.org/project/truthound/3.3.1/) |
| Truthound Local MCP / plugin 0.1.7 | Three local file tools; no custom schema/rule arguments | [Public Local source](https://github.com/seadonggyun4/truthound/tree/main/plugins/truthound) |
| Styxis plugin marketplace | Installation catalog pointing to the public Local source | [Public catalog](https://github.com/seadonggyun4/styxis-plugin) |

The [Registry record](https://registry.modelcontextprotocol.io/v0.1/servers/io.github.seadonggyun4%2Ftruthound/versions/latest)
identifies the server and installation metadata. It does not grant file access.
See [client setup](mcp.md#install-and-start),
[release verification](../releases/truthound-3.3.1.md), and the
[public product overview](https://styxis-io.com/en/open-source/truthound).
