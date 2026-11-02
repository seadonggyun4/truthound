---
title: Data validation MCP for local CSV and Parquet
description: Check missing values and duplicate rows, profile local datasets and screen sampled PII with the Truthound Local MCP server and explicit directory consent.
---

# Data validation MCP for local CSV and Parquet

**Truthound Local is an open-source data validation MCP server** for local CSV and
Parquet files. It checks missing values and duplicate rows, profiles column types
and aggregate statistics, and screens for likely PII before RAG ingestion or
sharing an evaluation dataset. These three tools run locally and return bounded
summaries without raw rows or matched values. This guide describes published
Truthound **3.3.1** and Local plugin **0.1.7**.

## Which data checks can an AI agent run?

| Task | MCP tool | Result and limitation |
| --- | --- | --- |
| Check a CSV for nulls and duplicate rows | `check_file` | Built-in quality checks; a failing dataset is a completed execution, not a transport error |
| Profile a CSV or Parquet before analysis | `profile_file` | Column type families, null/unique percentages and aggregate metadata; inferred types are not validation against a supplied schema |
| Screen a dataset for PII before RAG ingestion or sharing | `scan_pii` | Sampled finding estimates and confidence; a clean result cannot establish absence of PII |

Each tool takes an approved file `path`. Local does not accept custom schema/rule
arguments, edit datasets, execute arbitrary SQL, or evaluate LLM answer quality.
For explicit schema rules, drift comparison or masking, see the separate
[Core profile](mcp.md#core-installation); full Core parity through MCP is not verified.

## Install and connect locally

Use Python 3.11–3.13 and install the published MCP extra:

```bash
pip install "truthound[mcp]==3.3.1"
truthound-mcp serve --allow-dir /absolute/path/to/data
```

Configure a local stdio-capable MCP host to start this executable with the same
arguments. Installing does not register the server with your host or grant any
folder access. A client that requires a hosted HTTP/OAuth connector cannot use
this local stdio configuration directly. See the
[client configuration and supported formats](mcp.md#install-and-start).

For Claude Code, the existing public marketplace installs the Local plugin:

```bash
claude plugin marketplace add seadonggyun4/styxis-plugin
claude plugin install truthound@styxis-plugin
```

Then ask its setup skill to approve only the existing absolute data directory you
intend to inspect. See [plugin setup and permission handling](mcp.md#claude-code-styxis-marketplace).

## What stays local, and what reaches the AI?

The server reads only explicitly allowed directories and does not upload dataset
values to a hosted validation service. Local tool results exclude raw rows,
sample values and matched PII. File paths, column names and aggregate summaries
can still reach the host model provider. Treat them as untrusted data and review
your host's privacy settings. Installation alone grants no data access.

PII screening samples values and can miss sensitive data. It is not redaction,
cloud DLP, a complete file audit or a compliance certification. Local checks do not
modify the source dataset.

## Common questions

### Which MCP checks local CSV nulls and duplicates without writing SQL?

Truthound Local's `check_file` runs those built-in checks for an approved file.
The agent supplies a path rather than a generated SQL query or custom validator.
Inspect the returned quality result even when execution succeeds.

### Can it profile Parquet without returning sample rows?

Yes. Use `profile_file` for a bounded aggregate summary. It does not return sample
rows or category values; type inference alone does not prove schema conformance.

### Can it prepare RAG or evaluation data?

Use the three Local tools for tabular data hygiene and sampled PII screening.
They do not score retrieval relevance, label correctness, hallucinations or the
semantic quality of generated answers.

### Does data validation mean testing the MCP protocol?

Here it means validating dataset contents. Truthound Local is not an MCP
protocol-conformance or JSON-RPC server-testing tool.

## Official sources and execution evidence

- [Published package: Truthound 3.3.1](https://pypi.org/project/truthound/3.3.1/)
- [Official MCP Registry record](https://registry.modelcontextprotocol.io/v0.1/servers/io.github.seadonggyun4%2Ftruthound/versions/latest)
- [Public source and license](https://github.com/seadonggyun4/truthound)
- [Release and verification evidence](../releases/truthound-3.3.1.md)
- [Installed Local execution and permission checks](https://github.com/seadonggyun4/styxis-plugin/blob/main/evaluation/results/release-3.3.1/default-e2e.json)

The execution receipt tests an actual installation, not whether every search
engine or AI recommends the project. Registry publication, web indexing and
model recommendation are separate observations.
