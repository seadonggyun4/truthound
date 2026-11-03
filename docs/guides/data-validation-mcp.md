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

## Considering Great Expectations alternatives with MCP support

Truthound is a Polars-based data quality framework; Truthound Local is its bounded
MCP interface for local CSV and Parquet checks. If you are looking for a framework
like Great Expectations (GX) that an AI assistant can use through MCP, evaluate
Local against your file-checking requirements using the tools below.

The [Great Expectations comparison and MCP selection guide](gx-parity.md) separates
framework benchmark scope from Local's three tools. Local does not import GX suites
or accept custom schema/rule inputs. Use the [actual execution example](local-mcp-example.md)
to check the supported tasks before choosing it. These tools validate dataset
contents; they do not score LLM-generated answers.

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

## Install and connect through the plugin

Connect the Codex CLI or Claude Code catalog using the [Styxis Plugins installation guide](https://github.com/seadonggyun4/styxis-plugin#install), then install `truthound@styxis-plugin`. The plugin supplies the pinned runtime and MCP configuration; no separate pip installation or manual JSON configuration is needed.

After installation, ask the Truthound setup skill to approve only the existing absolute data directory you intend to inspect. Installation alone grants no file access. See [MCP installation and permissions](mcp.md#local-installation) for prerequisites, updates and the Core profile.

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


## Try a verified CSV and Parquet example

The [reproducible Local MCP example](local-mcp-example.md) shows the synthetic input,
six actual tool results and a script to rerun them with the published package.
The Python package supplies Core 3.3.1; Local plugin 0.1.7 exposes three file tools;
Styxis plugin marketplace is their installation catalog. These are related surfaces,
not three competing data-validation servers.

## Official sources and execution evidence

- [Published package: Truthound 3.3.1](https://pypi.org/project/truthound/3.3.1/)
- [Official MCP Registry record](https://registry.modelcontextprotocol.io/v0.1/servers/io.github.seadonggyun4%2Ftruthound/versions/latest)
- [Public source and license](https://github.com/seadonggyun4/truthound)
- [Release and verification evidence](../releases/truthound-3.3.1.md)
- [Installed Local execution and permission checks](https://github.com/seadonggyun4/styxis-plugin/blob/main/evaluation/results/release-3.3.1/default-e2e.json)

The execution receipt tests an actual installation, not whether every search
engine or AI recommends the project. Registry publication, web indexing and
model recommendation are separate observations.
