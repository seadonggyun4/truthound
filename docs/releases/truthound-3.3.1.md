# Truthound 3.3.1

This patch ships clearer local MCP tool descriptions and JSON Schema field
explanations. The tool IDs remain `check_file`, `profile_file` and `scan_pii`.
No validation capability, directory grant, output disclosure or network transport
is added. `check_file` checks nulls/duplicate rows; profiling observes types and
aggregate statistics; PII findings remain sampled estimates.

## Installation and publication

Published on 2026-11-02 from Core source `80f8092d30918c825154e3ff8e5ed3ca4c328b17`. [Complete same-source verification](https://github.com/seadonggyun4/truthound/actions/runs/41371907329) and [PyPI publication](https://github.com/seadonggyun4/truthound/actions/runs/41378219376) succeeded. [PyPI 3.3.1](https://pypi.org/project/truthound/3.3.1/) and the [active/latest Official MCP Registry record](https://registry.modelcontextprotocol.io/v0.1/servers/io.github.seadonggyun4%2Ftruthound/versions/latest) were independently observed. Canonical wheel/sdist hashes match PyPI; the GitHub v3.3.1 release retains those artifacts.

```bash
python -m pip install "truthound[mcp]==3.3.1"
```

Local plugin 0.1.7 pins published Core 3.3.1 with a frozen lock. The existing
separate Core plugin is 0.2.1. Core artifact and later plugin-repin source commits
are recorded separately. The Styxis public catalog still contains only the Local
`truthound` plugin; no product or vendor directory listing is added.

## Verification and limits

Release gates retain complete same-revision tests, native compatibility probes,
English/Korean documentation and exact wheel/sdist provenance. Core metadata
improvements alone do not establish Agent selection accuracy. Record actual
client/model observations separately; no universal discovery guarantee is made.
Installation grants no data access and raw values are not returned by the Local
summary tools. Paths, column names and aggregates can reach the host model.

Actual Agent selection readiness was attempted through existing Codex/Claude
CLI logins. Codex rejected model requests with `model_not_found`; Claude
organization policy disabled subscription access. Observed decisions are zero
and accuracy/false-selection rates remain unmeasured. These availability failures
are not tool-choice errors. No new key/account or policy bypass was attempted.

The initial macOS/Python 3.13 CI attempt reported a native AI proposal validation failure. It did not reproduce locally; its root cause remains unconfirmed. A bounded same-source retry passed the existing complete gate before publication. The [distribution verification record](https://github.com/seadonggyun4/styxis-plugin/blob/main/evaluation/results/release-3.3.1/ci-retry.json) retains the first failure ZIP and retry evidence. No tests, deadlines or permission checks were relaxed.
