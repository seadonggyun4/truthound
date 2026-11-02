# Discovery packaging change: 2026-11-02

Status: implemented, validation recorded in the distribution repository.
Planning → Development → QA; no synthetic cycle history is asserted.

## documentation_impact

Public contract changes: independent Claude manifest 0.1.6, absolute Claude
launcher, improved source-only tool descriptions and path/output descriptions,
setup discovery wording, package links, Registry description proposal.

Pages updated together: README.md, README.en.md (existing format preserved),
docs/guides/mcp.md, docs/locales/ko/guides/mcp.md; existing MCP navigation and public
manifest entry retained. Full/public/KO docs gates are required. SECURITY.md is
GitHub reporting guidance; no web security certification or SLA is implied.

The initial distribution retained published 3.3.0 while source descriptions were
unreleased. The 3.3.1 follow-up publishes the exact same-source verified Core
artifacts, then repins Local to 0.1.7/Core 3.3.1 and publishes a new immutable
Registry record. Historical 3.3.0 artifacts and Registry metadata remain unchanged.

QA: existing MCP, bundle, Registry, portable plugin and docs/public-surface tests;
Claude validates the actual standalone manifest; styxis-plugin scripts/e2e.py
installs and launches the package with exact directory consent only for generated
fixtures. Retain tool names for existing clients; no aliases or Core tool exposure.

Failure found: installed Claude config discarded portable cwd, and relative
server.py failed from an unrelated directory. New Claude manifest supplies its
own absolute command while portable/Codex files retain their previous contract.
Future packaging review must exercise the normalized installed command from an
unrelated working directory, never only execute a source checkout's server.py.


## 3.3.1 publication follow-up

User explicitly requests Core publication, Official MCP Registry update and actual
Agent selection checks. documentation_impact: README bilingual existing sections,
MCP guide and Korean overlay, new bilingual 3.3.1 release page, full/public navigation
and public page counts. Preserve all banners and GitHub About. Stage Local repin
after PyPI publication to avoid a verification dependency on an unpublished package.
Use the existing full same-revision release gate; never weaken or bypass it.

Publication evidence: docs/releases/truthound-3.3.1.md. Same-source Core gate: https://github.com/seadonggyun4/truthound/actions/runs/41371907329. Local repin is a later plugin packaging commit; do not assign it the Core artifact provenance. Actual Agent decisions remain unmeasured after readiness rejection.
