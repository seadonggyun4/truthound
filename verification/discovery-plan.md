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

Core package version remains 3.3.0 in this development checkout. Source metadata
is not in the already published 3.3.0 wheel. A future Core release must carry it;
do not alter the frozen Local dependency or claim installed descriptions changed.
Registry metadata for an existing published version is immutable. Prepare a new
Registry-only version or align a future package release before publishing.

QA: existing MCP, bundle, Registry, portable plugin and docs/public-surface tests;
Claude validates the actual standalone manifest; styxis-plugin scripts/e2e.py
installs and launches the package with exact directory consent only for generated
fixtures. Retain tool names for existing clients; no aliases or Core tool exposure.

Failure found: installed Claude config discarded portable cwd, and relative
server.py failed from an unrelated directory. New Claude manifest supplies its
own absolute command while portable/Codex files retain their previous contract.
Future packaging review must exercise the normalized installed command from an
unrelated working directory, never only execute a source checkout's server.py.
