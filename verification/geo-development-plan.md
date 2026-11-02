# GEO task evidence and entry paths — 2026-11-02

Planning → Development → QA. No package/runtime behavior or version is changed.

## documentation_impact

Update the bilingual documentation home, guide index and existing Local MCP
overview; add a bilingual local-mcp-example guide and public bounded-results
asset. Update full/public navigation and page counts together. Publish robots.txt
through the public support-file manifest. Keep all README/banner bytes intact.

The Korean guide landing page previously repeated placeholder prose instead of
describing real tasks. Replace it with task navigation and actual capability
boundaries; do not claim that prose quality alone establishes a ranking effect.

Deployment identity now covers the focused data-validation and execution-example
pages in both languages. Existing missing-file/invalid-identity gates remain.
Only explicit public pages are hashed; no build-state sweep is introduced.

The execution receipt comes from published Core 3.3.1, not an editable checkout.
The script creates synthetic CSV/Parquet fixtures and grants its subprocess only
that temporary directory. Existing desktop permissions remain unchanged.

QA requires docs/public-surface and deployment-identity contracts, strict full
and public builds, links, Korean prose, and deployed content/hash verification.
Runtime execution and public serving are distinct from search indexing and
AI answer recommendation. No new package publication or Registry version is
needed for these documentation changes. Private source details are omitted.

Local QA completed: 57 existing documentation/public-surface/deployment-identity
contracts passed, full/public strict builds and links passed, both Korean checks
and generated public surface passed. Live serving remains a separate gate.

Search-entry follow-up: branded diagnostic search returned existing file-source
and check-CLI pages, not the new focused guide. Add contextual links to the MCP
guide and reproducible example on those two established pages in both languages.
Extend exact deployment page hashes to include them. Branded diagnostics are not
unbranded discovery successes. Repeat required docs gates for this change.

Keep both deployment-verification guides aligned with the expanded explicit page allowlist; avoid a stale hardcoded page count.


## Besties category discovery follow-up — 2026-11-02

Planning: users searching for GX-like frameworks with MCP need an explicit link
between the data quality framework, its Local interface and the supported tasks.
Copilot's reported self-explanation is a hypothesis, not verified retrieval evidence.

Development: update canonical and Korean `gx-parity.md` and
`data-validation-mcp.md` together. Replace the Korean comparison placeholder prose
with a faithful translation. Add comparison pages to the deployment hash allowlist.
No runtime, package version, API, README or About change is involved.

Documentation impact: four existing pages; full/public nav paths and public page
counts remain unchanged. Besties receives a separate public input manifest and
checks both languages before and after deployment. Its generated draft is reviewed
and adapted to the existing documentation; it is not an automatic publishing adapter.

QA: run the existing full/public links, documentation contracts, strict builds,
Korean checks, public surface and deployment identity checks before release.

QA result: all ten documentation gates passed, including 57 documentation contract tests, strict full/public builds, both Korean checks, link checks and the public surface check. The four built target pages also passed the independent Besties article-content audit. Live deployment and search observations are recorded separately after publication.
