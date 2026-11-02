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
