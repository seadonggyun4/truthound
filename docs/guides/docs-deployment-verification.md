# Docs Deployment Verification

Use this page as the canonical operator checklist whenever the public Truthound docs portal is built, verified, or repaired.

## Netlify Configuration Checklist

- [ ] Netlify source repository is `seadonggyun4/Truthound`
- [ ] the production branch is `main`
- [ ] the build command is `sh docs/scripts/build_site.sh`
- [ ] the publish directory is `site`
- [ ] the deployed site uses `https://truthound.netlify.app/` as the canonical docs URL
- [ ] the docs operator runbook distinguishes the public docs host from any separately operated dashboard runtime

## Repo Configuration Checklist

- [ ] `mkdocs.yml` sets `site_url: https://truthound.netlify.app/`
- [ ] `mkdocs.yml` builds from the staged full docs tree
- [ ] `mkdocs.public.yml` builds from the staged public docs tree
- [ ] `netlify.toml` points at the repo-owned docs build script
- [ ] `docs/public_docs.yml` defines the public docs manifest, including section prefixes and exclusions
- [ ] `docs/_redirects` keeps `truthound.io` and `www.truthound.io` in redirect-only mode toward `truthound.netlify.app`
- [ ] `docs/orchestration/` is a checked-in snapshot mirrored from `truthound-orchestration`
- [ ] `docs/dashboard/index.md` is a locally maintained overview page rather than a mirrored dashboard manual
- [ ] `docs/ai/` is a locally maintained technical section describing the public `truthound.ai` contract
- [ ] docs verification does not assume a dashboard runtime is hosted inside the Netlify site
- [ ] `scripts/fetch-external-docs.sh orchestration` is the canonical pre-release sync step for the orchestration snapshot
- [ ] full and public docs staging run before the respective MkDocs builds
- [ ] brand assets remain unchanged:
  - [ ] `assets/truthound_banner.png`
  - [ ] `assets/truthound_icon.png`
  - [ ] the existing palette configuration in both MkDocs configs

## Generated Site Checklist

After a local or Netlify build, confirm all of the following under `site/`:

- [ ] `site/index.html` contains the 3.0 landing copy and expanded portal navigation
- [ ] `site/tutorials/index.html` exists
- [ ] `site/cli/index.html` exists
- [ ] `site/python-api/index.html` exists
- [ ] `site/reference/index.html` exists
- [ ] `site/ai/index.html` exists and renders the AI overview
- [ ] `site/ai/system-boundary/index.html` exists and renders the AI boundary document
- [ ] `site/orchestration/index.html` exists and renders the Orchestration overview
- [ ] `site/dashboard/index.html` exists and renders the Dashboard overview
- [ ] `site/orchestration/airflow/index.html` exists and renders one platform page under the integrated section
- [ ] `site/orchestration/airflow/observability-alerting/index.html` exists and renders an expanded platform subguide
- [ ] `site/orchestration/common/index.html` exists and renders the shared runtime overview
- [ ] `site/orchestration/common/engine-resolution-selection/index.html` exists and renders a shared runtime deep dive
- [ ] `site/orchestration/engines/index.html` exists and renders the engine overview
- [ ] `site/orchestration/engines/capability-matrix/index.html` exists and renders an advanced engine guide
- [ ] `site/orchestration/enterprise/index.html` exists and renders the enterprise operations overview
- [ ] `site/orchestration/enterprise/rollout-topologies/index.html` exists and renders an enterprise operations subguide
- [ ] `site/releases/latest-benchmark-summary/index.html` exists and shows the verified benchmark numbers
- [ ] `site/releases/truthound-3.1/index.html` exists and shows the AI surface and docs restructuring release note
- [ ] `site/sitemap.xml` exists
- [ ] `site/search/search_index.json` exists
- [ ] the home page banner resolves from `assets/truthound_banner.png`
- [ ] the benchmark summary icon banner resolves from `assets/Truthound_icon_banner.png`
- [ ] the staged public docs tree reflects the public manifest expansion from `docs/public_docs.yml`
- [ ] imported orchestration pages render the generated source banner and upstream edit link

## Search and Sitemap Checklist

- [ ] `sitemap.xml` contains exactly the number of pages declared by `expected_page_count` in `docs/public_docs.yml`
- [ ] `sitemap.xml` includes:
  - [ ] home
  - [ ] tutorials
  - [ ] CLI reference
  - [ ] AI overview
  - [ ] AI system boundary
  - [ ] Python API reference
  - [ ] performance
  - [ ] benchmark methodology
  - [ ] Great Expectations comparison
  - [ ] latest verified benchmark summary
  - [ ] orchestration overview
  - [ ] orchestration airflow
  - [ ] orchestration shared runtime
  - [ ] orchestration shared runtime deep dive
  - [ ] orchestration engines
  - [ ] orchestration engines capability matrix
  - [ ] orchestration enterprise
  - [ ] orchestration enterprise rollout topologies
- [ ] `search_index.json` contains exactly the number of pages declared by `expected_page_count` in `docs/public_docs.yml`
- [ ] `search_index.json` contains:
  - [ ] `TruthoundContext`
  - [ ] `ValidationRunResult`
  - [ ] `truthound.ai`
  - [ ] `Great Expectations`
  - [ ] `benchmark`
  - [ ] `release-grade`
  - [ ] `Truthound Orchestration`
- [ ] `search_index.json` contains imported orchestration families such as:
  - [ ] `shared runtime`
  - [ ] `engine`
  - [ ] `enterprise`
  - [ ] `airflow`
  - [ ] `prefect`
  - [ ] `dbt`
- [ ] neither `sitemap.xml` nor `search_index.json` expose explicitly excluded duplicate legacy markdown files

## Content Verification Checklist

- [ ] the home page reads like a finished 3.0 release, not a transition snapshot
- [ ] primary docs do not treat `RC1` as the current release note
- [ ] benchmark pages use `release-grade`, `fixed-runner benchmark verification`, or `verified benchmark summary` rather than repeated `GA` wording
- [ ] README and the docs site agree on the current benchmark positioning against Great Expectations
- [ ] Home, Core, AI, Orchestration, Dashboard, Release Notes, ADRs, and Legacy / Archive are all discoverable from the public nav
- [ ] the Core nav exposes Getting Started, Tutorials, Guides, Reference, and Concepts & Architecture
- [ ] dashboard docs remain intentionally high-level and do not claim a public pip install workflow
- [ ] AI docs describe the additive `truthound.ai` namespace without implying core reverse-imports
- [ ] imported orchestration pages identify `truthound-orchestration` as the upstream source repository
- [ ] source-banner edit links target `https://github.com/seadonggyun4/truthound-orchestration/edit/main/docs/...`

## Local Verification Commands

```bash
sh scripts/fetch-external-docs.sh orchestration
python docs/scripts/prepare_public_docs.py --mode full
python docs/scripts/prepare_public_docs.py --mode public
python docs/scripts/check_links.py --mkdocs mkdocs.yml README.md CLAUDE.md build/full-docs
python docs/scripts/check_links.py --mkdocs mkdocs.public.yml build/public-docs
python -m mkdocs build --strict
python -m mkdocs build --strict -f mkdocs.public.yml
python docs/scripts/verify_public_surface.py --manifest docs/public_docs.yml --site-dir site
```

## Related Reading

- [Performance and Benchmarks](performance.md)
- [Benchmark Methodology](benchmark-methodology.md)
- [Latest Verified Benchmark Summary](../releases/latest-benchmark-summary.md)
- [Truthound AI Overview](../ai/index.md)
- [Truthound Orchestration Overview](../orchestration/index.md)
