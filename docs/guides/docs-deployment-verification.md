# Docs Deployment Verification

Use this page as the canonical operator checklist whenever the public docs site needs to be verified or repaired.

## Netlify Configuration Checklist

- [ ] Netlify source repository is `seadonggyun4/Truthound`
- [ ] the production branch is `main`
- [ ] the build command is `sh docs/scripts/build_site.sh`
- [ ] the publish directory is `site`
- [ ] the deployed site uses `https://truthound.netlify.app/` as the canonical docs URL

## Repo Configuration Checklist

- [ ] `mkdocs.yml` sets `site_url: https://truthound.netlify.app/`
- [ ] `mkdocs.public.yml` is the only config used for the staged public build
- [ ] `netlify.toml` points at the repo-owned docs build script
- [ ] `docs/public_docs.yml` defines the strict public allowlist
- [ ] `docs/_redirects` keeps `truthound.io` and `www.truthound.io` in redirect-only mode toward `truthound.netlify.app`
- [ ] `truthound-orchestration` and `truthound-dashboard` are linked as related projects only and are not injected into the main docs build

## Generated Site Checklist

After a local or Netlify build, confirm all of the following under `site/`:

- [ ] `site/index.html` contains the 3.0 landing copy and current navigation
- [ ] `site/releases/latest-benchmark-summary/index.html` exists and shows the verified benchmark numbers
- [ ] `site/sitemap.xml` exists
- [ ] `site/search/search_index.json` exists
- [ ] the home page banner resolves from `assets/truthound_banner.png`
- [ ] the benchmark summary icon banner resolves from `assets/Truthound_icon_banner.png`
- [ ] the staged public docs tree only contains the curated allowlist from `docs/public_docs.yml`

## Search and Sitemap Checklist

- [ ] `sitemap.xml` contains exactly `22` public pages
- [ ] `sitemap.xml` includes:
  - [ ] home
  - [ ] performance
  - [ ] benchmark methodology
  - [ ] Great Expectations comparison
  - [ ] latest verified benchmark summary
- [ ] `search_index.json` contains exactly `22` public pages
- [ ] `search_index.json` contains:
  - [ ] `TruthoundContext`
  - [ ] `ValidationRunResult`
  - [ ] `Great Expectations`
  - [ ] `benchmark`
  - [ ] `release-grade`
- [ ] neither `sitemap.xml` nor `search_index.json` expose:
  - [ ] `cli/`
  - [ ] `tutorials/`
  - [ ] `python-api/`
  - [ ] `dashboard/`
  - [ ] excluded deep `guides/` or `concepts/` families

## Content Verification Checklist

- [ ] the home page reads like a finished 3.0 release, not a transition snapshot
- [ ] primary docs do not treat `RC1` as the current release note
- [ ] benchmark pages use `release-grade`, `fixed-runner benchmark verification`, or `verified benchmark summary` rather than repeated `GA` wording
- [ ] README and the docs site agree on the current benchmark positioning against Great Expectations

## Local Verification Commands

```bash
python docs/scripts/prepare_public_docs.py --manifest docs/public_docs.yml --output-dir build/public-docs
python docs/scripts/check_links.py --mkdocs mkdocs.yml README.md CLAUDE.md
python docs/scripts/check_links.py --mkdocs mkdocs.public.yml build/public-docs
python -m mkdocs build --strict -f mkdocs.public.yml
python docs/scripts/verify_public_surface.py --manifest docs/public_docs.yml --site-dir site
```

## Related Reading

- [Performance and Benchmarks](performance.md)
- [Benchmark Methodology](benchmark-methodology.md)
- [Latest Verified Benchmark Summary](../releases/latest-benchmark-summary.md)
