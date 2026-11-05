#!/usr/bin/env sh
set -eu
# Dependencies are installed from uv.lock by Vercel before this command.
# Use the committed orchestration snapshot; do not fetch a moving upstream branch.
.venv/bin/python docs/scripts/prepare_public_docs.py --mode full --manifest docs/public_docs.yml
.venv/bin/python docs/scripts/prepare_public_docs.py --mode public --manifest docs/public_docs.yml
.venv/bin/python docs/scripts/check_links.py --mkdocs mkdocs.public.yml build/public-docs
.venv/bin/mkdocs build --strict -f mkdocs.public.yml
.venv/bin/python docs/scripts/verify_public_surface.py --manifest docs/public_docs.yml --site-dir site
.venv/bin/python docs/scripts/verify_korean_docs.py --site-dir site
.venv/bin/python docs/scripts/write_deployment_manifest.py --site-dir site
