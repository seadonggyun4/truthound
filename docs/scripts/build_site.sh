#!/usr/bin/env sh
set -eu

python3 -m pip install --upgrade pip
python3 -m pip install -e ".[docs]"
python3 docs/scripts/fetch_external_docs.py --manifest docs/public_docs.yml
python3 docs/scripts/prepare_public_docs.py --manifest docs/public_docs.yml --output-dir build/public-docs
python3 docs/scripts/check_links.py --mkdocs mkdocs.yml README.md CLAUDE.md
python3 docs/scripts/check_links.py --mkdocs mkdocs.public.yml build/public-docs
python3 -m mkdocs build --strict -f mkdocs.public.yml
python3 docs/scripts/verify_public_surface.py --manifest docs/public_docs.yml --site-dir site
