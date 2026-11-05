#!/usr/bin/env sh
set -eu
# Vercel's system Python is externally managed. Keep build tools isolated.
tools_dir="$(mktemp -d /tmp/truthound-docs-tools.XXXXXX)"
trap 'rm -rf "$tools_dir"' EXIT
python3 -m venv "$tools_dir"
"$tools_dir/bin/pip" install uv==0.9.13
python3 --version
"$tools_dir/bin/uv" sync --frozen --extra docs --python "$(command -v python3)"
