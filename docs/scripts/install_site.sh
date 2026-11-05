#!/usr/bin/env sh
set -eu
# Vercel's system Python is externally managed. Keep build tools isolated.
tools_dir="$(mktemp -d /tmp/truthound-docs-tools.XXXXXX)"
trap 'rm -rf "$tools_dir"' EXIT
python3 -m venv "$tools_dir"
"$tools_dir/bin/pip" install uv==0.9.13
UV_PYTHON_DOWNLOADS=automatic "$tools_dir/bin/uv" python install 3.11
"$tools_dir/bin/uv" sync --frozen --extra docs --python 3.11
