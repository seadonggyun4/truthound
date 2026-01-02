#!/bin/bash
# Fetch orchestration docs from truthound-orchestration repository
# Usage: ./scripts/fetch-orchestration-docs.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
TEMP_DIR="$PROJECT_ROOT/_temp_orchestration"
TARGET_DIR="$PROJECT_ROOT/docs/orchestration"

echo "=== Fetching orchestration docs ==="
echo "Project root: $PROJECT_ROOT"

# Clean up any existing temp directory
rm -rf "$TEMP_DIR"

# Clone with sparse checkout (only docs folder)
echo "Cloning truthound-orchestration (docs only)..."
git clone --depth 1 --filter=blob:none --sparse \
  https://github.com/seadonggyun4/truthound-orchestration.git \
  "$TEMP_DIR"

cd "$TEMP_DIR"
git sparse-checkout set docs

# Copy docs to target directory
echo "Copying docs to $TARGET_DIR..."
mkdir -p "$TARGET_DIR"

# Remove old files except .gitkeep
find "$TARGET_DIR" -type f ! -name ".gitkeep" -delete 2>/dev/null || true
find "$TARGET_DIR" -type d -empty -delete 2>/dev/null || true

# Copy new files
cp -r docs/* "$TARGET_DIR/"

# Clean up temp directory
cd "$PROJECT_ROOT"
rm -rf "$TEMP_DIR"

# Show what was copied
echo ""
echo "=== Merged docs structure ==="
find "$TARGET_DIR" -type f -name "*.md" | sort

echo ""
echo "Done! Orchestration docs merged into docs/orchestration/"
echo ""
echo "You can now run: mkdocs serve"
