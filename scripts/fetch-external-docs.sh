#!/bin/bash
# Fetch external docs from truthound-* repositories
# Usage: ./scripts/fetch-external-docs.sh [repo-name]
#        ./scripts/fetch-external-docs.sh              # fetch all
#        ./scripts/fetch-external-docs.sh orchestration # fetch only orchestration
#        ./scripts/fetch-external-docs.sh dashboard     # fetch only dashboard

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# External repositories configuration
# Format: "name|repo_url|target_dir"
EXTERNAL_REPOS=(
    "orchestration|https://github.com/seadonggyun4/truthound-orchestration.git|docs/orchestration"
    "dashboard|https://github.com/seadonggyun4/truthound-dashboard.git|docs/dashboard"
    # Add new repos here:
    # "new-repo|https://github.com/seadonggyun4/truthound-new-repo.git|docs/new-repo"
)

fetch_repo_docs() {
    local name="$1"
    local repo_url="$2"
    local target_dir="$3"

    local temp_dir="$PROJECT_ROOT/_temp_$name"
    local full_target="$PROJECT_ROOT/$target_dir"

    echo ""
    echo "=== Fetching $name docs ==="
    echo "Repository: $repo_url"
    echo "Target: $target_dir"

    # Clean up any existing temp directory
    rm -rf "$temp_dir"

    # Clone with sparse checkout (only docs folder)
    echo "Cloning (docs only)..."
    git clone --depth 1 --filter=blob:none --sparse \
        "$repo_url" "$temp_dir" 2>/dev/null

    cd "$temp_dir"
    git sparse-checkout set docs 2>/dev/null

    # Check if docs directory exists
    if [ ! -d "docs" ]; then
        echo "Warning: No docs directory found in $name repository"
        cd "$PROJECT_ROOT"
        rm -rf "$temp_dir"
        return 1
    fi

    # Copy docs to target directory
    echo "Copying docs..."
    mkdir -p "$full_target"

    # Remove old files except .gitkeep
    find "$full_target" -type f ! -name ".gitkeep" -delete 2>/dev/null || true
    find "$full_target" -type d -empty -delete 2>/dev/null || true

    # Copy new files
    cp -r docs/* "$full_target/"

    # Clean up temp directory
    cd "$PROJECT_ROOT"
    rm -rf "$temp_dir"

    # Show what was copied
    local file_count=$(find "$full_target" -type f -name "*.md" | wc -l | tr -d ' ')
    echo "Copied $file_count markdown files"

    return 0
}

main() {
    local filter="$1"
    local success_count=0
    local fail_count=0

    echo "========================================"
    echo "  Truthound External Docs Fetcher"
    echo "========================================"

    for repo_config in "${EXTERNAL_REPOS[@]}"; do
        IFS='|' read -r name repo_url target_dir <<< "$repo_config"

        # Skip if filter specified and doesn't match
        if [ -n "$filter" ] && [ "$filter" != "$name" ]; then
            continue
        fi

        if fetch_repo_docs "$name" "$repo_url" "$target_dir"; then
            ((success_count++))
        else
            ((fail_count++))
        fi
    done

    echo ""
    echo "========================================"
    echo "  Summary"
    echo "========================================"
    echo "Successful: $success_count"
    echo "Failed: $fail_count"
    echo ""

    if [ $success_count -gt 0 ]; then
        echo "You can now run: mkdocs serve"
    fi

    return $fail_count
}

main "$1"
