#!/bin/bash
# Sync checked-in external docs snapshots from truthound-* repositories.
# Netlify builds use the committed snapshot only; this script should be run
# before commit or release when an external docs mirror needs to be refreshed.
#
# Usage: ./scripts/fetch-external-docs.sh [repo-name]
#        ./scripts/fetch-external-docs.sh               # sync all configured repos
#        ./scripts/fetch-external-docs.sh orchestration # sync only orchestration

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
WORKSPACE_ROOT="$(dirname "$PROJECT_ROOT")"

# External repositories configuration
# Format: "name|repo_url|target_dir|sync_mode"
EXTERNAL_REPOS=(
    "orchestration|https://github.com/seadonggyun4/truthound-orchestration.git|docs/orchestration|public_nav"
    # Add new repos here:
    # "new-repo|https://github.com/seadonggyun4/truthound-new-repo.git|docs/new-repo|docs_tree"
)

resolve_local_checkout() {
    local repo_url="$1"
    local repo_name
    repo_name="$(basename "$repo_url" .git)"
    local candidate="$WORKSPACE_ROOT/$repo_name"

    if [ -f "$candidate/mkdocs.yml" ] && [ -d "$candidate/docs" ]; then
        printf '%s\n' "$candidate"
        return 0
    fi

    return 1
}

clone_repo_snapshot() {
    local repo_url="$1"
    local temp_dir="$2"
    local sync_mode="$3"

    rm -rf "$temp_dir"

    echo "Cloning snapshot..."
    git clone --depth 1 --filter=blob:none --sparse \
        "$repo_url" "$temp_dir" 2>/dev/null

    cd "$temp_dir"
    if [ "$sync_mode" = "public_nav" ]; then
        git sparse-checkout set docs mkdocs.yml 2>/dev/null
    else
        git sparse-checkout set docs 2>/dev/null
    fi
    printf '%s\n' "$temp_dir"
}

sync_docs_tree() {
    local source_root="$1"
    local target_dir="$2"

    mkdir -p "$target_dir"
    find "$target_dir" -type f ! -name ".gitkeep" -delete 2>/dev/null || true
    find "$target_dir" -type d -empty -delete 2>/dev/null || true
    cp -r "$source_root/docs/"* "$target_dir/"
}

fetch_repo_docs() {
    local name="$1"
    local repo_url="$2"
    local target_dir="$3"
    local sync_mode="$4"

    local temp_dir="$PROJECT_ROOT/_temp_$name"
    local full_target="$PROJECT_ROOT/$target_dir"
    local source_root=""
    local used_temp_checkout="false"

    echo ""
    echo "=== Syncing $name docs snapshot ==="
    echo "Repository: $repo_url"
    echo "Target: $target_dir"
    echo "Mode: $sync_mode"

    if source_root="$(resolve_local_checkout "$repo_url")"; then
        echo "Using local checkout: $source_root"
    else
        source_root="$(clone_repo_snapshot "$repo_url" "$temp_dir" "$sync_mode")"
        used_temp_checkout="true"
        echo "Using temporary clone: $source_root"
    fi

    if [ ! -d "$source_root/docs" ]; then
        echo "Warning: No docs directory found in $name repository"
        cd "$PROJECT_ROOT"
        if [ "$used_temp_checkout" = "true" ]; then
            rm -rf "$temp_dir"
        fi
        return 1
    fi

    if [ "$sync_mode" = "public_nav" ]; then
        python3 "$PROJECT_ROOT/docs/scripts/sync_external_docs.py" \
            --source-root "$source_root" \
            --target-dir "$full_target"
    else
        sync_docs_tree "$source_root" "$full_target"
    fi

    cd "$PROJECT_ROOT"
    if [ "$used_temp_checkout" = "true" ]; then
        rm -rf "$temp_dir"
    fi

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
        IFS='|' read -r name repo_url target_dir sync_mode <<< "$repo_config"

        # Skip if filter specified and doesn't match
        if [ -n "$filter" ] && [ "$filter" != "$name" ]; then
            continue
        fi

        if fetch_repo_docs "$name" "$repo_url" "$target_dir" "$sync_mode"; then
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
