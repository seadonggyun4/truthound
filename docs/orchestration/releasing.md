# Releasing

`truthound-orchestration` publishes from the GitHub `Release Gate` workflow and uploads the built `truthound-foundation` artifact to PyPI.

## One-Time Setup

Before the first automated release, configure the repository secret:

- Secret name: `PYPI_API_TOKEN`
- Scope: repository secret on `seadonggyun4/truthound-orchestration`
- Value: the PyPI API token for the `truthound-orchestration` project

The publish workflow fails fast with a clear error if this secret is missing.

## Release Flow

1. Ensure `main` is green.
2. Create and push a version tag such as `v3.0.1`.
3. Let `Release Gate` run on the tag, or run it manually with `publish_release=true`.
4. Verify that the nested `Publish` workflow finishes successfully.

## Retry Behavior

The publish step uses `twine upload --skip-existing`.

That means a rerun is safe when:

- a previous attempt already uploaded one or more `3.x.y` files
- the release gate needs to be re-run after a transient GitHub or PyPI failure

PyPI still forbids replacing an existing file with different contents, so version numbers remain immutable.
