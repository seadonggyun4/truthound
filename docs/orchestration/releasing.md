# Releasing

`truthound` publishes package artifacts to PyPI from the GitHub `Release PyPI`
workflow.

## One-Time Setup

Before token-based publishing, configure the repository secret:

- Secret name: `PYPI_API_TOKEN`
- Scope: repository secret on `seadonggyun4/truthound`
- Value: the PyPI API token for the `truthound` project

The publish workflow fails fast with a clear error if this secret is missing.

PyPI publish is a package registry operation, not an application deployment.
The workflow intentionally does not use a GitHub `environment`, so PyPI release
attempts do not create entries in the repository Deployments panel. If Trusted
Publishing is used instead of the token fallback, configure the PyPI Trusted
Publisher without a GitHub environment constraint.

## Release Flow

1. Ensure `main` is green.
2. Run `Release PyPI` manually with the target version, for example `3.1.2`.
3. Use `publish_mode=token` for the repository secret fallback or
   `publish_mode=trusted` after PyPI Trusted Publishing is configured.
4. Verify that PyPI shows the uploaded version.

## Retry Behavior

PyPI release files are immutable.

That means a rerun is safe only when:

- the previous attempt failed before upload
- the version has not already been published

PyPI still forbids replacing an existing file with different contents, so version numbers remain immutable.
