# Cloud DW Integration Tests Setup Guide

This guide explains how to configure GitHub Actions secrets to run Cloud DW integration tests.

## Overview

The integration test workflow (`.github/workflows/integration-tests.yml`) supports:
- **BigQuery** (Google Cloud)
- **Snowflake**
- **Redshift** (AWS)
- **Databricks**

Tests run automatically on:
- Push to `main` branch (full execution)
- Pull requests (dry-run mode - no actual queries)
- Weekly schedule (Sunday midnight UTC)
- Manual trigger (workflow_dispatch)

## Required GitHub Secrets

Navigate to your repository's **Settings > Secrets and variables > Actions** to add these secrets.

### BigQuery

| Secret Name | Description | Example |
|-------------|-------------|---------|
| `GCP_SERVICE_ACCOUNT_KEY` | Service account JSON key (full content) | `{"type": "service_account", ...}` |
| `BIGQUERY_PROJECT` | GCP project ID | `my-project-123` |
| `BIGQUERY_LOCATION` | (Optional) Dataset location | `US` (default) |

**Setup Steps:**
1. Create a service account in GCP Console
2. Grant roles: `BigQuery Data Editor`, `BigQuery Job User`
3. Create and download JSON key
4. Paste the entire JSON content as `GCP_SERVICE_ACCOUNT_KEY`

### Snowflake

| Secret Name | Description | Example |
|-------------|-------------|---------|
| `SNOWFLAKE_ACCOUNT` | Account identifier | `abc12345.us-east-1` |
| `SNOWFLAKE_USER` | Username | `test_user` |
| `SNOWFLAKE_PASSWORD` | Password | `***` |
| `SNOWFLAKE_WAREHOUSE` | (Optional) Warehouse name | `COMPUTE_WH` |
| `SNOWFLAKE_DATABASE` | (Optional) Database name | `TEST_DB` |
| `SNOWFLAKE_ROLE` | (Optional) Role name | `SYSADMIN` |

**Setup Steps:**
1. Create a dedicated test user in Snowflake
2. Grant minimal permissions for test database
3. Add secrets to GitHub

### Redshift

| Secret Name | Description | Example |
|-------------|-------------|---------|
| `REDSHIFT_HOST` | Cluster endpoint | `cluster.abc123.us-east-1.redshift.amazonaws.com` |
| `REDSHIFT_DATABASE` | Database name | `dev` |
| `REDSHIFT_USER` | Username | `test_user` |
| `REDSHIFT_PASSWORD` | Password | `***` |
| `REDSHIFT_PORT` | (Optional) Port | `5439` (default) |
| `AWS_REGION` | (Optional) AWS region | `us-east-1` (default) |

**Setup Steps:**
1. Create a Redshift cluster or use existing
2. Create test user with limited permissions
3. Ensure security group allows GitHub Actions IPs

### Databricks

| Secret Name | Description | Example |
|-------------|-------------|---------|
| `DATABRICKS_HOST` | Workspace URL | `https://abc-123.cloud.databricks.com` |
| `DATABRICKS_HTTP_PATH` | SQL endpoint path | `/sql/1.0/warehouses/abc123` |
| `DATABRICKS_TOKEN` | Personal access token | `dapi...` |
| `DATABRICKS_CATALOG` | (Optional) Unity Catalog name | `main` |

**Setup Steps:**
1. Create SQL warehouse in Databricks
2. Generate personal access token
3. Note the HTTP path from SQL warehouse settings

## Cost Control

The workflow includes several cost control mechanisms:

### Automatic Dry-Run Mode
- PR tests always run in dry-run mode (no actual queries)
- Use `TRUTHOUND_TEST_DRY_RUN=true` for local dry-run testing

### Cost Limits
- Default max cost: $5.0 per test run
- Override via workflow dispatch: `max_cost` input
- Each query's estimated cost is checked before execution

### Resource Cleanup
- Weekly cleanup job removes datasets older than 24 hours
- Test datasets use prefix `truthound_test_` for easy identification

## Running Tests Locally

```bash
# Set environment variables
export BIGQUERY_PROJECT="your-project"
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/key.json"

# Run specific backend tests
pytest tests/integration/cloud_dw/ -v -m bigquery

# Run in dry-run mode
TRUTHOUND_TEST_DRY_RUN=true pytest tests/integration/cloud_dw/ -v

# Run with cost limit
TRUTHOUND_TEST_MAX_COST_USD=1.0 pytest tests/integration/cloud_dw/ -v
```

## Manual Workflow Trigger

You can manually trigger the workflow with custom options:

1. Go to **Actions > Cloud DW Integration Tests**
2. Click **Run workflow**
3. Options:
   - `backends`: Comma-separated list or "all"
   - `dry_run`: Run without executing actual queries
   - `max_cost`: Maximum cost limit in USD

## Troubleshooting

### Backend Not Running
- Check if the required secrets are set
- The preflight job logs which backends are configured

### Authentication Failures
- Verify secret values (no extra whitespace)
- Check service account permissions
- Ensure tokens haven't expired

### Cost Limit Exceeded
- Reduce test data size
- Use dry-run mode for development
- Increase `max_cost` for one-time runs

### Stale Resources
- Run cleanup manually via workflow dispatch
- Check for orphaned test datasets in cloud console

## Security Best Practices

1. **Use dedicated test credentials** - Never use production credentials
2. **Minimal permissions** - Grant only required permissions to test accounts
3. **Regular rotation** - Rotate secrets periodically
4. **Audit access** - Monitor secret usage in GitHub audit logs
5. **Environment isolation** - Use separate test projects/databases

## Viewing Results

After a test run:
1. Go to **Actions** tab
2. Click on the workflow run
3. View the **Summary** for pass/fail status per backend
4. Download **Artifacts** for detailed JUnit XML reports
