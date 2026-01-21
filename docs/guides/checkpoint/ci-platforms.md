# CI Platform Integration

Auto-detects 12 CI/CD platforms and extracts environment information.

## Supported Platforms

| Platform | Detection Variable | CIPlatform Value |
|----------|-------------------|------------------|
| GitHub Actions | `GITHUB_ACTIONS=true` | `github_actions` |
| GitLab CI | `GITLAB_CI=true` | `gitlab_ci` |
| Jenkins | `JENKINS_URL` or `BUILD_ID` | `jenkins` |
| CircleCI | `CIRCLECI=true` | `circleci` |
| Travis CI | `TRAVIS=true` | `travis_ci` |
| Azure DevOps | `TF_BUILD=True` | `azure_devops` |
| Bitbucket Pipelines | `BITBUCKET_BUILD_NUMBER` | `bitbucket_pipelines` |
| TeamCity | `TEAMCITY_VERSION` | `teamcity` |
| Buildkite | `BUILDKITE=true` | `buildkite` |
| Drone | `DRONE=true` | `drone` |
| AWS CodeBuild | `CODEBUILD_BUILD_ID` | `aws_codebuild` |
| Google Cloud Build | `BUILDER_OUTPUT` | `google_cloud_build` |

---

## Basic Usage

### Platform Detection

```python
from truthound.checkpoint.ci import (
    detect_ci_platform,
    is_ci_environment,
    get_ci_environment,
    CIPlatform,
)

# Detect current platform
platform = detect_ci_platform()
print(f"Platform: {platform}")  # CIPlatform.GITHUB_ACTIONS

# Check if running in CI
if is_ci_environment():
    print("Running in CI")

# Check if running locally
if platform == CIPlatform.LOCAL:
    print("Running locally")
```

### Getting Environment Information

```python
env = get_ci_environment()

print(f"Platform: {env.platform}")
print(f"Is CI: {env.is_ci}")
print(f"Is PR: {env.is_pr}")
print(f"Branch: {env.branch}")
print(f"Commit SHA: {env.commit_sha}")
print(f"PR Number: {env.pr_number}")
print(f"Repository: {env.repository}")
print(f"Run ID: {env.run_id}")
print(f"Run URL: {env.run_url}")
print(f"Actor: {env.actor}")
print(f"Job Name: {env.job_name}")
print(f"Workflow: {env.workflow_name}")
```

---

## CIPlatform Enum

```python
from truthound.checkpoint.ci import CIPlatform

class CIPlatform(str, Enum):
    GITHUB_ACTIONS = "github_actions"
    GITLAB_CI = "gitlab_ci"
    JENKINS = "jenkins"
    CIRCLECI = "circleci"
    TRAVIS_CI = "travis_ci"
    AZURE_DEVOPS = "azure_devops"
    BITBUCKET_PIPELINES = "bitbucket_pipelines"
    TEAMCITY = "teamcity"
    BUILDKITE = "buildkite"
    DRONE = "drone"
    AWS_CODEBUILD = "aws_codebuild"
    GOOGLE_CLOUD_BUILD = "google_cloud_build"
    LOCAL = "local"      # Local environment
    UNKNOWN = "unknown"  # CI but unsupported platform
```

---

## CIEnvironment

Dataclass containing CI environment information.

```python
@dataclass
class CIEnvironment:
    platform: CIPlatform = CIPlatform.LOCAL
    is_ci: bool = False
    is_pr: bool = False
    branch: str = ""
    commit_sha: str = ""
    commit_message: str = ""
    pr_number: int | None = None
    pr_target_branch: str = ""
    repository: str = ""
    run_id: str = ""
    run_url: str = ""
    actor: str = ""
    job_name: str = ""
    workflow_name: str = ""
    environment_vars: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        ...
```

---

## Platform-Specific Environment Variables

### GitHub Actions

| Property | Environment Variable |
|----------|---------------------|
| `branch` | `GITHUB_HEAD_REF` or `GITHUB_REF_NAME` |
| `commit_sha` | `GITHUB_SHA` |
| `pr_number` | Parsed from `GITHUB_REF` |
| `pr_target_branch` | `GITHUB_BASE_REF` |
| `repository` | `GITHUB_REPOSITORY` |
| `run_id` | `GITHUB_RUN_ID` |
| `run_url` | `{GITHUB_SERVER_URL}/{GITHUB_REPOSITORY}/actions/runs/{GITHUB_RUN_ID}` |
| `actor` | `GITHUB_ACTOR` |
| `job_name` | `GITHUB_JOB` |
| `workflow_name` | `GITHUB_WORKFLOW` |

### GitLab CI

| Property | Environment Variable |
|----------|---------------------|
| `branch` | `CI_COMMIT_REF_NAME` |
| `commit_sha` | `CI_COMMIT_SHA` |
| `commit_message` | `CI_COMMIT_MESSAGE` |
| `pr_number` | `CI_MERGE_REQUEST_IID` |
| `pr_target_branch` | `CI_MERGE_REQUEST_TARGET_BRANCH_NAME` |
| `repository` | `CI_PROJECT_PATH` |
| `run_id` | `CI_PIPELINE_ID` |
| `run_url` | `CI_PIPELINE_URL` |
| `actor` | `GITLAB_USER_LOGIN` |
| `job_name` | `CI_JOB_NAME` |
| `workflow_name` | `CI_PIPELINE_NAME` |

### Jenkins

| Property | Environment Variable |
|----------|---------------------|
| `branch` | `BRANCH_NAME` or `GIT_BRANCH` |
| `commit_sha` | `GIT_COMMIT` |
| `pr_number` | `CHANGE_ID` |
| `pr_target_branch` | `CHANGE_TARGET` |
| `run_id` | `BUILD_ID` |
| `run_url` | `BUILD_URL` |
| `actor` | `BUILD_USER` |
| `job_name` | `JOB_NAME` |

### CircleCI

| Property | Environment Variable |
|----------|---------------------|
| `branch` | `CIRCLE_BRANCH` |
| `commit_sha` | `CIRCLE_SHA1` |
| `pr_number` | `CIRCLE_PR_NUMBER` |
| `repository` | `{CIRCLE_PROJECT_USERNAME}/{CIRCLE_PROJECT_REPONAME}` |
| `run_id` | `CIRCLE_BUILD_NUM` |
| `run_url` | `CIRCLE_BUILD_URL` |
| `actor` | `CIRCLE_USERNAME` |
| `job_name` | `CIRCLE_JOB` |
| `workflow_name` | `CIRCLE_WORKFLOW_ID` |

### Travis CI

| Property | Environment Variable |
|----------|---------------------|
| `branch` | `TRAVIS_BRANCH` |
| `commit_sha` | `TRAVIS_COMMIT` |
| `commit_message` | `TRAVIS_COMMIT_MESSAGE` |
| `pr_number` | `TRAVIS_PULL_REQUEST` |
| `repository` | `TRAVIS_REPO_SLUG` |
| `run_id` | `TRAVIS_BUILD_ID` |
| `run_url` | `TRAVIS_BUILD_WEB_URL` |

### Azure DevOps

| Property | Environment Variable |
|----------|---------------------|
| `branch` | `BUILD_SOURCEBRANCHNAME` |
| `commit_sha` | `BUILD_SOURCEVERSION` |
| `commit_message` | `BUILD_SOURCEVERSIONMESSAGE` |
| `pr_number` | `SYSTEM_PULLREQUEST_PULLREQUESTNUMBER` |
| `pr_target_branch` | `SYSTEM_PULLREQUEST_TARGETBRANCH` |
| `repository` | `BUILD_REPOSITORY_NAME` |
| `run_id` | `BUILD_BUILDID` |
| `run_url` | `{SYSTEM_TEAMFOUNDATIONSERVERURI}/{SYSTEM_TEAMPROJECT}/_build/results?buildId={BUILD_BUILDID}` |
| `actor` | `BUILD_REQUESTEDFOR` |
| `job_name` | `AGENT_JOBNAME` |
| `workflow_name` | `BUILD_DEFINITIONNAME` |

### Bitbucket Pipelines

| Property | Environment Variable |
|----------|---------------------|
| `branch` | `BITBUCKET_BRANCH` |
| `commit_sha` | `BITBUCKET_COMMIT` |
| `pr_number` | `BITBUCKET_PR_ID` |
| `pr_target_branch` | `BITBUCKET_PR_DESTINATION_BRANCH` |
| `repository` | `BITBUCKET_REPO_FULL_NAME` |
| `run_id` | `BITBUCKET_BUILD_NUMBER` |

### Buildkite

| Property | Environment Variable |
|----------|---------------------|
| `branch` | `BUILDKITE_BRANCH` |
| `commit_sha` | `BUILDKITE_COMMIT` |
| `commit_message` | `BUILDKITE_MESSAGE` |
| `pr_number` | `BUILDKITE_PULL_REQUEST` |
| `pr_target_branch` | `BUILDKITE_PULL_REQUEST_BASE_BRANCH` |
| `repository` | `BUILDKITE_REPO` |
| `run_id` | `BUILDKITE_BUILD_ID` |
| `run_url` | `BUILDKITE_BUILD_URL` |
| `actor` | `BUILDKITE_BUILD_CREATOR` |
| `job_name` | `BUILDKITE_LABEL` |
| `workflow_name` | `BUILDKITE_PIPELINE_NAME` |

---

## Checkpoint Integration

### Adding CI Information to Metadata

```python
from truthound.checkpoint import Checkpoint
from truthound.checkpoint.ci import get_ci_environment

env = get_ci_environment()

checkpoint = Checkpoint(
    name="ci_validation",
    data_source="data.csv",
    validators=["null"],
    metadata={
        "ci_platform": env.platform.value,
        "branch": env.branch,
        "commit_sha": env.commit_sha,
        "pr_number": env.pr_number,
    },
)
```

### Run Only on PR Builds

```python
from truthound.checkpoint.ci import get_ci_environment

env = get_ci_environment()

if env.is_pr:
    # Run only on PR builds
    result = checkpoint.run()

    if result.status.value == "failure":
        # Post comment on PR
        post_pr_comment(
            pr_number=env.pr_number,
            message=f"Validation failed: {result.summary()}",
        )
```

### Branch-Specific Notification Conditions

```python
from truthound.checkpoint.routing import ActionRouter, Route, TagRule

env = get_ci_environment()

router = ActionRouter()

# Main branch: Slack + PagerDuty
if env.branch in ("main", "master"):
    router.add_route(Route(
        name="prod_alerts",
        rule=StatusRule(statuses=["failure"]),
        actions=[slack_action, pagerduty_action],
    ))
else:
    # Other branches: Slack only
    router.add_route(Route(
        name="dev_alerts",
        rule=StatusRule(statuses=["failure"]),
        actions=[slack_action],
    ))
```

---

## GitHub Actions Example

```yaml
# .github/workflows/data-quality.yml
name: Data Quality Check

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]
  schedule:
    - cron: '0 0 * * *'  # Daily at midnight

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: pip install truthound

      - name: Run validation
        run: |
          truthound checkpoint run production_check \
            --config truthound.yaml \
            --strict \
            --github-summary
        env:
          SLACK_WEBHOOK: ${{ secrets.SLACK_WEBHOOK }}

      - name: Upload results
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: validation-results
          path: truthound_results/
```

### GitHub Summary Output

Using the `--github-summary` flag displays results in the GitHub Actions Job Summary.

```bash
truthound checkpoint run my_check --github-summary
```

---

## GitLab CI Example

```yaml
# .gitlab-ci.yml
stages:
  - validate

data-quality:
  stage: validate
  image: python:3.11
  script:
    - pip install truthound
    - truthound checkpoint run production_check --config truthound.yaml --strict
  artifacts:
    when: always
    paths:
      - truthound_results/
    reports:
      junit: truthound_results/junit.xml
  only:
    - main
    - merge_requests
```

---

## Jenkins Example

```groovy
// Jenkinsfile
pipeline {
    agent any

    stages {
        stage('Data Quality Check') {
            steps {
                sh '''
                    pip install truthound
                    truthound checkpoint run production_check \
                        --config truthound.yaml \
                        --strict \
                        --format junit \
                        --output truthound_results/junit.xml
                '''
            }
            post {
                always {
                    junit 'truthound_results/junit.xml'
                }
            }
        }
    }
}
```

---

## CLI Options

```bash
# Strict mode (exit code 1 when issues found)
truthound checkpoint run my_check --strict

# GitHub Actions Summary
truthound checkpoint run my_check --github-summary

# JUnit report (Jenkins, GitLab CI)
truthound checkpoint run my_check --format junit --output junit.xml

# JSON output
truthound checkpoint run my_check --format json --output result.json
```

---

## Programmatic Exit Code Handling

```python
import sys
from truthound.checkpoint import Checkpoint
from truthound.checkpoint.ci import is_ci_environment

checkpoint = Checkpoint(
    name="ci_check",
    data_source="data.csv",
    validators=["null"],
)

result = checkpoint.run()

# Exit code 1 on failure in CI
if is_ci_environment():
    if result.status.value in ("failure", "error"):
        print(f"Validation failed: {result.summary()}")
        sys.exit(1)
```

---

## Platform-Specific Features

### GitHub Actions

- `--github-summary`: Outputs markdown results to Job Summary
- Can set output variables to `GITHUB_OUTPUT`

### GitLab CI

- Display test results in Merge Request via JUnit report
- Store results as artifacts

### Jenkins

- Visualize test results with JUnit plugin
- Display status per pipeline stage

### Azure DevOps

- Display JUnit results in Test Results tab
- Show status in pipeline summary
