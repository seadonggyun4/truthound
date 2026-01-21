# CI Platform Integration

12개 CI/CD 플랫폼을 자동 감지하고 환경 정보를 추출합니다.

## 지원 플랫폼

| 플랫폼 | 감지 변수 | CIPlatform 값 |
|--------|----------|---------------|
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

## 기본 사용법

### 플랫폼 감지

```python
from truthound.checkpoint.ci import (
    detect_ci_platform,
    is_ci_environment,
    get_ci_environment,
    CIPlatform,
)

# 현재 플랫폼 감지
platform = detect_ci_platform()
print(f"Platform: {platform}")  # CIPlatform.GITHUB_ACTIONS

# CI 환경 여부
if is_ci_environment():
    print("Running in CI")

# 로컬 환경 확인
if platform == CIPlatform.LOCAL:
    print("Running locally")
```

### 환경 정보 가져오기

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
    LOCAL = "local"      # 로컬 환경
    UNKNOWN = "unknown"  # CI이지만 미지원 플랫폼
```

---

## CIEnvironment

CI 환경 정보 데이터클래스입니다.

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
        """딕셔너리로 변환."""
        ...
```

---

## 플랫폼별 환경 변수

### GitHub Actions

| 속성 | 환경 변수 |
|------|----------|
| `branch` | `GITHUB_HEAD_REF` or `GITHUB_REF_NAME` |
| `commit_sha` | `GITHUB_SHA` |
| `pr_number` | `GITHUB_REF`에서 파싱 |
| `pr_target_branch` | `GITHUB_BASE_REF` |
| `repository` | `GITHUB_REPOSITORY` |
| `run_id` | `GITHUB_RUN_ID` |
| `run_url` | `{GITHUB_SERVER_URL}/{GITHUB_REPOSITORY}/actions/runs/{GITHUB_RUN_ID}` |
| `actor` | `GITHUB_ACTOR` |
| `job_name` | `GITHUB_JOB` |
| `workflow_name` | `GITHUB_WORKFLOW` |

### GitLab CI

| 속성 | 환경 변수 |
|------|----------|
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

| 속성 | 환경 변수 |
|------|----------|
| `branch` | `BRANCH_NAME` or `GIT_BRANCH` |
| `commit_sha` | `GIT_COMMIT` |
| `pr_number` | `CHANGE_ID` |
| `pr_target_branch` | `CHANGE_TARGET` |
| `run_id` | `BUILD_ID` |
| `run_url` | `BUILD_URL` |
| `actor` | `BUILD_USER` |
| `job_name` | `JOB_NAME` |

### CircleCI

| 속성 | 환경 변수 |
|------|----------|
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

| 속성 | 환경 변수 |
|------|----------|
| `branch` | `TRAVIS_BRANCH` |
| `commit_sha` | `TRAVIS_COMMIT` |
| `commit_message` | `TRAVIS_COMMIT_MESSAGE` |
| `pr_number` | `TRAVIS_PULL_REQUEST` |
| `repository` | `TRAVIS_REPO_SLUG` |
| `run_id` | `TRAVIS_BUILD_ID` |
| `run_url` | `TRAVIS_BUILD_WEB_URL` |

### Azure DevOps

| 속성 | 환경 변수 |
|------|----------|
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

| 속성 | 환경 변수 |
|------|----------|
| `branch` | `BITBUCKET_BRANCH` |
| `commit_sha` | `BITBUCKET_COMMIT` |
| `pr_number` | `BITBUCKET_PR_ID` |
| `pr_target_branch` | `BITBUCKET_PR_DESTINATION_BRANCH` |
| `repository` | `BITBUCKET_REPO_FULL_NAME` |
| `run_id` | `BITBUCKET_BUILD_NUMBER` |

### Buildkite

| 속성 | 환경 변수 |
|------|----------|
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

## Checkpoint와 통합

### CI 정보 메타데이터 추가

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

### PR 빌드에서만 실행

```python
from truthound.checkpoint.ci import get_ci_environment

env = get_ci_environment()

if env.is_pr:
    # PR 빌드에서만 실행
    result = checkpoint.run()

    if result.status.value == "failure":
        # PR에 코멘트 남기기
        post_pr_comment(
            pr_number=env.pr_number,
            message=f"Validation failed: {result.summary()}",
        )
```

### 브랜치별 알림 조건

```python
from truthound.checkpoint.routing import ActionRouter, Route, TagRule

env = get_ci_environment()

router = ActionRouter()

# main 브랜치는 Slack + PagerDuty
if env.branch in ("main", "master"):
    router.add_route(Route(
        name="prod_alerts",
        rule=StatusRule(statuses=["failure"]),
        actions=[slack_action, pagerduty_action],
    ))
else:
    # 다른 브랜치는 Slack만
    router.add_route(Route(
        name="dev_alerts",
        rule=StatusRule(statuses=["failure"]),
        actions=[slack_action],
    ))
```

---

## GitHub Actions 예시

```yaml
# .github/workflows/data-quality.yml
name: Data Quality Check

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]
  schedule:
    - cron: '0 0 * * *'  # 매일 자정

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

### GitHub Summary 출력

`--github-summary` 플래그를 사용하면 GitHub Actions Job Summary에 결과가 표시됩니다.

```bash
truthound checkpoint run my_check --github-summary
```

---

## GitLab CI 예시

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

## Jenkins 예시

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

## CLI 옵션

```bash
# 엄격 모드 (이슈 발견 시 exit code 1)
truthound checkpoint run my_check --strict

# GitHub Actions Summary
truthound checkpoint run my_check --github-summary

# JUnit 보고서 (Jenkins, GitLab CI)
truthound checkpoint run my_check --format junit --output junit.xml

# JSON 출력
truthound checkpoint run my_check --format json --output result.json
```

---

## 프로그래밍 방식 Exit Code 처리

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

# CI에서 실패 시 exit code 1
if is_ci_environment():
    if result.status.value in ("failure", "error"):
        print(f"Validation failed: {result.summary()}")
        sys.exit(1)
```

---

## 플랫폼별 특수 기능

### GitHub Actions

- `--github-summary`: Job Summary에 마크다운 결과 출력
- `GITHUB_OUTPUT`에 출력 변수 설정 가능

### GitLab CI

- JUnit 보고서로 Merge Request에 테스트 결과 표시
- Artifacts로 결과 보관

### Jenkins

- JUnit 플러그인으로 테스트 결과 시각화
- Pipeline 단계별 상태 표시

### Azure DevOps

- Test Results 탭에 JUnit 결과 표시
- Pipeline 요약에 상태 표시
