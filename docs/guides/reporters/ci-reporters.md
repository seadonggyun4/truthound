# CI/CD Reporters

Truthound는 주요 CI/CD 플랫폼에 최적화된 리포터를 제공합니다.

## 지원 플랫폼

| 플랫폼 | 리포터 | 주요 기능 |
|--------|--------|----------|
| GitHub Actions | `GitHubActionsReporter` | Annotations, Step Summaries, Output Variables |
| GitLab CI | `GitLabCIReporter` | Code Quality JSON, JUnit XML, Collapsible Sections |
| Jenkins | `JenkinsReporter` | JUnit XML, warnings-ng JSON |
| Azure DevOps | `AzureDevOpsReporter` | VSO Commands, Variables, Task Results |
| CircleCI | `CircleCIReporter` | Test Metadata, Artifacts |
| Bitbucket Pipelines | `BitbucketPipelinesReporter` | Reports, Annotations |

## 자동 감지

CI 플랫폼을 자동으로 감지하여 적절한 리포터를 생성합니다:

```python
from truthound.reporters.ci import get_ci_reporter, detect_ci_platform

# 자동 감지
platform = detect_ci_platform()
reporter = get_ci_reporter()  # 감지된 플랫폼에 맞는 리포터

# 명시적 지정
reporter = get_ci_reporter("github")
reporter = get_ci_reporter("gitlab")
reporter = get_ci_reporter("jenkins")
```

### 환경 정보 조회

```python
from truthound.reporters.ci import get_ci_environment

env = get_ci_environment()
print(f"Platform: {env.platform}")
print(f"Is PR: {env.is_pr}")
print(f"Branch: {env.branch}")
print(f"Commit: {env.commit}")
print(f"Build ID: {env.build_id}")
print(f"Build URL: {env.build_url}")
```

---

## GitHub Actions

### 기본 사용법

```python
from truthound.reporters.ci import GitHubActionsReporter

reporter = GitHubActionsReporter()
exit_code = reporter.report_to_ci(validation_result)
```

### 설정 옵션

| 옵션 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `step_summary` | `bool` | `True` | GITHUB_STEP_SUMMARY 파일에 요약 작성 |
| `use_groups` | `bool` | `True` | `::group::` 명령 사용 |
| `emoji_enabled` | `bool` | `True` | 이모지 포함 |
| `set_output` | `bool` | `False` | 워크플로우 출력 변수 설정 |
| `output_name` | `str` | `"validation_result"` | 출력 변수 이름 접두사 |

### 워크플로우 예시

```yaml
# .github/workflows/validate.yml
name: Data Validation
on: [push, pull_request]

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: pip install truthound

      - name: Run validation
        run: |
          python -c "
          import truthound as th
          from truthound.reporters.ci import GitHubActionsReporter

          result = th.check('data/*.csv')
          reporter = GitHubActionsReporter(set_output=True)
          exit_code = reporter.report_to_ci(result)
          exit(exit_code)
          "

      # 출력 변수 사용 (set_output=True일 때)
      - name: Use output
        if: always()
        run: |
          echo "Success: ${{ steps.validate.outputs.validation_result_success }}"
          echo "Issues: ${{ steps.validate.outputs.validation_result_issues }}"
```

### 출력 형식

**Annotations** (코드 주석):
```
::error file=data.csv,line=10,title=NullValidator::Found 5 null values (5 occurrences)
::warning file=data.csv,line=20,title=RangeValidator::3 values out of range
```

**Step Summary** (Job Summary에 표시):
```markdown
## ✅ Truthound Validation Report

### Summary

| Metric | Value |
|--------|-------|
| **Status** | PASSED |
| **Data Asset** | `customer_data.csv` |
| **Total Validators** | 10 |
...
```

---

## GitLab CI

### 기본 사용법

```python
from truthound.reporters.ci import GitLabCIReporter

reporter = GitLabCIReporter(
    code_quality_path="gl-code-quality-report.json",
    output_format="both"  # code_quality + junit
)
exit_code = reporter.report_to_ci(validation_result)
```

### 설정 옵션

| 옵션 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `code_quality_path` | `str` | `"gl-code-quality-report.json"` | Code Quality 리포트 경로 |
| `junit_path` | `str` | `"gl-junit-report.xml"` | JUnit 리포트 경로 |
| `output_format` | `str` | `"code_quality"` | `"code_quality"`, `"junit"`, `"both"` |
| `include_fingerprint` | `bool` | `True` | 중복 제거용 fingerprint 포함 |
| `collapse_sections` | `bool` | `True` | 접이식 섹션 사용 |

### .gitlab-ci.yml 예시

```yaml
validate:
  stage: test
  script:
    - pip install truthound
    - python -c "
        import truthound as th
        from truthound.reporters.ci import GitLabCIReporter

        result = th.check('data/')
        reporter = GitLabCIReporter(output_format='both')
        exit_code = reporter.report_to_ci(result)
        exit(exit_code)
      "
  artifacts:
    reports:
      codequality: gl-code-quality-report.json
      junit: gl-junit-report.xml
    when: always
```

### Code Quality Report 형식

```json
[
  {
    "description": "Found 5 null values",
    "check_name": "NullValidator",
    "severity": "critical",
    "categories": ["Data Quality"],
    "location": {
      "path": "data.csv",
      "lines": { "begin": 1 }
    },
    "fingerprint": "abc123def456..."
  }
]
```

---

## Jenkins

### 기본 사용법

```python
from truthound.reporters.ci import JenkinsReporter

reporter = JenkinsReporter(
    junit_path="junit-report.xml",
    output_format="junit"
)
exit_code = reporter.report_to_ci(validation_result)
```

### 설정 옵션

| 옵션 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `junit_path` | `str` | `"junit-report.xml"` | JUnit XML 경로 |
| `warnings_path` | `str` | `"warnings-report.json"` | warnings-ng JSON 경로 |
| `output_format` | `str` | `"junit"` | `"junit"`, `"warnings"`, `"both"` |
| `testsuite_name` | `str` | `"Truthound Validation"` | 테스트 스위트 이름 |
| `include_stdout` | `bool` | `True` | system-out 포함 |
| `use_pipeline_steps` | `bool` | `True` | Pipeline step annotations 사용 |

### Jenkinsfile 예시

```groovy
pipeline {
    agent any

    stages {
        stage('Validate') {
            steps {
                sh '''
                    pip install truthound
                    python -c "
import truthound as th
from truthound.reporters.ci import JenkinsReporter

result = th.check('data/')
reporter = JenkinsReporter(output_format='junit')
exit_code = reporter.report_to_ci(result)
exit(exit_code)
"
                '''
            }
            post {
                always {
                    junit 'junit-report.xml'
                }
            }
        }
    }
}
```

### JUnit XML 출력

```xml
<?xml version="1.0" encoding="UTF-8"?>
<testsuites name="Truthound Validation Results" tests="10" failures="2" errors="0">
  <testsuite name="Truthound Validation" tests="10" failures="2" timestamp="2024-01-15T10:30:45">
    <properties>
      <property name="data_asset" value="customer_data.csv"/>
      <property name="run_id" value="abc123"/>
    </properties>
    <testcase classname="truthound.email" name="NullValidator" time="0.001">
      <failure type="null_values" message="Found 5 null values">
        Validator: NullValidator
        Severity: critical
        Issue Count: 5
      </failure>
    </testcase>
  </testsuite>
</testsuites>
```

---

## Azure DevOps

### 기본 사용법

```python
from truthound.reporters.ci import AzureDevOpsReporter

reporter = AzureDevOpsReporter(
    variable_prefix="TRUTHOUND",
    upload_summary=True
)
exit_code = reporter.report_to_ci(validation_result)
```

### 설정 옵션

| 옵션 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `set_variable` | `bool` | `True` | 파이프라인 변수 설정 |
| `variable_prefix` | `str` | `"TRUTHOUND"` | 변수 이름 접두사 |
| `upload_summary` | `bool` | `True` | 마크다운 요약 업로드 |
| `summary_path` | `str` | `"truthound-summary.md"` | 요약 파일 경로 |
| `use_task_commands` | `bool` | `True` | task.complete 명령 사용 |
| `timeline_records` | `bool` | `False` | 타임라인 레코드 생성 |

### azure-pipelines.yml 예시

```yaml
trigger:
  - main

pool:
  vmImage: 'ubuntu-latest'

steps:
  - task: UsePythonVersion@0
    inputs:
      versionSpec: '3.11'

  - script: |
      pip install truthound
      python -c "
      import truthound as th
      from truthound.reporters.ci import AzureDevOpsReporter

      result = th.check('data/')
      reporter = AzureDevOpsReporter()
      exit_code = reporter.report_to_ci(result)
      exit(exit_code)
      "
    displayName: 'Run validation'

  # 설정된 변수 사용
  - script: |
      echo "Success: $(TRUTHOUND_SUCCESS)"
      echo "Total Issues: $(TRUTHOUND_TOTAL_ISSUES)"
    condition: always()
    displayName: 'Check results'
```

### VSO 명령 출력

```
##vso[task.logissue type=error;sourcepath=data.csv;linenumber=10;code=NullValidator]Found 5 null values
##vso[task.setvariable variable=TRUTHOUND_SUCCESS;isOutput=true;]false
##vso[task.setvariable variable=TRUTHOUND_TOTAL_ISSUES;isOutput=true;]5
##vso[task.uploadsummary]truthound-summary.md
##vso[task.complete result=Failed;]Validation failed with critical issues
```

---

## CircleCI

### 기본 사용법

```python
from truthound.reporters.ci import CircleCIReporter

reporter = CircleCIReporter()
exit_code = reporter.report_to_ci(validation_result)
```

### config.yml 예시

```yaml
version: 2.1

jobs:
  validate:
    docker:
      - image: cimg/python:3.11
    steps:
      - checkout
      - run:
          name: Install dependencies
          command: pip install truthound
      - run:
          name: Run validation
          command: |
            python -c "
            import truthound as th
            from truthound.reporters.ci import CircleCIReporter

            result = th.check('data/')
            reporter = CircleCIReporter()
            exit_code = reporter.report_to_ci(result)
            exit(exit_code)
            "
      - store_test_results:
          path: test-results
```

---

## Bitbucket Pipelines

### 기본 사용법

```python
from truthound.reporters.ci import BitbucketPipelinesReporter

reporter = BitbucketPipelinesReporter()
exit_code = reporter.report_to_ci(validation_result)
```

### bitbucket-pipelines.yml 예시

```yaml
pipelines:
  default:
    - step:
        name: Validate data
        script:
          - pip install truthound
          - python -c "
              import truthound as th
              from truthound.reporters.ci import BitbucketPipelinesReporter

              result = th.check('data/')
              reporter = BitbucketPipelinesReporter()
              exit_code = reporter.report_to_ci(result)
              exit(exit_code)
            "
```

---

## 공통 설정 (CIReporterConfig)

모든 CI 리포터는 `CIReporterConfig` 기반 설정을 공유합니다:

| 옵션 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `fail_on_error` | `bool` | `True` | 에러 시 non-zero exit code |
| `fail_on_warning` | `bool` | `False` | 경고 시 non-zero exit code |
| `annotations_enabled` | `bool` | `True` | 코드 주석 생성 |
| `summary_enabled` | `bool` | `True` | 요약 리포트 생성 |
| `max_annotations` | `int` | `50` | 최대 주석 수 |
| `group_by_file` | `bool` | `True` | 파일별 주석 그룹화 |
| `include_passed` | `bool` | `False` | 통과 항목 포함 |
| `artifact_path` | `str \| None` | `None` | 아티팩트 경로 |
| `custom_properties` | `dict` | `{}` | 플랫폼별 커스텀 속성 |

---

## Annotation 시스템

### AnnotationLevel

검증 심각도를 CI 플랫폼 annotation 레벨로 변환합니다:

```python
from truthound.reporters.ci import AnnotationLevel

# 심각도 → 레벨 변환
level = AnnotationLevel.from_severity("critical")  # ERROR
level = AnnotationLevel.from_severity("high")      # ERROR
level = AnnotationLevel.from_severity("medium")    # WARNING
level = AnnotationLevel.from_severity("low")       # NOTICE
```

| 심각도 | Annotation Level |
|--------|------------------|
| `critical`, `high` | `ERROR` |
| `medium` | `WARNING` |
| `low` | `NOTICE` |
| 기타 | `INFO` |

### CIAnnotation

플랫폼 독립적인 주석 표현:

```python
from truthound.reporters.ci import CIAnnotation, AnnotationLevel

annotation = CIAnnotation(
    message="Found 5 null values",
    level=AnnotationLevel.ERROR,
    file="data.csv",
    line=10,
    column=5,
    title="NullValidator",
    validator_name="NullValidator",
    raw_severity="critical"
)
```

---

## 커스텀 CI Reporter 등록

```python
from truthound.reporters.ci import BaseCIReporter, register_ci_reporter, CIPlatform

@register_ci_reporter("my_ci")
class MyCIReporter(BaseCIReporter):
    platform = CIPlatform.GENERIC
    name = "my_ci"

    def format_annotation(self, annotation):
        return f"[{annotation.level.value}] {annotation.message}"

    def format_summary(self, result):
        return f"Validation: {result.status.value}"

# 사용
reporter = get_ci_reporter("my_ci")
```

## API 레퍼런스

### BaseCIReporter

```python
class BaseCIReporter(ValidationReporter[CIReporterConfig]):
    """CI 리포터 기본 클래스."""

    platform: CIPlatform = CIPlatform.GENERIC
    supports_annotations: bool = True
    supports_summary: bool = True
    max_annotations_limit: int = 50

    @abstractmethod
    def format_annotation(self, annotation: CIAnnotation) -> str:
        """플랫폼별 annotation 포맷."""
        ...

    @abstractmethod
    def format_summary(self, result: ValidationResult) -> str:
        """플랫폼별 summary 포맷."""
        ...

    def report_to_ci(self, result: ValidationResult) -> int:
        """CI에 출력하고 exit code 반환."""
        ...

    def get_exit_code(self, result: ValidationResult) -> int:
        """결과에 따른 exit code 결정."""
        ...
```
