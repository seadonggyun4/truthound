# CI/CD Reporters

Truthound provides reporters optimized for major CI/CD platforms.

## Supported Platforms

| Platform | Reporter | Key Features |
|----------|----------|--------------|
| GitHub Actions | `GitHubActionsReporter` | Annotations, Step Summaries, Output Variables |
| GitLab CI | `GitLabCIReporter` | Code Quality JSON, JUnit XML, Collapsible Sections |
| Jenkins | `JenkinsReporter` | JUnit XML, warnings-ng JSON |
| Azure DevOps | `AzureDevOpsReporter` | VSO Commands, Variables, Task Results |
| CircleCI | `CircleCIReporter` | Test Metadata, Artifacts |
| Bitbucket Pipelines | `BitbucketPipelinesReporter` | Reports, Annotations |

## Auto-Detection

Automatically detects the CI platform and creates the appropriate reporter:

```python
from truthound.reporters.ci import get_ci_reporter, detect_ci_platform

# Auto-detect
platform = detect_ci_platform()
reporter = get_ci_reporter()  # Reporter for detected platform

# Explicit specification
reporter = get_ci_reporter("github")
reporter = get_ci_reporter("gitlab")
reporter = get_ci_reporter("jenkins")
```

### Retrieve Environment Information

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

### Basic Usage

```python
from truthound.reporters.ci import GitHubActionsReporter

reporter = GitHubActionsReporter()
exit_code = reporter.report_to_ci(validation_result)
```

### Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `step_summary` | `bool` | `True` | Write summary to GITHUB_STEP_SUMMARY file |
| `use_groups` | `bool` | `True` | Use `::group::` commands |
| `emoji_enabled` | `bool` | `True` | Include emojis |
| `set_output` | `bool` | `False` | Set workflow output variables |
| `output_name` | `str` | `"validation_result"` | Output variable name prefix |

### Workflow Example

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

      # Use output variables (when set_output=True)
      - name: Use output
        if: always()
        run: |
          echo "Success: ${{ steps.validate.outputs.validation_result_success }}"
          echo "Issues: ${{ steps.validate.outputs.validation_result_issues }}"
```

### Output Format

**Annotations** (code comments):
```
::error file=data.csv,line=10,title=NullValidator::Found 5 null values (5 occurrences)
::warning file=data.csv,line=20,title=RangeValidator::3 values out of range
```

**Step Summary** (displayed in Job Summary):
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

### Basic Usage

```python
from truthound.reporters.ci import GitLabCIReporter

reporter = GitLabCIReporter(
    code_quality_path="gl-code-quality-report.json",
    output_format="both"  # code_quality + junit
)
exit_code = reporter.report_to_ci(validation_result)
```

### Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `code_quality_path` | `str` | `"gl-code-quality-report.json"` | Code Quality report path |
| `junit_path` | `str` | `"gl-junit-report.xml"` | JUnit report path |
| `output_format` | `str` | `"code_quality"` | `"code_quality"`, `"junit"`, `"both"` |
| `include_fingerprint` | `bool` | `True` | Include fingerprint for deduplication |
| `collapse_sections` | `bool` | `True` | Use collapsible sections |

### .gitlab-ci.yml Example

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

### Code Quality Report Format

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

### Basic Usage

```python
from truthound.reporters.ci import JenkinsReporter

reporter = JenkinsReporter(
    junit_path="junit-report.xml",
    output_format="junit"
)
exit_code = reporter.report_to_ci(validation_result)
```

### Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `junit_path` | `str` | `"junit-report.xml"` | JUnit XML path |
| `warnings_path` | `str` | `"warnings-report.json"` | warnings-ng JSON path |
| `output_format` | `str` | `"junit"` | `"junit"`, `"warnings"`, `"both"` |
| `testsuite_name` | `str` | `"Truthound Validation"` | Test suite name |
| `include_stdout` | `bool` | `True` | Include system-out |
| `use_pipeline_steps` | `bool` | `True` | Use Pipeline step annotations |

### Jenkinsfile Example

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

### JUnit XML Output

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

### Basic Usage

```python
from truthound.reporters.ci import AzureDevOpsReporter

reporter = AzureDevOpsReporter(
    variable_prefix="TRUTHOUND",
    upload_summary=True
)
exit_code = reporter.report_to_ci(validation_result)
```

### Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `set_variable` | `bool` | `True` | Set pipeline variables |
| `variable_prefix` | `str` | `"TRUTHOUND"` | Variable name prefix |
| `upload_summary` | `bool` | `True` | Upload markdown summary |
| `summary_path` | `str` | `"truthound-summary.md"` | Summary file path |
| `use_task_commands` | `bool` | `True` | Use task.complete commands |
| `timeline_records` | `bool` | `False` | Create timeline records |

### azure-pipelines.yml Example

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

  # Use set variables
  - script: |
      echo "Success: $(TRUTHOUND_SUCCESS)"
      echo "Total Issues: $(TRUTHOUND_TOTAL_ISSUES)"
    condition: always()
    displayName: 'Check results'
```

### VSO Command Output

```
##vso[task.logissue type=error;sourcepath=data.csv;linenumber=10;code=NullValidator]Found 5 null values
##vso[task.setvariable variable=TRUTHOUND_SUCCESS;isOutput=true;]false
##vso[task.setvariable variable=TRUTHOUND_TOTAL_ISSUES;isOutput=true;]5
##vso[task.uploadsummary]truthound-summary.md
##vso[task.complete result=Failed;]Validation failed with critical issues
```

---

## CircleCI

### Basic Usage

```python
from truthound.reporters.ci import CircleCIReporter

reporter = CircleCIReporter()
exit_code = reporter.report_to_ci(validation_result)
```

### config.yml Example

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

### Basic Usage

```python
from truthound.reporters.ci import BitbucketPipelinesReporter

reporter = BitbucketPipelinesReporter()
exit_code = reporter.report_to_ci(validation_result)
```

### bitbucket-pipelines.yml Example

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

## Common Configuration (CIReporterConfig)

All CI reporters share `CIReporterConfig`-based settings:

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `fail_on_error` | `bool` | `True` | Non-zero exit code on error |
| `fail_on_warning` | `bool` | `False` | Non-zero exit code on warning |
| `annotations_enabled` | `bool` | `True` | Generate code annotations |
| `summary_enabled` | `bool` | `True` | Generate summary report |
| `max_annotations` | `int` | `50` | Maximum annotation count |
| `group_by_file` | `bool` | `True` | Group annotations by file |
| `include_passed` | `bool` | `False` | Include passed items |
| `artifact_path` | `str \| None` | `None` | Artifact path |
| `custom_properties` | `dict` | `{}` | Platform-specific custom properties |

---

## Annotation System

### AnnotationLevel

Converts validation severity to CI platform annotation levels:

```python
from truthound.reporters.ci import AnnotationLevel

# Severity → Level conversion
level = AnnotationLevel.from_severity("critical")  # ERROR
level = AnnotationLevel.from_severity("high")      # ERROR
level = AnnotationLevel.from_severity("medium")    # WARNING
level = AnnotationLevel.from_severity("low")       # NOTICE
```

| Severity | Annotation Level |
|----------|------------------|
| `critical`, `high` | `ERROR` |
| `medium` | `WARNING` |
| `low` | `NOTICE` |
| Other | `INFO` |

### CIAnnotation

Platform-independent annotation representation:

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

## Custom CI Reporter Registration

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

# Usage
reporter = get_ci_reporter("my_ci")
```

## API Reference

### BaseCIReporter

```python
class BaseCIReporter(ValidationReporter[CIReporterConfig]):
    """Base class for CI reporters."""

    platform: CIPlatform = CIPlatform.GENERIC
    supports_annotations: bool = True
    supports_summary: bool = True
    max_annotations_limit: int = 50

    @abstractmethod
    def format_annotation(self, annotation: CIAnnotation) -> str:
        """Platform-specific annotation format."""
        ...

    @abstractmethod
    def format_summary(self, result: ValidationResult) -> str:
        """Platform-specific summary format."""
        ...

    def report_to_ci(self, result: ValidationResult) -> int:
        """Output to CI and return exit code."""
        ...

    def get_exit_code(self, result: ValidationResult) -> int:
        """Determine exit code based on result."""
        ...
```
