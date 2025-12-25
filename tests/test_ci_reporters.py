"""Tests for CI/CD Platform Reporters.

This module tests all CI reporter implementations including:
- Base CI reporter functionality
- Platform-specific reporters (GitHub, GitLab, Jenkins, Azure, CircleCI, Bitbucket)
- CI platform detection
- Reporter factory
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from unittest.mock import patch, MagicMock
from xml.etree import ElementTree as ET

import pytest

from truthound.stores.results import (
    ValidationResult,
    ValidatorResult,
    ResultStatus,
    ResultStatistics,
)

# CI Reporter imports
from truthound.reporters.ci import (
    BaseCIReporter,
    CIReporterConfig,
    CIPlatform,
    CIAnnotation,
    AnnotationLevel,
    detect_ci_platform,
    get_ci_environment,
    get_ci_reporter,
    register_ci_reporter,
    GitHubActionsReporter,
    GitLabCIReporter,
    JenkinsReporter,
    AzureDevOpsReporter,
    CircleCIReporter,
    BitbucketPipelinesReporter,
)
from truthound.reporters import get_reporter


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_validation_result() -> ValidationResult:
    """Create a sample validation result for testing."""
    return ValidationResult(
        run_id="test_run_001",
        run_time=datetime(2024, 1, 15, 10, 30, 0),
        data_asset="test_dataset.csv",
        status=ResultStatus.WARNING,
        results=[
            ValidatorResult(
                validator_name="null_check",
                success=False,
                column="email",
                issue_type="null_values",
                count=5,
                severity="high",
                message="Found 5 null values in column 'email'",
                details={"sample_values": [None, None]},
            ),
            ValidatorResult(
                validator_name="format_check",
                success=False,
                column="phone",
                issue_type="invalid_format",
                count=10,
                severity="medium",
                message="Found 10 values with invalid phone format",
                details={"expected": "XXX-XXX-XXXX", "sample_values": ["123", "abc"]},
            ),
            ValidatorResult(
                validator_name="uniqueness_check",
                success=True,
                column="id",
            ),
        ],
        statistics=ResultStatistics(
            total_validators=3,
            passed_validators=1,
            failed_validators=2,
            total_issues=15,
            critical_issues=0,
            high_issues=1,
            medium_issues=1,
            low_issues=0,
            execution_time_ms=1234.5,
        ),
    )


@pytest.fixture
def critical_validation_result() -> ValidationResult:
    """Create a validation result with critical issues."""
    return ValidationResult(
        run_id="test_run_002",
        run_time=datetime(2024, 1, 15, 11, 0, 0),
        data_asset="critical_data.csv",
        status=ResultStatus.FAILURE,
        results=[
            ValidatorResult(
                validator_name="schema_validation",
                success=False,
                column=None,
                issue_type="schema_mismatch",
                count=1,
                severity="critical",
                message="Schema does not match expected structure",
            ),
        ],
        statistics=ResultStatistics(
            total_validators=1,
            passed_validators=0,
            failed_validators=1,
            total_issues=1,
            critical_issues=1,
        ),
    )


@pytest.fixture
def passed_validation_result() -> ValidationResult:
    """Create a validation result that passed."""
    return ValidationResult(
        run_id="test_run_003",
        run_time=datetime(2024, 1, 15, 12, 0, 0),
        data_asset="clean_data.csv",
        status=ResultStatus.SUCCESS,
        results=[
            ValidatorResult(
                validator_name="null_check",
                success=True,
                column="email",
            ),
        ],
        statistics=ResultStatistics(
            total_validators=1,
            passed_validators=1,
            failed_validators=0,
            total_issues=0,
        ),
    )


# =============================================================================
# Base CI Reporter Tests
# =============================================================================


class TestAnnotationLevel:
    """Tests for AnnotationLevel enum."""

    def test_from_severity_critical(self):
        """Test critical severity maps to error."""
        assert AnnotationLevel.from_severity("critical") == AnnotationLevel.ERROR

    def test_from_severity_high(self):
        """Test high severity maps to error."""
        assert AnnotationLevel.from_severity("high") == AnnotationLevel.ERROR

    def test_from_severity_medium(self):
        """Test medium severity maps to warning."""
        assert AnnotationLevel.from_severity("medium") == AnnotationLevel.WARNING

    def test_from_severity_low(self):
        """Test low severity maps to notice."""
        assert AnnotationLevel.from_severity("low") == AnnotationLevel.NOTICE

    def test_from_severity_none(self):
        """Test None severity maps to info."""
        assert AnnotationLevel.from_severity(None) == AnnotationLevel.INFO

    def test_from_severity_case_insensitive(self):
        """Test severity mapping is case insensitive."""
        assert AnnotationLevel.from_severity("HIGH") == AnnotationLevel.ERROR
        assert AnnotationLevel.from_severity("Medium") == AnnotationLevel.WARNING


class TestCIAnnotation:
    """Tests for CIAnnotation dataclass."""

    def test_basic_annotation(self):
        """Test basic annotation creation."""
        annotation = CIAnnotation(
            message="Test message",
            level=AnnotationLevel.WARNING,
        )
        assert annotation.message == "Test message"
        assert annotation.level == AnnotationLevel.WARNING
        assert annotation.file is None

    def test_annotation_with_file_context(self):
        """Test annotation with file context."""
        annotation = CIAnnotation(
            message="Test",
            level=AnnotationLevel.ERROR,
            file="test.py",
            line=42,
            column=10,
        )
        assert annotation.file == "test.py"
        assert annotation.line == 42
        assert annotation.column == 10

    def test_with_file_context(self):
        """Test updating annotation with file context."""
        annotation = CIAnnotation(message="Test", level=AnnotationLevel.WARNING)
        updated = annotation.with_file_context(file="new.py", line=100)

        assert updated.file == "new.py"
        assert updated.line == 100
        assert annotation.file is None  # Original unchanged


class TestCIReporterConfig:
    """Tests for CIReporterConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = CIReporterConfig()
        assert config.fail_on_error is True
        assert config.fail_on_warning is False
        assert config.annotations_enabled is True
        assert config.max_annotations == 50

    def test_custom_config(self):
        """Test custom configuration."""
        config = CIReporterConfig(
            fail_on_error=False,
            max_annotations=100,
        )
        assert config.fail_on_error is False
        assert config.max_annotations == 100


# =============================================================================
# GitHub Actions Reporter Tests
# =============================================================================


class TestGitHubActionsReporter:
    """Tests for GitHub Actions reporter."""

    def test_format_annotation_simple(self, sample_validation_result):
        """Test simple annotation formatting."""
        reporter = GitHubActionsReporter()
        annotation = CIAnnotation(
            message="Test error",
            level=AnnotationLevel.ERROR,
        )
        result = reporter.format_annotation(annotation)
        assert result.startswith("::error::")
        assert "Test error" in result

    def test_format_annotation_with_location(self):
        """Test annotation with file location."""
        reporter = GitHubActionsReporter()
        annotation = CIAnnotation(
            message="Error in file",
            level=AnnotationLevel.ERROR,
            file="src/main.py",
            line=42,
            column=10,
            title="Test Error",
        )
        result = reporter.format_annotation(annotation)
        assert "file=src/main.py" in result
        assert "line=42" in result
        assert "col=10" in result
        assert "title=Test Error" in result

    def test_format_annotation_warning(self):
        """Test warning annotation."""
        reporter = GitHubActionsReporter()
        annotation = CIAnnotation(
            message="Test warning",
            level=AnnotationLevel.WARNING,
        )
        result = reporter.format_annotation(annotation)
        assert result.startswith("::warning::")

    def test_format_summary(self, sample_validation_result):
        """Test summary formatting."""
        reporter = GitHubActionsReporter()
        summary = reporter.format_summary(sample_validation_result)

        assert "Truthound Validation Report" in summary
        assert "test_dataset.csv" in summary
        assert "WARNING" in summary or "warning" in summary.lower()

    def test_format_summary_passed(self, passed_validation_result):
        """Test summary for passed validation."""
        reporter = GitHubActionsReporter()
        summary = reporter.format_summary(passed_validation_result)

        assert "✅" in summary or "Passed" in summary

    def test_format_group(self):
        """Test collapsible group formatting."""
        reporter = GitHubActionsReporter()
        start = reporter.format_group_start("Test Group")
        end = reporter.format_group_end()

        assert start == "::group::Test Group"
        assert end == "::endgroup::"

    def test_exit_code_failure(self, sample_validation_result):
        """Test exit code for failed validation."""
        reporter = GitHubActionsReporter()
        exit_code = reporter.get_exit_code(sample_validation_result)
        assert exit_code == 1

    def test_exit_code_success(self, passed_validation_result):
        """Test exit code for passed validation."""
        reporter = GitHubActionsReporter()
        exit_code = reporter.get_exit_code(passed_validation_result)
        assert exit_code == 0

    def test_render_complete(self, sample_validation_result):
        """Test complete render output."""
        reporter = GitHubActionsReporter()
        output = reporter.render(sample_validation_result)

        assert "::error" in output or "::warning" in output
        assert "Truthound" in output

    @patch.dict(os.environ, {"GITHUB_ACTIONS": "true"})
    def test_is_github_actions(self):
        """Test GitHub Actions detection."""
        assert GitHubActionsReporter.is_github_actions() is True

    @patch.dict(os.environ, {}, clear=True)
    def test_is_not_github_actions(self):
        """Test not in GitHub Actions."""
        assert GitHubActionsReporter.is_github_actions() is False


# =============================================================================
# GitLab CI Reporter Tests
# =============================================================================


class TestGitLabCIReporter:
    """Tests for GitLab CI reporter."""

    def test_format_annotation_colored(self, sample_validation_result):
        """Test ANSI colored annotation."""
        reporter = GitLabCIReporter()
        annotation = CIAnnotation(
            message="Test error",
            level=AnnotationLevel.ERROR,
            title="Error Title",
        )
        result = reporter.format_annotation(annotation)
        assert "\033[31m" in result  # Red color
        assert "ERROR" in result

    def test_generate_code_quality_report(self, sample_validation_result):
        """Test Code Quality report generation."""
        reporter = GitLabCIReporter()
        report = reporter.generate_code_quality_report(sample_validation_result)

        assert isinstance(report, list)
        assert len(report) == 2  # 2 failed validators
        assert all("description" in issue for issue in report)
        assert all("severity" in issue for issue in report)
        assert all("location" in issue for issue in report)

    def test_generate_code_quality_fingerprint(self, sample_validation_result):
        """Test fingerprint is included in Code Quality report."""
        reporter = GitLabCIReporter(include_fingerprint=True)
        report = reporter.generate_code_quality_report(sample_validation_result)

        assert all("fingerprint" in issue for issue in report)

    def test_generate_junit_report(self, sample_validation_result):
        """Test JUnit XML generation."""
        reporter = GitLabCIReporter()
        junit_xml = reporter.generate_junit_report(sample_validation_result)

        # Parse the XML
        root = ET.fromstring(junit_xml)
        assert root.tag == "testsuite"
        assert root.get("tests") == "3"
        assert root.get("failures") == "2"

    def test_format_summary(self, sample_validation_result):
        """Test summary formatting."""
        reporter = GitLabCIReporter()
        summary = reporter.format_summary(sample_validation_result)

        assert "Validation" in summary
        assert "test_dataset.csv" in summary

    @patch.dict(os.environ, {"GITLAB_CI": "true"})
    def test_is_gitlab_ci(self):
        """Test GitLab CI detection."""
        assert GitLabCIReporter.is_gitlab_ci() is True


# =============================================================================
# Jenkins Reporter Tests
# =============================================================================


class TestJenkinsReporter:
    """Tests for Jenkins reporter."""

    def test_format_annotation(self, sample_validation_result):
        """Test annotation formatting."""
        reporter = JenkinsReporter()
        annotation = CIAnnotation(
            message="Test error",
            level=AnnotationLevel.ERROR,
        )
        result = reporter.format_annotation(annotation)
        assert "[ERROR]" in result

    def test_generate_junit_report(self, sample_validation_result):
        """Test JUnit XML generation."""
        reporter = JenkinsReporter()
        junit_xml = reporter.generate_junit_report(sample_validation_result)

        # Parse the XML
        root = ET.fromstring(junit_xml)
        assert root.tag == "testsuites"

        testsuite = root.find("testsuite")
        assert testsuite is not None
        assert testsuite.get("tests") == "3"

    def test_generate_warnings_report(self, sample_validation_result):
        """Test warnings-ng JSON generation."""
        reporter = JenkinsReporter()
        report_json = reporter.generate_warnings_report(sample_validation_result)
        report = json.loads(report_json)

        assert "issues" in report
        assert len(report["issues"]) == 2

    def test_format_summary(self, sample_validation_result):
        """Test summary formatting."""
        reporter = JenkinsReporter()
        summary = reporter.format_summary(sample_validation_result)

        assert "TRUTHOUND VALIDATION REPORT" in summary
        assert "Total Validators" in summary

    @patch.dict(os.environ, {"JENKINS_URL": "http://jenkins.example.com"})
    def test_is_jenkins(self):
        """Test Jenkins detection."""
        assert JenkinsReporter.is_jenkins() is True


# =============================================================================
# Azure DevOps Reporter Tests
# =============================================================================


class TestAzureDevOpsReporter:
    """Tests for Azure DevOps reporter."""

    def test_format_annotation_vso(self, sample_validation_result):
        """Test VSO logging command format."""
        reporter = AzureDevOpsReporter()
        annotation = CIAnnotation(
            message="Test error",
            level=AnnotationLevel.ERROR,
            file="test.py",
            line=42,
        )
        result = reporter.format_annotation(annotation)

        assert result.startswith("##vso[task.logissue")
        assert "type=error" in result
        assert "sourcepath=test.py" in result
        assert "linenumber=42" in result

    def test_set_variable(self):
        """Test variable setting command."""
        reporter = AzureDevOpsReporter()
        cmd = reporter.set_variable("TEST", "value")

        assert "##vso[task.setvariable" in cmd
        assert "TRUTHOUND_TEST" in cmd
        assert "value" in cmd

    def test_generate_variable_commands(self, sample_validation_result):
        """Test variable commands generation."""
        reporter = AzureDevOpsReporter()
        commands = reporter.generate_variable_commands(sample_validation_result)

        assert len(commands) > 0
        assert any("SUCCESS" in cmd for cmd in commands)
        assert any("STATUS" in cmd for cmd in commands)

    def test_format_summary(self, sample_validation_result):
        """Test markdown summary."""
        reporter = AzureDevOpsReporter()
        summary = reporter.format_summary(sample_validation_result)

        assert "# " in summary  # Markdown header
        assert "Truthound Validation Report" in summary

    @patch.dict(os.environ, {"TF_BUILD": "True"})
    def test_is_azure_devops(self):
        """Test Azure DevOps detection."""
        assert AzureDevOpsReporter.is_azure_devops() is True


# =============================================================================
# CircleCI Reporter Tests
# =============================================================================


class TestCircleCIReporter:
    """Tests for CircleCI reporter."""

    def test_format_annotation_ansi(self, sample_validation_result):
        """Test ANSI formatted annotation."""
        reporter = CircleCIReporter()
        annotation = CIAnnotation(
            message="Test error",
            level=AnnotationLevel.ERROR,
        )
        result = reporter.format_annotation(annotation)

        assert "\033[1;31m" in result  # Bold red
        assert "ERROR" in result

    def test_generate_junit_report(self, sample_validation_result):
        """Test JUnit XML generation."""
        reporter = CircleCIReporter()
        junit_xml = reporter.generate_junit_report(sample_validation_result)

        root = ET.fromstring(junit_xml)
        assert root.tag == "testsuites"

    def test_generate_json_report(self, sample_validation_result):
        """Test JSON report generation."""
        reporter = CircleCIReporter()
        report_json = reporter.generate_json_report(sample_validation_result)
        report = json.loads(report_json)

        assert report["platform"] == "circleci"
        assert "validation" in report
        assert "statistics" in report
        assert "issues" in report

    def test_format_summary(self, sample_validation_result):
        """Test summary formatting."""
        reporter = CircleCIReporter()
        summary = reporter.format_summary(sample_validation_result)

        assert "Validation" in summary
        assert "Statistics:" in summary

    @patch.dict(os.environ, {"CIRCLECI": "true"})
    def test_is_circleci(self):
        """Test CircleCI detection."""
        assert CircleCIReporter.is_circleci() is True


# =============================================================================
# Bitbucket Pipelines Reporter Tests
# =============================================================================


class TestBitbucketPipelinesReporter:
    """Tests for Bitbucket Pipelines reporter."""

    def test_format_annotation(self, sample_validation_result):
        """Test annotation formatting."""
        reporter = BitbucketPipelinesReporter()
        annotation = CIAnnotation(
            message="Test error",
            level=AnnotationLevel.ERROR,
        )
        result = reporter.format_annotation(annotation)

        assert "✖" in result or "[ERROR]" in result

    def test_generate_report(self, sample_validation_result):
        """Test Code Insights report generation."""
        reporter = BitbucketPipelinesReporter()
        report = reporter.generate_report(sample_validation_result)

        assert "title" in report
        assert "result" in report
        assert "data" in report
        assert report["reporter"] == "Truthound"

    def test_generate_annotations(self, sample_validation_result):
        """Test annotations generation."""
        reporter = BitbucketPipelinesReporter()
        annotations = reporter.generate_annotations(sample_validation_result)

        assert len(annotations) == 2
        assert all("external_id" in ann for ann in annotations)
        assert all("severity" in ann for ann in annotations)

    def test_format_pipes_output(self, sample_validation_result):
        """Test Bitbucket Pipes output."""
        reporter = BitbucketPipelinesReporter()
        output = reporter.format_pipes_output(sample_validation_result)

        assert "TRUTHOUND_STATUS=" in output
        assert "TRUTHOUND_PASS_RATE=" in output

    def test_format_summary(self, sample_validation_result):
        """Test summary with box drawing."""
        reporter = BitbucketPipelinesReporter()
        summary = reporter.format_summary(sample_validation_result)

        assert "TRUTHOUND VALIDATION" in summary
        assert "Metric" in summary or "Total Validators" in summary

    @patch.dict(os.environ, {"BITBUCKET_BUILD_NUMBER": "123"})
    def test_is_bitbucket(self):
        """Test Bitbucket detection."""
        assert BitbucketPipelinesReporter.is_bitbucket() is True


# =============================================================================
# CI Detection Tests
# =============================================================================


class TestCIDetection:
    """Tests for CI platform detection."""

    @patch.dict(os.environ, {"GITHUB_ACTIONS": "true"}, clear=True)
    def test_detect_github_actions(self):
        """Test GitHub Actions detection."""
        platform = detect_ci_platform()
        assert platform == CIPlatform.GITHUB_ACTIONS

    @patch.dict(os.environ, {"GITLAB_CI": "true"}, clear=True)
    def test_detect_gitlab_ci(self):
        """Test GitLab CI detection."""
        platform = detect_ci_platform()
        assert platform == CIPlatform.GITLAB_CI

    @patch.dict(os.environ, {"JENKINS_URL": "http://jenkins.example.com"}, clear=True)
    def test_detect_jenkins(self):
        """Test Jenkins detection."""
        platform = detect_ci_platform()
        assert platform == CIPlatform.JENKINS

    @patch.dict(os.environ, {"TF_BUILD": "True"}, clear=True)
    def test_detect_azure_devops(self):
        """Test Azure DevOps detection."""
        platform = detect_ci_platform()
        assert platform == CIPlatform.AZURE_DEVOPS

    @patch.dict(os.environ, {"CIRCLECI": "true"}, clear=True)
    def test_detect_circleci(self):
        """Test CircleCI detection."""
        platform = detect_ci_platform()
        assert platform == CIPlatform.CIRCLECI

    @patch.dict(os.environ, {"BITBUCKET_BUILD_NUMBER": "123"}, clear=True)
    def test_detect_bitbucket(self):
        """Test Bitbucket detection."""
        platform = detect_ci_platform()
        assert platform == CIPlatform.BITBUCKET

    @patch.dict(os.environ, {}, clear=True)
    def test_detect_no_ci(self):
        """Test no CI environment."""
        platform = detect_ci_platform()
        assert platform is None

    @patch.dict(os.environ, {"CI": "true"}, clear=True)
    def test_detect_generic_ci(self):
        """Test generic CI environment."""
        platform = detect_ci_platform()
        assert platform == CIPlatform.GENERIC


class TestGetCIEnvironment:
    """Tests for get_ci_environment function."""

    @patch.dict(
        os.environ,
        {
            "GITHUB_ACTIONS": "true",
            "GITHUB_REPOSITORY": "owner/repo",
            "GITHUB_SHA": "abc123",
            "GITHUB_REF_NAME": "main",
        },
        clear=True,
    )
    def test_github_environment(self):
        """Test GitHub environment info."""
        env = get_ci_environment()

        assert env.is_ci is True
        assert env.platform == CIPlatform.GITHUB_ACTIONS
        assert env.repository == "owner/repo"
        assert env.commit == "abc123"
        assert env.branch == "main"

    @patch.dict(os.environ, {}, clear=True)
    def test_no_ci_environment(self):
        """Test no CI environment info."""
        env = get_ci_environment()

        assert env.is_ci is False
        assert env.platform == CIPlatform.GENERIC


# =============================================================================
# CI Reporter Factory Tests
# =============================================================================


class TestCIReporterFactory:
    """Tests for CI reporter factory."""

    def test_get_reporter_github(self):
        """Test getting GitHub reporter."""
        reporter = get_ci_reporter("github")
        assert isinstance(reporter, GitHubActionsReporter)

    def test_get_reporter_gitlab(self):
        """Test getting GitLab reporter."""
        reporter = get_ci_reporter("gitlab")
        assert isinstance(reporter, GitLabCIReporter)

    def test_get_reporter_jenkins(self):
        """Test getting Jenkins reporter."""
        reporter = get_ci_reporter("jenkins")
        assert isinstance(reporter, JenkinsReporter)

    def test_get_reporter_azure(self):
        """Test getting Azure DevOps reporter."""
        reporter = get_ci_reporter("azure")
        assert isinstance(reporter, AzureDevOpsReporter)

    def test_get_reporter_circleci(self):
        """Test getting CircleCI reporter."""
        reporter = get_ci_reporter("circleci")
        assert isinstance(reporter, CircleCIReporter)

    def test_get_reporter_bitbucket(self):
        """Test getting Bitbucket reporter."""
        reporter = get_ci_reporter("bitbucket")
        assert isinstance(reporter, BitbucketPipelinesReporter)

    def test_get_reporter_with_config(self):
        """Test reporter with custom configuration."""
        reporter = get_ci_reporter("github", max_annotations=100, emoji_enabled=False)
        assert reporter._config.max_annotations == 100

    def test_get_reporter_invalid_platform(self):
        """Test invalid platform raises error."""
        with pytest.raises(ValueError):
            get_ci_reporter("invalid_platform")

    @patch.dict(os.environ, {"GITHUB_ACTIONS": "true"}, clear=True)
    def test_get_reporter_auto_detect(self):
        """Test auto-detection of platform."""
        reporter = get_ci_reporter(None)
        assert isinstance(reporter, GitHubActionsReporter)

    def test_get_reporter_platform_enum(self):
        """Test using CIPlatform enum."""
        reporter = get_ci_reporter(CIPlatform.GITLAB_CI)
        assert isinstance(reporter, GitLabCIReporter)


# =============================================================================
# Main Reporter Factory Integration Tests
# =============================================================================


class TestMainReporterFactoryIntegration:
    """Tests for main get_reporter integration with CI reporters."""

    def test_get_reporter_ci_auto(self):
        """Test getting CI reporter via main factory."""
        reporter = get_reporter("ci")
        assert hasattr(reporter, "report_to_ci")

    def test_get_reporter_github(self):
        """Test getting GitHub reporter via main factory."""
        reporter = get_reporter("github")
        assert isinstance(reporter, GitHubActionsReporter)

    def test_get_reporter_gitlab(self):
        """Test getting GitLab reporter via main factory."""
        reporter = get_reporter("gitlab")
        assert isinstance(reporter, GitLabCIReporter)

    def test_get_reporter_jenkins(self):
        """Test getting Jenkins reporter via main factory."""
        reporter = get_reporter("jenkins")
        assert isinstance(reporter, JenkinsReporter)

    def test_list_formats_includes_ci(self):
        """Test list_available_formats includes CI platforms."""
        from truthound.reporters.factory import list_available_formats

        formats = list_available_formats()
        assert "ci" in formats
        assert "github" in formats
        assert "gitlab" in formats

    def test_is_format_available_ci(self):
        """Test is_format_available for CI platforms."""
        from truthound.reporters.factory import is_format_available

        assert is_format_available("ci") is True
        assert is_format_available("github") is True
        assert is_format_available("gitlab") is True


# =============================================================================
# Custom Reporter Registration Tests
# =============================================================================


class TestCustomReporterRegistration:
    """Tests for custom CI reporter registration."""

    def test_register_custom_reporter(self):
        """Test registering a custom CI reporter."""

        @register_ci_reporter("custom_ci")
        class CustomCIReporter(BaseCIReporter):
            platform = CIPlatform.GENERIC

            def format_annotation(self, annotation):
                return f"CUSTOM: {annotation.message}"

            def format_summary(self, result):
                return "Custom Summary"

        reporter = get_ci_reporter("custom_ci")
        assert isinstance(reporter, CustomCIReporter)


# =============================================================================
# Edge Cases and Error Handling Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_validation_result(self):
        """Test handling empty validation result."""
        result = ValidationResult(
            run_id="empty",
            run_time=datetime.now(),
            data_asset="empty.csv",
            status=ResultStatus.SUCCESS,
            results=[],
            statistics=ResultStatistics(),
        )

        reporter = GitHubActionsReporter()
        output = reporter.render(result)
        assert "Truthound" in output

    def test_max_annotations_limit(self, sample_validation_result):
        """Test annotation limit is respected."""
        reporter = GitHubActionsReporter(max_annotations=1)
        annotations = list(reporter.iter_annotations(sample_validation_result))
        assert len(annotations) <= 1

    def test_special_characters_in_message(self):
        """Test handling special characters in messages."""
        reporter = GitHubActionsReporter()
        annotation = CIAnnotation(
            message="Test with\nnewline and % percent",
            level=AnnotationLevel.ERROR,
        )
        result = reporter.format_annotation(annotation)

        assert "%0A" in result  # Escaped newline
        assert "%25" in result  # Escaped percent

    def test_unicode_in_message(self):
        """Test handling unicode in messages."""
        reporter = GitHubActionsReporter()
        annotation = CIAnnotation(
            message="Error: 데이터 검증 실패 🚫",
            level=AnnotationLevel.ERROR,
        )
        result = reporter.format_annotation(annotation)
        assert "데이터" in result or "%EB" in result  # Unicode or escaped

    def test_very_long_message_bitbucket(self, sample_validation_result):
        """Test Bitbucket truncates long messages."""
        # Add a validator with a very long message
        sample_validation_result.results.append(
            ValidatorResult(
                validator_name="long_message_test",
                success=False,
                message="x" * 1000,  # Very long message
                severity="high",
            )
        )

        reporter = BitbucketPipelinesReporter()
        annotations = reporter.generate_annotations(sample_validation_result)

        # Find the long message annotation
        long_ann = [a for a in annotations if a["summary"].startswith("x")]
        if long_ann:
            assert len(long_ann[0]["summary"]) <= 450


# =============================================================================
# Output File Writing Tests
# =============================================================================


class TestOutputFileWriting:
    """Tests for file output functionality."""

    def test_gitlab_write_code_quality(self, sample_validation_result, tmp_path):
        """Test writing Code Quality report file."""
        reporter = GitLabCIReporter(
            code_quality_path=str(tmp_path / "code_quality.json")
        )
        reporter._write_code_quality_artifact(sample_validation_result)

        output_file = tmp_path / "code_quality.json"
        assert output_file.exists()

        content = json.loads(output_file.read_text())
        assert isinstance(content, list)

    def test_jenkins_write_junit(self, sample_validation_result, tmp_path):
        """Test writing JUnit XML file."""
        reporter = JenkinsReporter(
            junit_path=str(tmp_path / "junit.xml")
        )
        reporter._write_junit_artifact(sample_validation_result)

        output_file = tmp_path / "junit.xml"
        assert output_file.exists()

        # Verify XML is valid
        ET.parse(str(output_file))

    def test_circleci_write_artifacts(self, sample_validation_result, tmp_path):
        """Test writing CircleCI artifacts."""
        reporter = CircleCIReporter(
            test_results_path=str(tmp_path / "test-results"),
            artifacts_path=str(tmp_path / "artifacts"),
            output_format="both",
        )
        reporter._write_junit_artifact(sample_validation_result)
        reporter._write_json_artifact(sample_validation_result)

        assert (tmp_path / "test-results" / "results.xml").exists()
        assert (tmp_path / "artifacts" / "validation-report.json").exists()
