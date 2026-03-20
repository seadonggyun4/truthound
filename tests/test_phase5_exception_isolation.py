"""Tests for PHASE 5: Exception Isolation, Auto Retry, Resilience Integration.

Covers:
    TASK 5-1: ExceptionInfo structure
    TASK 5-2: ValidatorExecutionResult extension
    TASK 5-3: ValidationIssue exception_info
    TASK 5-4: Expression-level partial failure (3-tier fallback)
    TASK 5-5: _validate_safe with auto retry
    TASK 5-6: ValidatorConfig PHASE 5 fields
    TASK 5-7: check() API catch_exceptions / max_retries
    TASK 5-8: Resilience bridge (ValidationResiliencePolicy)
    TASK 5-9: Report ExceptionSummary
    TASK 5-10: CLI options (--catch-exceptions, --max-retries, --show-exceptions)
"""

import json
import time
from unittest.mock import MagicMock, patch

import polars as pl
import pytest

from truthound.types import ResultFormat, ResultFormatConfig, Severity


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_lf():
    """Sample LazyFrame for testing."""
    return pl.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "name": ["Alice", "Bob", None, "Dave", "Eve"],
        "age": [25, 30, None, 40, 45],
        "score": [90.5, 85.0, 78.3, 92.1, 88.7],
    }).lazy()


@pytest.fixture
def sample_df():
    """Sample DataFrame for testing."""
    return pl.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "name": ["Alice", "Bob", None, "Dave", "Eve"],
        "age": [25, 30, None, 40, 45],
        "score": [90.5, 85.0, 78.3, 92.1, 88.7],
    })


# ============================================================================
# TASK 5-1: ExceptionInfo
# ============================================================================

class TestExceptionInfo:
    """Tests for ExceptionInfo dataclass."""

    def test_from_exception_basic(self):
        from truthound.validators.base import ExceptionInfo

        exc = ValueError("bad value")
        info = ExceptionInfo.from_exception(exc, validator_name="test_v")

        assert info.raised_exception is True
        assert info.exception_type == "ValueError"
        assert info.exception_message == "bad value"
        assert info.validator_name == "test_v"
        assert info.failure_category == "configuration"
        assert info.is_retryable is False

    def test_from_exception_transient(self):
        from truthound.validators.base import ExceptionInfo

        exc = ConnectionError("network down")
        info = ExceptionInfo.from_exception(exc)

        assert info.failure_category == "transient"
        assert info.is_retryable is True

    def test_from_exception_timeout(self):
        from truthound.validators.base import ExceptionInfo, ValidationTimeoutError

        exc = ValidationTimeoutError(30.0, "slow_validator")
        info = ExceptionInfo.from_exception(exc)

        assert info.failure_category == "transient"
        assert info.is_retryable is True

    def test_classify_column_not_found(self):
        from truthound.validators.base import ExceptionInfo, ColumnNotFoundError

        exc = ColumnNotFoundError("missing_col", ["a", "b"])
        info = ExceptionInfo.from_exception(exc)

        assert info.failure_category == "configuration"
        assert info.is_retryable is False

    def test_classify_polars_error(self):
        from truthound.validators.base import ExceptionInfo
        import polars.exceptions as pl_exc

        exc = pl_exc.ComputeError("invalid computation")
        info = ExceptionInfo.from_exception(exc)

        assert info.failure_category == "data"

    def test_classify_unknown(self):
        from truthound.validators.base import ExceptionInfo

        exc = RuntimeError("something weird")
        info = ExceptionInfo.from_exception(exc)

        assert info.failure_category == "permanent"

    def test_to_dict_minimal(self):
        from truthound.validators.base import ExceptionInfo

        info = ExceptionInfo()
        d = info.to_dict()
        assert d == {}

    def test_to_dict_full(self):
        from truthound.validators.base import ExceptionInfo

        exc = ValueError("bad")
        info = ExceptionInfo.from_exception(exc, validator_name="v", column="col")
        info.retry_count = 2
        info.max_retries = 3

        d = info.to_dict()
        assert d["raised_exception"] is True
        assert d["exception_type"] == "ValueError"
        assert d["retry_count"] == 2
        assert d["max_retries"] == 3
        assert d["validator_name"] == "v"
        assert d["column"] == "col"
        assert d["failure_category"] == "configuration"

    def test_to_error_context_compatibility(self):
        from truthound.validators.base import ExceptionInfo, ErrorContext

        exc = ValueError("test")
        info = ExceptionInfo.from_exception(exc)
        ctx = info.to_error_context()

        assert isinstance(ctx, ErrorContext)
        assert ctx.error_type == "ValueError"
        assert ctx.message == "test"


# ============================================================================
# TASK 5-2: ValidatorExecutionResult extension
# ============================================================================

class TestValidatorExecutionResult:
    """Tests for extended ValidatorExecutionResult."""

    def test_has_exception_false(self):
        from truthound.validators.base import (
            ValidatorExecutionResult, ValidationResult,
        )

        result = ValidatorExecutionResult(
            validator_name="test",
            status=ValidationResult.SUCCESS,
            issues=[],
        )
        assert result.has_exception is False

    def test_has_exception_true(self):
        from truthound.validators.base import (
            ValidatorExecutionResult, ValidationResult, ExceptionInfo,
        )

        exc_info = ExceptionInfo.from_exception(ValueError("x"))
        result = ValidatorExecutionResult(
            validator_name="test",
            status=ValidationResult.FAILED,
            issues=[],
            exception_info=exc_info,
        )
        assert result.has_exception is True

    def test_is_partial(self):
        from truthound.validators.base import (
            ValidatorExecutionResult, ValidationResult, ValidationIssue,
        )

        issue = ValidationIssue(column="a", issue_type="t", count=1, severity=Severity.LOW)
        result = ValidatorExecutionResult(
            validator_name="test",
            status=ValidationResult.PARTIAL,
            issues=[issue],
            partial_issues=[issue],
        )
        assert result.is_partial is True

    def test_to_dict_with_exception(self):
        from truthound.validators.base import (
            ValidatorExecutionResult, ValidationResult, ExceptionInfo,
        )

        exc_info = ExceptionInfo.from_exception(ValueError("x"))
        result = ValidatorExecutionResult(
            validator_name="test",
            status=ValidationResult.FAILED,
            issues=[],
            exception_info=exc_info,
            retry_count=2,
        )
        d = result.to_dict()
        assert "exception_info" in d
        assert d["retry_count"] == 2


# ============================================================================
# TASK 5-3: ValidationIssue exception_info
# ============================================================================

class TestValidationIssueExceptionInfo:
    """Tests for ValidationIssue.exception_info field."""

    def test_default_none(self):
        from truthound.validators.base import ValidationIssue

        issue = ValidationIssue(column="a", issue_type="t", count=1, severity=Severity.LOW)
        assert issue.exception_info is None

    def test_with_exception_info(self):
        from truthound.validators.base import ValidationIssue, ExceptionInfo

        exc_info = ExceptionInfo.from_exception(ValueError("x"))
        issue = ValidationIssue(
            column="a", issue_type="t", count=1,
            severity=Severity.LOW, exception_info=exc_info,
        )
        assert issue.exception_info is not None
        assert issue.exception_info.exception_type == "ValueError"

    def test_to_dict_includes_exception_info(self):
        from truthound.validators.base import ValidationIssue, ExceptionInfo

        exc_info = ExceptionInfo.from_exception(ValueError("x"))
        issue = ValidationIssue(
            column="a", issue_type="t", count=1,
            severity=Severity.LOW, exception_info=exc_info,
        )
        d = issue.to_dict()
        assert "exception_info" in d
        assert d["exception_info"]["exception_type"] == "ValueError"


# ============================================================================
# TASK 5-5: _validate_safe with auto retry
# ============================================================================

class TestValidateSafeRetry:
    """Tests for _validate_safe retry behaviour."""

    def test_no_retry_on_success(self, sample_lf):
        from truthound.validators.base import _validate_safe, ValidationResult
        from truthound.validators.completeness.null import NullValidator

        validator = NullValidator()
        result = _validate_safe(validator, sample_lf, max_retries=3)
        assert result.status == ValidationResult.SUCCESS
        assert result.retry_count == 0

    def test_retry_on_transient_error(self, sample_lf):
        from truthound.validators.base import (
            _validate_safe, ValidationResult, Validator, ValidatorConfig,
        )

        class FlakyValidator(Validator):
            name = "flaky"
            _call_count = 0

            def validate(self, lf):
                self._call_count += 1
                if self._call_count < 3:
                    raise ConnectionError("network error")
                return []

        v = FlakyValidator()
        result = _validate_safe(v, sample_lf, max_retries=3)
        assert result.status == ValidationResult.SUCCESS
        assert result.retry_count == 2
        assert v._call_count == 3

    def test_no_retry_on_config_error(self, sample_lf):
        from truthound.validators.base import (
            _validate_safe, ValidationResult, Validator,
        )

        class BadConfigValidator(Validator):
            name = "bad_config"

            def validate(self, lf):
                raise ValueError("bad config")

        v = BadConfigValidator()
        result = _validate_safe(v, sample_lf, max_retries=3)
        assert result.status == ValidationResult.FAILED
        assert result.retry_count == 0
        assert result.exception_info is not None
        assert result.exception_info.failure_category == "configuration"

    def test_column_not_found_skipped(self, sample_lf):
        from truthound.validators.base import (
            _validate_safe, ValidationResult, Validator, ColumnNotFoundError,
        )

        class MissingColValidator(Validator):
            name = "missing_col"

            def validate(self, lf):
                raise ColumnNotFoundError("nonexistent", ["id", "name"])

        v = MissingColValidator()
        result = _validate_safe(v, sample_lf, max_retries=3)
        assert result.status == ValidationResult.SKIPPED
        assert result.exception_info is not None
        assert result.exception_info.failure_category == "configuration"

    def test_all_retries_exhausted(self, sample_lf):
        from truthound.validators.base import (
            _validate_safe, ValidationResult, Validator,
        )

        class AlwaysFail(Validator):
            name = "always_fail"
            _count = 0

            def validate(self, lf):
                self._count += 1
                raise ConnectionError("always fails")

        v = AlwaysFail()
        result = _validate_safe(v, sample_lf, max_retries=2)
        assert result.status == ValidationResult.FAILED
        assert v._count == 3  # 1 initial + 2 retries
        assert result.retry_count == 3
        assert result.exception_info is not None
        assert result.exception_info.retry_count == 3
        assert result.exception_info.max_retries == 2

    def test_skip_on_error_false_raises(self, sample_lf):
        from truthound.validators.base import _validate_safe, Validator

        class RaiseValidator(Validator):
            name = "raiser"

            def validate(self, lf):
                raise RuntimeError("boom")

        v = RaiseValidator()
        with pytest.raises(RuntimeError, match="boom"):
            _validate_safe(v, sample_lf, skip_on_error=False)


# ============================================================================
# TASK 5-6: ValidatorConfig PHASE 5 fields
# ============================================================================

class TestValidatorConfigPhase5:
    """Tests for ValidatorConfig PHASE 5 fields."""

    def test_default_values(self):
        from truthound.validators.base import ValidatorConfig

        config = ValidatorConfig()
        assert config.catch_exceptions is True
        assert config.max_retries == 0
        assert config.partial_failure_mode == "collect"

    def test_custom_values(self):
        from truthound.validators.base import ValidatorConfig

        config = ValidatorConfig(
            catch_exceptions=False,
            max_retries=5,
            partial_failure_mode="skip",
        )
        assert config.catch_exceptions is False
        assert config.max_retries == 5
        assert config.partial_failure_mode == "skip"

    def test_invalid_max_retries(self):
        from truthound.validators.base import ValidatorConfig

        with pytest.raises(ValueError, match="max_retries"):
            ValidatorConfig(max_retries=-1)

    def test_invalid_partial_failure_mode(self):
        from truthound.validators.base import ValidatorConfig

        with pytest.raises(ValueError, match="partial_failure_mode"):
            ValidatorConfig(partial_failure_mode="invalid")

    def test_replace_preserves_phase5_fields(self):
        from truthound.validators.base import ValidatorConfig

        config = ValidatorConfig(max_retries=3, catch_exceptions=False)
        new_config = config.replace(sample_size=10)
        assert new_config.max_retries == 3
        assert new_config.catch_exceptions is False
        assert new_config.sample_size == 10

    def test_from_kwargs_includes_phase5_fields(self):
        from truthound.validators.base import ValidatorConfig

        config = ValidatorConfig.from_kwargs(
            max_retries=2,
            catch_exceptions=False,
            partial_failure_mode="raise",
            unknown_field="ignored",
        )
        assert config.max_retries == 2
        assert config.catch_exceptions is False
        assert config.partial_failure_mode == "raise"


# ============================================================================
# TASK 5-7: check() API catch_exceptions / max_retries
# ============================================================================

class TestCheckAPIPhase5:
    """Tests for check() API PHASE 5 parameters."""

    def test_check_with_catch_exceptions_true(self, sample_df):
        from truthound.api import check

        report = check(sample_df, catch_exceptions=True)
        assert report is not None

    def test_check_with_max_retries(self, sample_df):
        from truthound.api import check

        report = check(sample_df, max_retries=2)
        assert report is not None

    def test_check_injects_catch_exceptions_into_configs(self, sample_df):
        from truthound.api import check

        report = check(
            sample_df,
            validators=["null"],
            catch_exceptions=False,
            max_retries=1,
        )
        assert report is not None

    def test_check_catch_exceptions_false_propagates(self, sample_lf):
        """When catch_exceptions=False, permanent errors should propagate."""
        from truthound.api import check
        from truthound.validators.base import Validator, ValidatorConfig

        class BombValidator(Validator):
            name = "bomb"

            def validate(self, lf):
                raise RuntimeError("intentional explosion")

        # When catch_exceptions=False, should raise
        with pytest.raises(RuntimeError, match="intentional explosion"):
            check(
                sample_lf.collect(),
                validators=[BombValidator()],
                catch_exceptions=False,
            )


# ============================================================================
# TASK 5-8: Resilience Bridge
# ============================================================================

class TestResilienceBridge:
    """Tests for ValidationResiliencePolicy."""

    def test_create_default_policy(self):
        from truthound.validators.resilience_bridge import create_default_policy

        policy = create_default_policy(max_retries=2)
        assert policy.max_retries == 2

    def test_create_strict_policy(self):
        from truthound.validators.resilience_bridge import create_strict_policy

        policy = create_strict_policy()
        assert policy.max_retries == 0

    def test_execute_success(self, sample_lf):
        from truthound.validators.resilience_bridge import create_default_policy
        from truthound.validators.base import ValidationResult
        from truthound.validators.completeness.null import NullValidator

        policy = create_default_policy()
        v = NullValidator()
        result = policy.execute(v, sample_lf)
        assert result.status == ValidationResult.SUCCESS

    def test_execute_with_retry(self, sample_lf):
        from truthound.validators.resilience_bridge import ValidationResiliencePolicy
        from truthound.common.resilience.config import CircuitBreakerConfig
        from truthound.validators.base import ValidationResult, Validator

        class FlakyValidator(Validator):
            name = "flaky"
            _call_count = 0

            def validate(self, lf):
                self._call_count += 1
                if self._call_count < 2:
                    raise ConnectionError("transient")
                return []

        # Use a truly permissive circuit breaker for test
        policy = ValidationResiliencePolicy(
            circuit_breaker_config=CircuitBreakerConfig(
                failure_threshold=1_000_000,
                failure_rate_threshold=100.0,  # Never trip on rate
                timeout_seconds=0.1,
            ),
            max_retries=3,
        )
        v = FlakyValidator()
        result = policy.execute(v, sample_lf)
        assert result.status == ValidationResult.SUCCESS
        assert result.retry_count == 1

    def test_circuit_state(self):
        from truthound.validators.resilience_bridge import create_default_policy

        policy = create_default_policy()
        state = policy.get_circuit_state("nonexistent")
        assert state == "unknown"

    def test_reset(self, sample_lf):
        from truthound.validators.resilience_bridge import create_default_policy
        from truthound.validators.completeness.null import NullValidator

        policy = create_default_policy()
        v = NullValidator()
        policy.execute(v, sample_lf)
        policy.reset()  # Should not raise


# ============================================================================
# TASK 5-9: Report ExceptionSummary
# ============================================================================

class TestReportExceptionSummary:
    """Tests for ExceptionSummary in Report."""

    def test_no_exceptions(self):
        from truthound.report import Report, ExceptionSummary
        from truthound.validators.base import ValidationIssue

        issues = [
            ValidationIssue(column="a", issue_type="null", count=5, severity=Severity.HIGH),
        ]
        report = Report(issues=issues)
        assert report.exception_summary is None

    def test_with_exceptions(self):
        from truthound.report import Report, ExceptionSummary
        from truthound.validators.base import ValidationIssue, ExceptionInfo

        exc_info = ExceptionInfo.from_exception(
            ValueError("bad"), validator_name="v1",
        )
        issues = [
            ValidationIssue(
                column="a", issue_type="error", count=0,
                severity=Severity.LOW, exception_info=exc_info,
            ),
        ]
        report = Report(issues=issues)
        assert report.exception_summary is not None
        assert report.exception_summary.total_exceptions == 1
        assert report.exception_summary.exceptions_by_type == {"ValueError": 1}
        assert report.exception_summary.exceptions_by_validator == {"v1": 1}

    def test_exception_summary_to_dict(self):
        from truthound.report import ExceptionSummary
        from truthound.validators.base import ValidationIssue, ExceptionInfo

        exc_info = ExceptionInfo.from_exception(
            ConnectionError("net"), validator_name="v1",
        )
        exc_info.retry_count = 2
        issues = [
            ValidationIssue(
                column="a", issue_type="error", count=0,
                severity=Severity.LOW, exception_info=exc_info,
            ),
        ]
        summary = ExceptionSummary.from_issues(issues)
        d = summary.to_dict()
        assert d["total_exceptions"] == 1
        assert d["total_retries"] == 2
        assert d["retryable_count"] == 1

    def test_report_to_dict_includes_exception_summary(self):
        from truthound.report import Report
        from truthound.validators.base import ValidationIssue, ExceptionInfo

        exc_info = ExceptionInfo.from_exception(ValueError("x"))
        issues = [
            ValidationIssue(
                column="a", issue_type="e", count=0,
                severity=Severity.LOW, exception_info=exc_info,
            ),
        ]
        report = Report(issues=issues)
        d = report.to_dict()
        assert "exception_summary" in d

    def test_report_to_json_with_exception_summary(self):
        from truthound.report import Report
        from truthound.validators.base import ValidationIssue, ExceptionInfo

        exc_info = ExceptionInfo.from_exception(ValueError("x"))
        issues = [
            ValidationIssue(
                column="a", issue_type="e", count=0,
                severity=Severity.LOW, exception_info=exc_info,
            ),
        ]
        report = Report(issues=issues)
        j = json.loads(report.to_json())
        assert "exception_summary" in j

    def test_report_print_with_exceptions(self):
        from truthound.report import Report
        from truthound.validators.base import ValidationIssue, ExceptionInfo

        exc_info = ExceptionInfo.from_exception(ValueError("x"), validator_name="v1")
        issues = [
            ValidationIssue(
                column="a", issue_type="e", count=0,
                severity=Severity.LOW, exception_info=exc_info,
            ),
        ]
        report = Report(issues=issues)
        output = str(report)
        assert "Exceptions:" in output or "exception" in output.lower()

    def test_exception_summary_from_issues_empty(self):
        from truthound.report import ExceptionSummary

        summary = ExceptionSummary.from_issues([])
        assert summary.has_exceptions is False
        assert summary.total_exceptions == 0


# ============================================================================
# TASK 5-4: Expression-level partial failure (3-tier fallback)
# ============================================================================

class TestExpressionPartialFailure:
    """Tests for 3-tier fallback in ExpressionBatchExecutor."""

    def test_batch_success_no_fallback(self, sample_lf):
        """Normal case — batch collect succeeds."""
        from truthound.validators.base import ExpressionBatchExecutor
        from truthound.validators.completeness.null import NullValidator

        executor = ExpressionBatchExecutor()
        executor.add_validator(NullValidator())
        issues = executor.execute(sample_lf)
        # Should find null issues in name and age columns
        assert isinstance(issues, list)

    def test_traditional_validator_error_isolation(self, sample_lf):
        """Traditional validator failure is caught."""
        from truthound.validators.base import (
            ExpressionBatchExecutor, Validator, ValidatorConfig,
        )

        class FailingValidator(Validator):
            name = "failing"

            def validate(self, lf):
                raise RuntimeError("boom")

        executor = ExpressionBatchExecutor()
        executor.add_validator(FailingValidator())
        issues = executor.execute(sample_lf)
        # Should have an error issue instead of raising
        assert len(issues) == 1
        assert issues[0].issue_type == "validator_error"
        assert issues[0].exception_info is not None

    def test_traditional_validator_catch_false_raises(self, sample_lf):
        """Traditional validator failure propagates when catch_exceptions=False."""
        from truthound.validators.base import (
            ExpressionBatchExecutor, Validator, ValidatorConfig,
        )

        class StrictFailValidator(Validator):
            name = "strict_fail"

            def validate(self, lf):
                raise RuntimeError("strict boom")

        v = StrictFailValidator(config=ValidatorConfig(catch_exceptions=False))
        executor = ExpressionBatchExecutor()
        executor.add_validator(v)
        with pytest.raises(RuntimeError, match="strict boom"):
            executor.execute(sample_lf)

    def test_multiple_validators_mixed_success_fail(self, sample_lf):
        """One validator fails, others succeed."""
        from truthound.validators.base import (
            ExpressionBatchExecutor, Validator,
        )
        from truthound.validators.completeness.null import NullValidator

        class FailingTraditional(Validator):
            name = "fail_traditional"

            def validate(self, lf):
                raise RuntimeError("fail")

        executor = ExpressionBatchExecutor()
        executor.add_validator(NullValidator())
        executor.add_validator(FailingTraditional())
        issues = executor.execute(sample_lf)
        # Should have both null issues and error issue
        error_issues = [i for i in issues if i.issue_type == "validator_error"]
        null_issues = [i for i in issues if i.issue_type != "validator_error"]
        assert len(error_issues) == 1
        assert len(null_issues) > 0


# ============================================================================
# Integration test: end-to-end with check()
# ============================================================================

class TestIntegrationPhase5:
    """End-to-end integration tests."""

    def test_check_default_params(self, sample_df):
        from truthound.api import check

        report = check(sample_df)
        assert report is not None
        assert isinstance(report.to_dict(), dict)

    def test_check_with_retries_and_catch(self, sample_df):
        from truthound.api import check

        report = check(
            sample_df,
            validators=["null"],
            catch_exceptions=True,
            max_retries=1,
        )
        assert report is not None
        assert report.success is not None

    def test_report_json_serializable(self, sample_df):
        from truthound.api import check

        run_result = check(sample_df, validators=["null"])
        j = run_result.to_json()
        parsed = json.loads(j)
        assert "issues" in parsed
        assert "checks" in parsed
        assert "execution_mode" in parsed

    def test_check_result_format_propagation(self, sample_df):
        from truthound.api import check

        report = check(
            sample_df,
            validators=["null"],
            result_format="boolean_only",
            catch_exceptions=True,
        )
        assert report.result_format == ResultFormat.BOOLEAN_ONLY
