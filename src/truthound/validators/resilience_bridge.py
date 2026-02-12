"""Bridge between the common resilience module and the validation system.

Provides ``ValidationResiliencePolicy`` — a composable wrapper that adds
circuit-breaker and advanced retry behaviour to validator execution.

Example:
    from truthound.validators.resilience_bridge import (
        ValidationResiliencePolicy,
        create_default_policy,
    )

    policy = create_default_policy()
    result = policy.execute(validator, lf)

The bridge is intentionally *optional*: if callers do not need circuit-
breaker semantics, they can use ``_validate_safe()`` directly — which
already has built-in exponential-backoff retry.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

import polars as pl

from truthound.common.resilience.config import CircuitBreakerConfig, RetryConfig
from truthound.common.resilience.circuit_breaker import (
    CircuitBreaker,
    CircuitOpenError,
    CircuitState,
)
from truthound.common.resilience.retry import (
    RetryPolicy,
    RetryExhaustedError,
    ExponentialBackoff,
    JitteredBackoff,
)
from truthound.validators.base import (
    ExceptionInfo,
    ValidationResult,
    ValidatorExecutionResult,
    _get_logger,
)

if TYPE_CHECKING:
    from truthound.validators.base import Validator

logger = logging.getLogger("truthound.resilience_bridge")


# ============================================================================
# Validation-specific retry config factory
# ============================================================================

def _validation_retry_config(max_retries: int) -> RetryConfig:
    """Build a ``RetryConfig`` tuned for validation workloads."""
    from truthound.validators.base import (
        ValidationTimeoutError,
        ColumnNotFoundError,
        RegexValidationError,
    )
    return RetryConfig(
        max_attempts=max_retries + 1,
        base_delay=0.1,
        max_delay=5.0,
        exponential_base=2.0,
        jitter=True,
        jitter_factor=0.3,
        retryable_exceptions=(
            ConnectionError,
            TimeoutError,
            OSError,
            ValidationTimeoutError,
        ),
        non_retryable_exceptions=(
            ValueError,
            TypeError,
            KeyError,
            ColumnNotFoundError,
            RegexValidationError,
        ),
    )


# ============================================================================
# Validation Resilience Policy
# ============================================================================

@dataclass
class ValidationResiliencePolicy:
    """Composable resilience policy for validator execution.

    Wraps a single validation call with:
    - **Circuit breaker** — prevents hammering a failing validator.
    - **Retry policy** — retries transient errors with backoff.

    The circuit breaker is per-validator-name, so a single flaky
    validator won't block unrelated ones.

    Attributes:
        circuit_breaker_config: Config for circuit breaker behaviour.
        max_retries: Number of retry attempts.
        on_retry: Optional callback ``(attempt, error, delay)`` invoked
                  before each retry sleep.
    """

    circuit_breaker_config: CircuitBreakerConfig = field(
        default_factory=lambda: CircuitBreakerConfig.lenient(),
    )
    max_retries: int = 0
    on_retry: Callable[[int, Exception, float], None] | None = None

    # Internal: per-validator circuit breakers
    _breakers: dict[str, CircuitBreaker] = field(
        default_factory=dict, repr=False, compare=False,
    )

    def _get_breaker(self, name: str) -> CircuitBreaker:
        """Get or create a circuit breaker for a validator name."""
        if name not in self._breakers:
            self._breakers[name] = CircuitBreaker(
                name=f"validator-{name}",
                config=self.circuit_breaker_config,
            )
        return self._breakers[name]

    def execute(
        self,
        validator: "Validator",
        lf: pl.LazyFrame,
        skip_on_error: bool = True,
        log_errors: bool = True,
    ) -> ValidatorExecutionResult:
        """Execute a validator with circuit-breaker + retry protection.

        Args:
            validator: Validator to execute.
            lf: LazyFrame to validate.
            skip_on_error: If True, catch errors; if False, propagate.
            log_errors: Log errors if True.

        Returns:
            ValidatorExecutionResult — always returns a result when
            ``skip_on_error`` is True.
        """
        start_time = time.time()
        breaker = self._get_breaker(validator.name)

        # Check circuit state first
        if breaker.state == CircuitState.OPEN:
            exc_info = ExceptionInfo(
                raised_exception=True,
                exception_type="CircuitOpenError",
                exception_message=f"Circuit breaker open for validator '{validator.name}'",
                validator_name=validator.name,
                failure_category="transient",
                is_retryable=False,
            )
            if log_errors:
                logger.warning(
                    "Circuit open for %s — skipping execution", validator.name,
                )
            return ValidatorExecutionResult(
                validator_name=validator.name,
                status=ValidationResult.SKIPPED,
                issues=[],
                error_message=exc_info.exception_message,
                error_context=exc_info.to_error_context(),
                execution_time_ms=(time.time() - start_time) * 1000,
                exception_info=exc_info,
            )

        # Build retry config from max_retries
        retry_cfg = _validation_retry_config(self.max_retries)
        retry_policy = RetryPolicy(
            config=retry_cfg,
            backoff=JitteredBackoff(
                base=ExponentialBackoff(
                    base_delay=retry_cfg.base_delay,
                    multiplier=retry_cfg.exponential_base,
                    max_delay=retry_cfg.max_delay,
                ),
                jitter_factor=retry_cfg.jitter_factor,
            ),
            on_retry=self.on_retry,
        )

        last_exc: Exception | None = None
        retry_count = 0

        # Check circuit before starting retry loop
        if not breaker.allow_request():
            exc_info = ExceptionInfo(
                raised_exception=True,
                exception_type="CircuitOpenError",
                exception_message=f"Circuit open for validator '{validator.name}'",
                validator_name=validator.name,
                failure_category="transient",
                is_retryable=False,
            )
            if skip_on_error:
                return ValidatorExecutionResult(
                    validator_name=validator.name,
                    status=ValidationResult.SKIPPED,
                    issues=[],
                    error_message=exc_info.exception_message,
                    error_context=exc_info.to_error_context(),
                    execution_time_ms=(time.time() - start_time) * 1000,
                    exception_info=exc_info,
                )
            raise CircuitOpenError(breaker.name, breaker.state, 0)

        for attempt in range(retry_cfg.max_attempts):
            try:
                issues = validator.validate(lf)

                # Record success with circuit breaker
                breaker.record_success()

                return ValidatorExecutionResult(
                    validator_name=validator.name,
                    status=ValidationResult.SUCCESS,
                    issues=issues,
                    execution_time_ms=(time.time() - start_time) * 1000,
                    retry_count=retry_count,
                )

            except Exception as e:
                last_exc = e
                retry_count = attempt + 1

                if retry_policy.should_retry(attempt, e) and attempt < retry_cfg.max_attempts - 1:
                    delay = retry_policy.get_delay(attempt)
                    if log_errors:
                        logger.warning(
                            "Validator '%s' failed (attempt %d/%d), retrying in %.2fs: %s",
                            validator.name, attempt + 1, retry_cfg.max_attempts, delay, e,
                        )
                    time.sleep(delay)
                else:
                    break

        # All retries exhausted — record failure with circuit breaker
        if last_exc is not None:
            breaker.record_failure(last_exc)

        # All attempts exhausted
        exc_info = ExceptionInfo.from_exception(
            last_exc, validator_name=validator.name,  # type: ignore[arg-type]
        )
        exc_info.retry_count = retry_count
        exc_info.max_retries = self.max_retries

        if log_errors:
            logger.error(
                "Validator '%s' failed after %d attempts: %s",
                validator.name, retry_count, last_exc,
            )

        if skip_on_error:
            from truthound.validators.base import ValidationTimeoutError
            status = (
                ValidationResult.TIMEOUT
                if isinstance(last_exc, ValidationTimeoutError)
                else ValidationResult.FAILED
            )
            return ValidatorExecutionResult(
                validator_name=validator.name,
                status=status,
                issues=[],
                error_message=f"Failed after {retry_count} attempts: {last_exc}",
                error_context=exc_info.to_error_context(),
                execution_time_ms=(time.time() - start_time) * 1000,
                exception_info=exc_info,
                retry_count=retry_count,
            )
        raise last_exc  # type: ignore[misc]

    def get_circuit_state(self, validator_name: str) -> str:
        """Return the circuit breaker state for a validator."""
        breaker = self._breakers.get(validator_name)
        if breaker is None:
            return "unknown"
        return breaker.state.value

    def reset(self, validator_name: str | None = None) -> None:
        """Reset circuit breakers.

        Args:
            validator_name: Reset a specific breaker, or all if None.
        """
        if validator_name is not None:
            breaker = self._breakers.get(validator_name)
            if breaker:
                breaker.reset()
        else:
            for breaker in self._breakers.values():
                breaker.reset()


# ============================================================================
# Factory helpers
# ============================================================================

def create_default_policy(max_retries: int = 0) -> ValidationResiliencePolicy:
    """Create a sensible default resilience policy.

    Uses lenient circuit breaker (10 failures to open, 15s timeout)
    and exponential backoff retry.
    """
    return ValidationResiliencePolicy(
        circuit_breaker_config=CircuitBreakerConfig.lenient(),
        max_retries=max_retries,
    )


def create_strict_policy() -> ValidationResiliencePolicy:
    """Create a strict policy: no retries, aggressive circuit breaker."""
    return ValidationResiliencePolicy(
        circuit_breaker_config=CircuitBreakerConfig.aggressive(),
        max_retries=0,
    )
