"""Integration module for connecting profiler output to validation.

This module bridges the gap between profile-generated validation suites
and the actual validation execution, enabling:
- Direct execution of generated suites
- Conversion of rules to validators
- Execution strategy customization

Example:
    from truthound.profiler import generate_suite
    from truthound.profiler.integration import SuiteExecutor

    # Generate suite from profile
    suite = generate_suite(profile)

    # Execute directly
    result = suite.execute(data)

    # Or use custom executor
    executor = SuiteExecutor(parallel=True, fail_fast=False)
    result = executor.execute(suite, data)
"""

from truthound.profiler.integration.protocols import (
    ValidationExecutor,
    ValidatorFactory,
    ExecutionContext,
    ExecutionResult,
)
from truthound.profiler.integration.executor import (
    SuiteExecutor,
    AsyncSuiteExecutor,
    DryRunExecutor,
    ParallelExecutor,
    create_executor,
)
from truthound.profiler.integration.adapters import (
    RuleToValidatorAdapter,
    ValidatorRegistry,
    create_validator_from_rule,
    register_rule_adapter,
)
from truthound.profiler.integration.context import (
    ExecutionConfig,
    ExecutionContextBuilder,
    create_context,
)
from truthound.profiler.integration.naming import resolve_validator_name

__all__ = [
    # Protocols
    "ValidationExecutor",
    "ValidatorFactory",
    "ExecutionContext",
    "ExecutionResult",
    # Executors
    "SuiteExecutor",
    "AsyncSuiteExecutor",
    "DryRunExecutor",
    "ParallelExecutor",
    "create_executor",
    # Adapters
    "RuleToValidatorAdapter",
    "ValidatorRegistry",
    "create_validator_from_rule",
    "register_rule_adapter",
    # Context
    "ExecutionConfig",
    "ExecutionContextBuilder",
    "create_context",
    # Naming
    "resolve_validator_name",
]
