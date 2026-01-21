"""Validation suite generator.

This module provides the main ValidationSuiteGenerator that combines
all rule generators to create a complete validation suite from a profile.

Key Features:
- Generates validation rules from profile results
- Supports multiple profile types (TableProfile, ProfileReport, dict)
- Configurable strictness levels
- Filtering by category and confidence
- Export to YAML, JSON, and Python code

Example:
    from truthound.profiler import profile_file, generate_suite

    # Using TableProfile (recommended)
    profile = profile_file("data.parquet")
    suite = generate_suite(profile)

    # Using ProfileReport from th.profile()
    import truthound as th
    profile_report = th.profile("data.csv")
    suite = generate_suite(profile_report)

    # Using dict
    profile_dict = profile.to_dict()
    suite = generate_suite(profile_dict)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Sequence, TYPE_CHECKING, Union

from truthound.profiler.base import (
    ColumnProfile,
    DataType,
    ProfilerConfig,
    Strictness,
    TableProfile,
)
from truthound.profiler.generators.base import (
    GeneratedRule,
    RuleCategory,
    RuleConfidence,
    RuleGenerator,
    rule_generator_registry,
)

if TYPE_CHECKING:
    from truthound.validators.base import Validator
    from truthound.profiler.integration.protocols import ExecutionContext, ExecutionResult
    from truthound.report import ProfileReport

logger = logging.getLogger(__name__)

# Type alias for supported profile types
ProfileInput = Union["TableProfile", "ProfileReport", dict[str, Any]]


@dataclass(frozen=True)
class ValidationSuite:
    """A complete validation suite generated from a profile.

    This is an immutable collection of generated rules that can be
    exported, filtered, and converted to actual validators.
    """

    name: str
    rules: tuple[GeneratedRule, ...] = field(default_factory=tuple)
    source_profile: str = ""  # Reference to source profile
    strictness: Strictness = Strictness.MEDIUM
    metadata: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.rules)

    def __iter__(self):
        return iter(self.rules)

    def filter_by_category(
        self,
        *categories: RuleCategory,
    ) -> "ValidationSuite":
        """Filter rules by category."""
        filtered = tuple(r for r in self.rules if r.category in categories)
        return ValidationSuite(
            name=self.name,
            rules=filtered,
            source_profile=self.source_profile,
            strictness=self.strictness,
            metadata=self.metadata,
        )

    def filter_by_confidence(
        self,
        min_confidence: RuleConfidence,
    ) -> "ValidationSuite":
        """Filter rules by minimum confidence level."""
        confidence_order = {
            RuleConfidence.LOW: 0,
            RuleConfidence.MEDIUM: 1,
            RuleConfidence.HIGH: 2,
        }
        min_level = confidence_order[min_confidence]

        filtered = tuple(
            r for r in self.rules
            if confidence_order[r.confidence] >= min_level
        )
        return ValidationSuite(
            name=self.name,
            rules=filtered,
            source_profile=self.source_profile,
            strictness=self.strictness,
            metadata=self.metadata,
        )

    def filter_by_columns(self, *columns: str) -> "ValidationSuite":
        """Filter rules that apply to specific columns."""
        column_set = set(columns)
        filtered = tuple(
            r for r in self.rules
            if not r.columns or any(c in column_set for c in r.columns)
        )
        return ValidationSuite(
            name=self.name,
            rules=filtered,
            source_profile=self.source_profile,
            strictness=self.strictness,
            metadata=self.metadata,
        )

    def exclude_categories(
        self,
        *categories: RuleCategory,
    ) -> "ValidationSuite":
        """Exclude rules in specific categories."""
        excluded = set(categories)
        filtered = tuple(r for r in self.rules if r.category not in excluded)
        return ValidationSuite(
            name=self.name,
            rules=filtered,
            source_profile=self.source_profile,
            strictness=self.strictness,
            metadata=self.metadata,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "rules": [r.to_dict() for r in self.rules],
            "source_profile": self.source_profile,
            "strictness": self.strictness.value,
            "metadata": self.metadata,
            "summary": {
                "total_rules": len(self.rules),
                "by_category": self._count_by_category(),
                "by_confidence": self._count_by_confidence(),
            },
        }

    def _count_by_category(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for rule in self.rules:
            cat = rule.category.value
            counts[cat] = counts.get(cat, 0) + 1
        return counts

    def _count_by_confidence(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for rule in self.rules:
            conf = rule.confidence.value
            counts[conf] = counts.get(conf, 0) + 1
        return counts

    def to_yaml(self) -> str:
        """Convert to YAML format for human-readable output."""
        lines = [
            f"# Validation Suite: {self.name}",
            f"# Strictness: {self.strictness.value}",
            f"# Total rules: {len(self.rules)}",
            "",
            "rules:",
        ]

        for rule in self.rules:
            lines.append(f"  - name: {rule.name}")
            lines.append(f"    validator: {rule.validator_class}")
            lines.append(f"    category: {rule.category.value}")
            lines.append(f"    confidence: {rule.confidence.value}")

            if rule.columns:
                lines.append(f"    columns: {list(rule.columns)}")

            if rule.parameters:
                lines.append("    parameters:")
                for k, v in rule.parameters.items():
                    lines.append(f"      {k}: {v}")

            if rule.mostly is not None:
                lines.append(f"    mostly: {rule.mostly}")

            if rule.description:
                lines.append(f"    description: \"{rule.description}\"")

            lines.append("")

        return "\n".join(lines)

    def execute(
        self,
        data: Any,
        *,
        parallel: bool = False,
        fail_fast: bool = False,
        max_workers: int | None = None,
        timeout_seconds: float | None = None,
        context: "ExecutionContext | None" = None,
    ) -> "ExecutionResult":
        """Execute the validation suite against data.

        This method provides a convenient way to run all validators
        in the suite against the provided data.

        Args:
            data: Data to validate (LazyFrame, DataFrame, or file path).
            parallel: Whether to run validators in parallel.
            fail_fast: Whether to stop on first failure.
            max_workers: Maximum number of parallel workers.
            timeout_seconds: Maximum execution time per validator.
            context: Pre-configured execution context (overrides other params).

        Returns:
            ExecutionResult with validation report and metrics.

        Example:
            suite = generate_suite(profile)
            result = suite.execute(data, parallel=True)

            if result.success:
                print(f"All {result.passed_rules} rules passed!")
            else:
                print(f"Failed: {result.failed_rules} rules")
        """
        from truthound.profiler.integration.executor import SuiteExecutor
        from truthound.profiler.integration.protocols import ExecutionContext as ExecCtx

        # Create context if not provided
        if context is None:
            context = ExecCtx(
                parallel=parallel,
                fail_fast=fail_fast,
                max_workers=max_workers,
                timeout_seconds=timeout_seconds,
            )

        # Create executor and run
        executor = SuiteExecutor(
            parallel=context.parallel,
            fail_fast=context.fail_fast,
            max_workers=context.max_workers,
            timeout_seconds=context.timeout_seconds,
        )

        return executor.execute(self, data, context)

    async def execute_async(
        self,
        data: Any,
        *,
        parallel: bool = True,
        fail_fast: bool = False,
        context: "ExecutionContext | None" = None,
    ) -> "ExecutionResult":
        """Execute the validation suite asynchronously.

        Args:
            data: Data to validate.
            parallel: Whether to run validators in parallel.
            fail_fast: Whether to stop on first failure.
            context: Pre-configured execution context.

        Returns:
            ExecutionResult with validation report and metrics.
        """
        from truthound.profiler.integration.executor import AsyncSuiteExecutor
        from truthound.profiler.integration.protocols import ExecutionContext as ExecCtx

        if context is None:
            context = ExecCtx(parallel=parallel, fail_fast=fail_fast)

        executor = AsyncSuiteExecutor(
            parallel=context.parallel,
            fail_fast=context.fail_fast,
        )

        return await executor.execute_async(self, data, context)

    def to_python_code(self) -> str:
        """Generate Python code to create validators."""
        lines = [
            '"""Auto-generated validation suite."""',
            "",
            "from truthound.validators import (",
        ]

        # Collect unique validator classes
        validators = sorted(set(r.validator_class for r in self.rules))
        for v in validators:
            lines.append(f"    {v},")
        lines.append(")")
        lines.append("")
        lines.append("")
        lines.append("def create_validators():")
        lines.append('    """Create validation rules."""')
        lines.append("    validators = []")
        lines.append("")

        for rule in self.rules:
            lines.append(f"    # {rule.name}")
            if rule.description:
                lines.append(f"    # {rule.description}")

            # Build parameters
            params = []
            if rule.columns:
                params.append(f"columns={list(rule.columns)}")
            for k, v in rule.parameters.items():
                if isinstance(v, str):
                    params.append(f'{k}="{v}"')
                else:
                    params.append(f"{k}={v!r}")
            if rule.mostly is not None:
                params.append(f"mostly={rule.mostly}")

            param_str = ", ".join(params)
            lines.append(f"    validators.append({rule.validator_class}({param_str}))")
            lines.append("")

        lines.append("    return validators")
        lines.append("")

        return "\n".join(lines)


class ValidationSuiteGenerator:
    """Generates validation suites by combining multiple rule generators.

    This is the main entry point for automatic rule generation. It
    orchestrates multiple generators and combines their output into
    a cohesive validation suite.

    Example:
        generator = ValidationSuiteGenerator()
        suite = generator.generate_from_profile(
            profile,
            strictness=Strictness.MEDIUM,
            include_categories=["schema", "completeness", "format"]
        )

        # Export
        suite.to_yaml()
        suite.to_python_code()
    """

    def __init__(
        self,
        generators: Sequence[RuleGenerator] | None = None,
        **kwargs: Any,
    ):
        """Initialize suite generator.

        Args:
            generators: Custom list of generators to use.
                       If None, uses all registered generators.
            **kwargs: Additional arguments passed to generators.
        """
        if generators is not None:
            self.generators = list(generators)
        else:
            # Use all registered generators
            self.generators = rule_generator_registry.create_all(**kwargs)

    def add_generator(self, generator: RuleGenerator) -> None:
        """Add a custom generator."""
        self.generators.append(generator)
        # Re-sort by priority
        self.generators.sort(key=lambda g: -g.priority)

    def generate_from_profile(
        self,
        profile: TableProfile,
        *,
        strictness: Strictness = Strictness.MEDIUM,
        include_categories: Sequence[str] | None = None,
        exclude_categories: Sequence[str] | None = None,
        min_confidence: RuleConfidence | None = None,
        name: str | None = None,
    ) -> ValidationSuite:
        """Generate a validation suite from a profile.

        Args:
            profile: Table profile to generate rules from
            strictness: How strict the generated rules should be
            include_categories: Only include rules from these categories
            exclude_categories: Exclude rules from these categories
            min_confidence: Only include rules with at least this confidence
            name: Name for the suite (defaults to profile name)

        Returns:
            Generated validation suite
        """
        all_rules: list[GeneratedRule] = []

        # Convert category strings to enums
        include_cats = None
        if include_categories:
            include_cats = {RuleCategory(c) for c in include_categories}

        exclude_cats = set()
        if exclude_categories:
            exclude_cats = {RuleCategory(c) for c in exclude_categories}

        # Run each generator
        for generator in self.generators:
            # Skip if generator doesn't produce any included categories
            if include_cats:
                if not generator.categories & include_cats:
                    continue

            # Skip if all generator categories are excluded
            if exclude_cats:
                if generator.categories <= exclude_cats:
                    continue

            try:
                rules = generator.generate(profile, strictness)

                # Filter by category
                if include_cats:
                    rules = [r for r in rules if r.category in include_cats]
                if exclude_cats:
                    rules = [r for r in rules if r.category not in exclude_cats]

                all_rules.extend(rules)
            except Exception:
                # Skip failed generators
                pass

        # Filter by confidence
        if min_confidence:
            confidence_order = {
                RuleConfidence.LOW: 0,
                RuleConfidence.MEDIUM: 1,
                RuleConfidence.HIGH: 2,
            }
            min_level = confidence_order[min_confidence]
            all_rules = [
                r for r in all_rules
                if confidence_order[r.confidence] >= min_level
            ]

        # Deduplicate rules (same name = same rule)
        seen_names: set[str] = set()
        unique_rules: list[GeneratedRule] = []
        for rule in all_rules:
            if rule.name not in seen_names:
                seen_names.add(rule.name)
                unique_rules.append(rule)

        return ValidationSuite(
            name=name or profile.name or "generated_suite",
            rules=tuple(unique_rules),
            source_profile=profile.name,
            strictness=strictness,
            metadata={
                "profile_row_count": profile.row_count,
                "profile_column_count": profile.column_count,
                "generators_used": [g.name for g in self.generators],
            },
        )


# =============================================================================
# Profile Adapter
# =============================================================================


class ProfileAdapter:
    """Adapter for converting various profile types to TableProfile.

    This enables generate_suite() to work with different profile types:
    - TableProfile: Native profiler output (used directly)
    - ProfileReport: Simplified report from th.profile() API
    - dict: Dictionary representation of a profile

    Example:
        # From ProfileReport (th.profile() output)
        import truthound as th
        profile_report = th.profile("data.csv")
        table_profile = ProfileAdapter.to_table_profile(profile_report)

        # From dict
        profile_dict = {"row_count": 100, "columns": [...]}
        table_profile = ProfileAdapter.to_table_profile(profile_dict)
    """

    @staticmethod
    def to_table_profile(profile: ProfileInput) -> TableProfile:
        """Convert any profile type to TableProfile.

        Args:
            profile: Profile in any supported format.

        Returns:
            TableProfile instance.

        Raises:
            TypeError: If the profile type is not supported.
        """
        # Already a TableProfile
        if isinstance(profile, TableProfile):
            return profile

        # Check for ProfileReport type (from truthound.report)
        if hasattr(profile, 'source') and hasattr(profile, 'columns') and hasattr(profile, 'row_count'):
            # Duck typing for ProfileReport
            return ProfileAdapter._from_profile_report(profile)

        # Dictionary format
        if isinstance(profile, dict):
            return ProfileAdapter._from_dict(profile)

        raise TypeError(
            f"Unsupported profile type: {type(profile).__name__}. "
            "Expected TableProfile, ProfileReport, or dict."
        )

    @staticmethod
    def _from_profile_report(report: Any) -> TableProfile:
        """Convert ProfileReport to TableProfile.

        Args:
            report: ProfileReport instance.

        Returns:
            TableProfile with extracted information.
        """
        # Extract column profiles from ProfileReport
        column_profiles: list[ColumnProfile] = []

        for col_dict in getattr(report, 'columns', []):
            # Parse null percentage
            null_pct_str = col_dict.get('null_pct', '0%')
            null_ratio = ProfileAdapter._parse_percentage(null_pct_str)

            # Parse unique percentage
            unique_pct_str = col_dict.get('unique_pct', '0%')
            unique_ratio = ProfileAdapter._parse_percentage(unique_pct_str)

            # Infer data type from dtype string
            dtype_str = col_dict.get('dtype', 'unknown')
            inferred_type = ProfileAdapter._infer_data_type(dtype_str)

            row_count = getattr(report, 'row_count', 0)
            null_count = int(null_ratio * row_count) if row_count > 0 else 0
            distinct_count = int(unique_ratio * row_count) if row_count > 0 else 0

            col_profile = ColumnProfile(
                name=col_dict.get('name', ''),
                physical_type=dtype_str,
                inferred_type=inferred_type,
                row_count=row_count,
                null_count=null_count,
                null_ratio=null_ratio,
                distinct_count=distinct_count,
                unique_ratio=unique_ratio,
                is_unique=unique_ratio >= 0.99,
                is_constant=distinct_count <= 1,
            )
            column_profiles.append(col_profile)

        return TableProfile(
            name=getattr(report, 'source', 'unknown'),
            row_count=getattr(report, 'row_count', 0),
            column_count=getattr(report, 'column_count', len(column_profiles)),
            estimated_memory_bytes=getattr(report, 'size_bytes', 0),
            columns=tuple(column_profiles),
            source=getattr(report, 'source', 'unknown'),
            profiled_at=datetime.now(),
        )

    @staticmethod
    def _from_dict(data: dict[str, Any]) -> TableProfile:
        """Convert dict to TableProfile.

        Args:
            data: Dictionary with profile data.

        Returns:
            TableProfile instance.
        """
        # Extract column profiles
        column_profiles: list[ColumnProfile] = []

        for col_dict in data.get('columns', []):
            # Handle both TableProfile and ProfileReport dict formats
            if 'inferred_type' in col_dict:
                # TableProfile format
                inferred_type = DataType(col_dict.get('inferred_type', 'unknown'))
            else:
                # ProfileReport format
                dtype_str = col_dict.get('dtype', col_dict.get('physical_type', 'unknown'))
                inferred_type = ProfileAdapter._infer_data_type(dtype_str)

            null_ratio = col_dict.get('null_ratio', 0.0)
            if isinstance(null_ratio, str):
                null_ratio = ProfileAdapter._parse_percentage(null_ratio)

            unique_ratio = col_dict.get('unique_ratio', 0.0)
            if isinstance(unique_ratio, str):
                unique_ratio = ProfileAdapter._parse_percentage(unique_ratio)

            row_count = col_dict.get('row_count', data.get('row_count', 0))

            col_profile = ColumnProfile(
                name=col_dict.get('name', ''),
                physical_type=col_dict.get('physical_type', col_dict.get('dtype', 'unknown')),
                inferred_type=inferred_type,
                row_count=row_count,
                null_count=col_dict.get('null_count', int(null_ratio * row_count)),
                null_ratio=null_ratio,
                distinct_count=col_dict.get('distinct_count', int(unique_ratio * row_count)),
                unique_ratio=unique_ratio,
                is_unique=col_dict.get('is_unique', unique_ratio >= 0.99),
                is_constant=col_dict.get('is_constant', False),
            )
            column_profiles.append(col_profile)

        return TableProfile(
            name=data.get('name', data.get('source', 'unknown')),
            row_count=data.get('row_count', 0),
            column_count=data.get('column_count', len(column_profiles)),
            estimated_memory_bytes=data.get('estimated_memory_bytes', data.get('size_bytes', 0)),
            columns=tuple(column_profiles),
            duplicate_row_count=data.get('duplicate_row_count', 0),
            duplicate_row_ratio=data.get('duplicate_row_ratio', 0.0),
            source=data.get('source', data.get('name', '')),
            profiled_at=datetime.now(),
        )

    @staticmethod
    def _parse_percentage(pct_str: str) -> float:
        """Parse percentage string to float ratio.

        Args:
            pct_str: String like "10.5%" or "10.5"

        Returns:
            Float ratio between 0.0 and 1.0
        """
        if isinstance(pct_str, (int, float)):
            # Already a number, assume it's a ratio if < 1, percentage if >= 1
            return pct_str / 100.0 if pct_str > 1 else pct_str

        pct_str = str(pct_str).strip()
        if pct_str.endswith('%'):
            try:
                return float(pct_str[:-1]) / 100.0
            except ValueError:
                return 0.0
        try:
            value = float(pct_str)
            return value / 100.0 if value > 1 else value
        except ValueError:
            return 0.0

    @staticmethod
    def _infer_data_type(dtype_str: str) -> DataType:
        """Infer DataType from Polars dtype string.

        Args:
            dtype_str: Polars dtype string like "Int64", "String", etc.

        Returns:
            Inferred DataType enum value.
        """
        dtype_lower = dtype_str.lower()

        # Integer types
        if any(t in dtype_lower for t in ['int', 'i8', 'i16', 'i32', 'i64', 'u8', 'u16', 'u32', 'u64']):
            return DataType.INTEGER

        # Float types
        if any(t in dtype_lower for t in ['float', 'f32', 'f64', 'decimal']):
            return DataType.FLOAT

        # Boolean
        if 'bool' in dtype_lower:
            return DataType.BOOLEAN

        # Datetime types
        if 'datetime' in dtype_lower:
            return DataType.DATETIME
        if 'date' in dtype_lower:
            return DataType.DATE
        if 'time' in dtype_lower:
            return DataType.TIME
        if 'duration' in dtype_lower:
            return DataType.DURATION

        # String types
        if any(t in dtype_lower for t in ['str', 'string', 'utf8', 'categorical']):
            return DataType.STRING

        return DataType.UNKNOWN


# =============================================================================
# Convenience Functions
# =============================================================================


def generate_suite(
    profile: ProfileInput,
    *,
    strictness: str | Strictness = "medium",
    include_categories: Sequence[str] | None = None,
    exclude_categories: Sequence[str] | None = None,
    min_confidence: str | RuleConfidence | None = None,
    name: str | None = None,
) -> ValidationSuite:
    """Generate a validation suite from a profile.

    Supports multiple profile types:
    - TableProfile: Native profiler output (from profile_file(), profile_dataframe())
    - ProfileReport: Simplified report from th.profile() API
    - dict: Dictionary representation of a profile

    Args:
        profile: Profile in any supported format (TableProfile, ProfileReport, or dict)
        strictness: "loose", "medium", or "strict"
        include_categories: Only include rules from these categories
        exclude_categories: Exclude rules from these categories
        min_confidence: "low", "medium", or "high"
        name: Name for the suite

    Returns:
        Generated validation suite

    Example:
        # Using profile_file() - recommended for full features
        from truthound.profiler import profile_file, generate_suite
        profile = profile_file("data.parquet")
        suite = generate_suite(profile, strictness="medium")

        # Using th.profile() - simpler API
        import truthound as th
        profile_report = th.profile("data.csv")
        suite = generate_suite(profile_report)

        # View as YAML
        print(suite.to_yaml())

        # Generate Python code
        print(suite.to_python_code())
    """
    # Convert to TableProfile if needed
    table_profile = ProfileAdapter.to_table_profile(profile)

    # Convert strings to enums
    if isinstance(strictness, str):
        strictness = Strictness(strictness)

    if isinstance(min_confidence, str):
        try:
            min_confidence = RuleConfidence(min_confidence)
        except ValueError:
            valid_values = [e.value for e in RuleConfidence]
            raise ValueError(
                f"Invalid min_confidence '{min_confidence}'. "
                f"Valid values: {valid_values}"
            ) from None

    generator = ValidationSuiteGenerator()
    return generator.generate_from_profile(
        table_profile,
        strictness=strictness,
        include_categories=include_categories,
        exclude_categories=exclude_categories,
        min_confidence=min_confidence,
        name=name,
    )


def save_suite(
    suite: ValidationSuite,
    path: str | Path,
    format: str = "json",
) -> None:
    """Save a validation suite to a file.

    Args:
        suite: Suite to save
        path: Output file path
        format: "json", "yaml", or "python"
    """
    path = Path(path)

    if format == "json":
        with open(path, "w", encoding="utf-8") as f:
            json.dump(suite.to_dict(), f, indent=2, ensure_ascii=False)
    elif format == "yaml":
        with open(path, "w", encoding="utf-8") as f:
            f.write(suite.to_yaml())
    elif format == "python":
        with open(path, "w", encoding="utf-8") as f:
            f.write(suite.to_python_code())
    else:
        raise ValueError(f"Unknown format: {format}. Use 'json', 'yaml', or 'python'")


def load_suite(path: str | Path) -> ValidationSuite:
    """Load a validation suite from a JSON file.

    Args:
        path: Path to the suite JSON file

    Returns:
        Loaded validation suite
    """
    path = Path(path)

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rules = tuple(
        GeneratedRule(
            name=r["name"],
            validator_class=r["validator_class"],
            category=RuleCategory(r["category"]),
            parameters=r.get("parameters", {}),
            columns=tuple(r.get("columns", [])),
            confidence=RuleConfidence(r.get("confidence", "medium")),
            description=r.get("description", ""),
            rationale=r.get("rationale", ""),
            mostly=r.get("mostly"),
        )
        for r in data.get("rules", [])
    )

    return ValidationSuite(
        name=data.get("name", ""),
        rules=rules,
        source_profile=data.get("source_profile", ""),
        strictness=Strictness(data.get("strictness", "medium")),
        metadata=data.get("metadata", {}),
    )
