from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from truthound.core.contracts import DataAsset
from truthound.core.suite import CheckSpec, ValidationSuite


@dataclass(frozen=True)
class PlanStep:
    check: CheckSpec


@dataclass(frozen=True)
class ScanPlan:
    suite: ValidationSuite
    steps: tuple[PlanStep, ...]
    execution_mode: str = 'sequential'
    max_workers: int | None = None
    pushdown_enabled: bool = False
    duplicate_check_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


class ScanPlanner:
    def plan(
        self,
        *,
        suite: ValidationSuite,
        asset: DataAsset,
        parallel: bool = False,
        max_workers: int | None = None,
        pushdown: bool | None = None,
    ) -> ScanPlan:
        steps = tuple(PlanStep(check=check) for check in suite.checks)
        duplicate_check_count = self._count_duplicates(suite.checks)

        pushdown_enabled = asset.capabilities.pushdown if pushdown is None else (
            pushdown and asset.capabilities.pushdown
        )
        if pushdown_enabled and getattr(asset, 'sql_source', None) is not None:
            execution_mode = 'pushdown'
        elif parallel and len(steps) > 1 and asset.capabilities.parallel:
            execution_mode = 'parallel'
        else:
            execution_mode = 'sequential'

        return ScanPlan(
            suite=suite,
            steps=steps,
            execution_mode=execution_mode,
            max_workers=max_workers,
            pushdown_enabled=bool(pushdown_enabled),
            duplicate_check_count=duplicate_check_count,
            metadata={
                'planner': 'truthound.core.ScanPlanner',
                'asset_backend': asset.backend_name,
            },
        )

    def _count_duplicates(self, checks: tuple[CheckSpec, ...]) -> int:
        signatures = [
            (
                check.name,
                check.category,
                tuple(sorted(check.tags)),
                repr(check.metadata.get('config', {})),
            )
            for check in checks
        ]
        counts = Counter(signatures)
        return sum(count - 1 for count in counts.values() if count > 1)
