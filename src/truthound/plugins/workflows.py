"""Native manager-local extension workflows over the public validation kernel.

The kernel remains independent of plugins. Providers/factories/reporters keep
their operator-owned configuration and lifecycle within one manager.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from truthound.plugins.hooks import HookExecutionResult, HookType

if TYPE_CHECKING:
    from truthound.core.results import ValidationRunResult


@dataclass(frozen=True)
class PluginValidationResult:
    run: ValidationRunResult
    hooks: dict[str, tuple[HookExecutionResult, ...]] = field(default_factory=dict)

    @property
    def hooks_failed(self) -> bool:
        return any(not row.success for rows in self.hooks.values() for row in rows)


@dataclass(frozen=True)
class PluginRenderResult:
    content: str | bytes
    hooks: dict[str, tuple[HookExecutionResult, ...]] = field(default_factory=dict)

    @property
    def hooks_failed(self) -> bool:
        return any(not row.success for rows in self.hooks.values() for row in rows)


def synchronous(coroutine):
    """Own a fresh loop only when called outside an existing event loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coroutine)
    coroutine.close()
    raise RuntimeError("Use the async plugin workflow inside an event loop")


def validation_asset(manager, data=None, *, asset_provider=None, pushdown=None):
    from truthound.core import build_validation_asset
    from truthound.core.contracts import DataAsset

    guard = getattr(manager, "_execution_guard", None)
    if guard is not None:
        guard()

    if asset_provider is not None:
        if data is not None:
            raise ValueError("Select data or one registered asset provider")
        provider = manager.get_data_asset_provider(asset_provider)
        if provider is None or not callable(getattr(provider, "get_asset", None)):
            raise ValueError("A registered native DataAssetProvider is required")
        asset = provider.get_asset()
        if not isinstance(asset, DataAsset):
            raise TypeError("The provider must return a native DataAsset")
        return asset
    return build_validation_asset(data=data, pushdown=pushdown)


def validation_suite(manager, check_factories, *, name, result_format):
    from truthound.core import CheckSpec, ValidationSuite
    from truthound.core.suite import EvidencePolicy
    from truthound.validators.base import Validator

    specs = []
    for alias in check_factories:
        guard = getattr(manager, "_execution_guard", None)
        if guard is not None:
            guard()
        factory = manager.get_check_factory(alias)
        if factory is None:
            raise ValueError("A selected check factory is not registered")
        if callable(getattr(factory, "build_check_specs", None)):
            built = factory.build_check_specs()
        else:
            validator = (
                factory()
                if isinstance(factory, type) and issubclass(factory, Validator)
                else factory
            )
            if not isinstance(validator, Validator):
                raise TypeError("Use a CheckSpecFactory or native validator registration")
            built = ValidationSuite.from_legacy(
                validators=[validator], result_format=result_format
            ).checks
        if not isinstance(built, (list, tuple)) or any(
            not isinstance(row, CheckSpec) for row in built
        ):
            raise TypeError("A check factory must return native CheckSpec objects")
        specs.extend(built)
    return ValidationSuite(
        name=name, checks=tuple(specs), evidence_policy=EvidencePolicy.from_any(result_format)
    )


async def validate_async(
    manager,
    check_factories,
    data=None,
    *,
    asset_provider=None,
    name="plugin-suite",
    result_format="summary",
    parallel=False,
    max_workers=None,
    pushdown=None,
) -> PluginValidationResult:
    """Execute registered native ports and await lifecycle hooks on this thread.

    The provider/factory is invoked once per selection. Hooks receive immutable
    CheckSpec intent in ``checks`` and ``validators``; the kernel alone builds
    validators. Hook failures are visible and do not veto a validation run.
    No implicit context persistence, workspace creation or provider close occurs.
    """
    from truthound.core import ScanPlanner, ValidationRuntime

    asset = validation_asset(manager, data, asset_provider=asset_provider, pushdown=pushdown)
    suite = validation_suite(manager, check_factories, name=name, result_format=result_format)
    source = getattr(asset, "source", None) or asset
    before = await manager.hooks.trigger_detailed_async(
        HookType.BEFORE_VALIDATION,
        datasource=source,
        asset=asset,
        validators=suite.checks,
        checks=suite.checks,
        suite=suite,
    )
    try:
        guard = getattr(manager, "_execution_guard", None)
        if guard is not None:
            guard()
        plan = ScanPlanner().plan(
            suite=suite, asset=asset, parallel=parallel, max_workers=max_workers, pushdown=pushdown
        )
        run = ValidationRuntime().execute(asset=asset, plan=plan)
    except Exception as error:
        await manager.hooks.trigger_detailed_async(
            HookType.ON_ERROR, error=error, context="validation"
        )
        raise
    after = await manager.hooks.trigger_detailed_async(
        HookType.AFTER_VALIDATION,
        datasource=source,
        asset=asset,
        result=run,
        issues=run.issues,
        checks=suite.checks,
    )
    found = []
    checks = {
        id(issue): spec
        for spec, check in zip(suite.checks, run.checks, strict=True)
        for issue in check.issues
    }
    for issue in run.issues:
        found.extend(
            await manager.hooks.trigger_detailed_async(
                HookType.ON_ISSUE_FOUND,
                issue=issue,
                validator=checks.get(id(issue)),
                result=run,
            )
        )
    return PluginValidationResult(
        run,
        {
            "before_validation": tuple(before),
            "after_validation": tuple(after),
            "on_issue_found": tuple(found),
        },
    )


async def render_async(manager, reporter_name, run) -> PluginRenderResult:
    """Render through a registered v3 native reporter without writing a path."""
    from truthound.core.results import ValidationRunResult

    if not isinstance(run, ValidationRunResult):
        raise TypeError("A canonical ValidationRunResult is required")
    guard = getattr(manager, "_execution_guard", None)
    if guard is not None:
        guard()
    reporter = manager.get_reporter(reporter_name)
    if isinstance(reporter, type) or (
        callable(reporter) and not callable(getattr(reporter, "render", None))
    ):
        reporter = reporter()
    if getattr(reporter, "contract_version", None) != 3 or not callable(
        getattr(reporter, "render", None)
    ):
        raise TypeError("A registered canonical v3 reporter is required")
    content = reporter.render(run)
    if not isinstance(content, (str, bytes)):
        raise TypeError("A reporter must render text or bytes")
    hooks = await manager.hooks.trigger_detailed_async(
        HookType.ON_REPORT_GENERATE,
        report=run,
        format=reporter_name,
        content=content,
    )
    return PluginRenderResult(content, {"on_report_generate": tuple(hooks)})
