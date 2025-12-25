"""GitHub Actions integration.

This action integrates with GitHub Actions to update check runs,
create comments on PRs, and set workflow outputs.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from truthound.checkpoint.actions.base import (
    ActionConfig,
    ActionResult,
    ActionStatus,
    BaseAction,
    NotifyCondition,
)

if TYPE_CHECKING:
    from truthound.checkpoint.checkpoint import CheckpointResult


@dataclass
class GitHubConfig(ActionConfig):
    """Configuration for GitHub Actions integration.

    Attributes:
        token: GitHub token (defaults to GITHUB_TOKEN env var).
        repository: Repository in owner/repo format.
        create_check_run: Create a GitHub check run.
        comment_on_pr: Add comment to pull request.
        pr_number: Pull request number for comments.
        set_output: Set GitHub Actions workflow outputs.
        set_summary: Write to GitHub Actions job summary.
        fail_on_issues: Fail the action if issues are found.
        min_severity_to_fail: Minimum severity to fail ("low", "medium", "high", "critical").
        annotations: Create inline annotations for issues.
    """

    token: str | None = None
    repository: str | None = None
    create_check_run: bool = True
    comment_on_pr: bool = False
    pr_number: int | None = None
    set_output: bool = True
    set_summary: bool = True
    fail_on_issues: bool = False
    min_severity_to_fail: str = "high"
    annotations: bool = True
    notify_on: NotifyCondition | str = NotifyCondition.ALWAYS


class GitHubAction(BaseAction[GitHubConfig]):
    """Action to integrate with GitHub Actions.

    Provides rich integration with GitHub Actions including check runs,
    PR comments, workflow outputs, and job summaries.

    Example:
        >>> action = GitHubAction(
        ...     create_check_run=True,
        ...     set_summary=True,
        ...     fail_on_issues=True,
        ... )
        >>> result = action.execute(checkpoint_result)
    """

    action_type = "github_action"

    @classmethod
    def _default_config(cls) -> GitHubConfig:
        return GitHubConfig()

    def _execute(self, checkpoint_result: "CheckpointResult") -> ActionResult:
        """Execute GitHub Actions integrations."""
        config = self._config
        results: list[str] = []
        errors: list[str] = []

        # Get token from config or environment
        token = config.token or os.environ.get("GITHUB_TOKEN")
        repository = config.repository or os.environ.get("GITHUB_REPOSITORY")

        # Set workflow outputs
        if config.set_output:
            try:
                self._set_outputs(checkpoint_result)
                results.append("outputs set")
            except Exception as e:
                errors.append(f"outputs: {e}")

        # Write job summary
        if config.set_summary:
            try:
                self._write_summary(checkpoint_result)
                results.append("summary written")
            except Exception as e:
                errors.append(f"summary: {e}")

        # Create check run
        if config.create_check_run and token and repository:
            try:
                self._create_check_run(checkpoint_result, token, repository)
                results.append("check run created")
            except Exception as e:
                errors.append(f"check run: {e}")

        # Comment on PR
        if config.comment_on_pr and token and repository and config.pr_number:
            try:
                self._comment_on_pr(checkpoint_result, token, repository)
                results.append("PR comment added")
            except Exception as e:
                errors.append(f"PR comment: {e}")

        # Determine if we should fail
        should_fail = False
        if config.fail_on_issues:
            validation = checkpoint_result.validation_result
            if validation and validation.statistics:
                stats = validation.statistics
                if config.min_severity_to_fail == "low":
                    should_fail = stats.total_issues > 0
                elif config.min_severity_to_fail == "medium":
                    should_fail = (
                        stats.medium_issues + stats.high_issues + stats.critical_issues > 0
                    )
                elif config.min_severity_to_fail == "high":
                    should_fail = stats.high_issues + stats.critical_issues > 0
                elif config.min_severity_to_fail == "critical":
                    should_fail = stats.critical_issues > 0

        status = ActionStatus.SUCCESS if not errors else ActionStatus.FAILURE
        message = f"GitHub Actions: {', '.join(results)}" if results else "No actions performed"

        return ActionResult(
            action_name=self.name,
            action_type=self.action_type,
            status=status,
            message=message,
            details={
                "results": results,
                "errors": errors,
                "should_fail": should_fail,
                "repository": repository,
            },
            error="; ".join(errors) if errors else None,
        )

    def _set_outputs(self, checkpoint_result: "CheckpointResult") -> None:
        """Set GitHub Actions workflow outputs."""
        github_output = os.environ.get("GITHUB_OUTPUT")
        if not github_output:
            return

        validation = checkpoint_result.validation_result
        stats = validation.statistics if validation else None

        outputs = {
            "status": checkpoint_result.status.value,
            "run_id": checkpoint_result.run_id,
            "checkpoint": checkpoint_result.checkpoint_name,
            "total_issues": str(stats.total_issues if stats else 0),
            "critical_issues": str(stats.critical_issues if stats else 0),
            "high_issues": str(stats.high_issues if stats else 0),
            "pass_rate": f"{stats.pass_rate * 100:.1f}" if stats else "100.0",
            "has_issues": str(stats.total_issues > 0 if stats else False).lower(),
        }

        with open(github_output, "a") as f:
            for key, value in outputs.items():
                f.write(f"{key}={value}\n")

    def _write_summary(self, checkpoint_result: "CheckpointResult") -> None:
        """Write GitHub Actions job summary."""
        github_step_summary = os.environ.get("GITHUB_STEP_SUMMARY")
        if not github_step_summary:
            return

        validation = checkpoint_result.validation_result
        stats = validation.statistics if validation else None
        status = checkpoint_result.status.value

        # Status emoji
        status_emoji = {
            "success": ":white_check_mark:",
            "failure": ":x:",
            "error": ":exclamation:",
            "warning": ":warning:",
        }.get(status, ":question:")

        summary = f"""## {status_emoji} Data Quality Validation Report

**Checkpoint:** `{checkpoint_result.checkpoint_name}`
**Status:** {status.upper()}
**Run ID:** `{checkpoint_result.run_id}`
**Time:** {checkpoint_result.run_time.strftime('%Y-%m-%d %H:%M:%S')}

### Summary

| Metric | Value |
|--------|-------|
| Data Asset | `{checkpoint_result.data_asset}` |
| Total Issues | {stats.total_issues if stats else 0} |
| Pass Rate | {stats.pass_rate * 100 if stats else 100:.1f}% |
| Rows Checked | {stats.total_rows if stats else 0:,} |

### Issue Breakdown

| Severity | Count |
|----------|-------|
| :red_circle: Critical | {stats.critical_issues if stats else 0} |
| :orange_circle: High | {stats.high_issues if stats else 0} |
| :yellow_circle: Medium | {stats.medium_issues if stats else 0} |
| :blue_circle: Low | {stats.low_issues if stats else 0} |

---
<sub>Generated by Truthound</sub>
"""
        with open(github_step_summary, "a") as f:
            f.write(summary)

    def _create_check_run(
        self,
        checkpoint_result: "CheckpointResult",
        token: str,
        repository: str,
    ) -> None:
        """Create GitHub check run."""
        import urllib.request
        import urllib.error

        # Get SHA from environment
        sha = os.environ.get("GITHUB_SHA")
        if not sha:
            return

        validation = checkpoint_result.validation_result
        stats = validation.statistics if validation else None
        status = checkpoint_result.status.value

        # Determine conclusion
        conclusion_map = {
            "success": "success",
            "failure": "failure",
            "error": "failure",
            "warning": "neutral",
        }
        conclusion = conclusion_map.get(status, "neutral")

        # Build check run payload
        payload = {
            "name": f"Truthound: {checkpoint_result.checkpoint_name}",
            "head_sha": sha,
            "status": "completed",
            "conclusion": conclusion,
            "output": {
                "title": f"Data Quality - {status.upper()}",
                "summary": self._build_check_summary(checkpoint_result),
            },
        }

        # Add annotations if enabled
        if self._config.annotations and validation:
            annotations = self._build_annotations(checkpoint_result)
            if annotations:
                payload["output"]["annotations"] = annotations[:50]  # GitHub limit

        url = f"https://api.github.com/repos/{repository}/check-runs"
        data = json.dumps(payload).encode("utf-8")

        request = urllib.request.Request(
            url,
            data=data,
            headers={
                "Accept": "application/vnd.github.v3+json",
                "Authorization": f"token {token}",
                "Content-Type": "application/json",
            },
        )

        with urllib.request.urlopen(request) as response:
            response.read()

    def _comment_on_pr(
        self,
        checkpoint_result: "CheckpointResult",
        token: str,
        repository: str,
    ) -> None:
        """Add comment to pull request."""
        import urllib.request

        config = self._config
        if not config.pr_number:
            return

        validation = checkpoint_result.validation_result
        stats = validation.statistics if validation else None
        status = checkpoint_result.status.value

        status_emoji = {
            "success": ":white_check_mark:",
            "failure": ":x:",
            "error": ":exclamation:",
            "warning": ":warning:",
        }.get(status, ":question:")

        body = f"""## {status_emoji} Truthound Data Quality Report

**Checkpoint:** `{checkpoint_result.checkpoint_name}`
**Status:** {status.upper()}

| Metric | Value |
|--------|-------|
| Total Issues | {stats.total_issues if stats else 0} |
| Critical | {stats.critical_issues if stats else 0} |
| High | {stats.high_issues if stats else 0} |
| Pass Rate | {stats.pass_rate * 100 if stats else 100:.1f}% |

<sub>Run ID: {checkpoint_result.run_id}</sub>
"""

        url = f"https://api.github.com/repos/{repository}/issues/{config.pr_number}/comments"
        data = json.dumps({"body": body}).encode("utf-8")

        request = urllib.request.Request(
            url,
            data=data,
            headers={
                "Accept": "application/vnd.github.v3+json",
                "Authorization": f"token {token}",
                "Content-Type": "application/json",
            },
        )

        with urllib.request.urlopen(request) as response:
            response.read()

    def _build_check_summary(self, checkpoint_result: "CheckpointResult") -> str:
        """Build check run summary."""
        validation = checkpoint_result.validation_result
        stats = validation.statistics if validation else None

        return f"""### Validation Summary

- **Data Asset:** {checkpoint_result.data_asset}
- **Total Issues:** {stats.total_issues if stats else 0}
- **Pass Rate:** {stats.pass_rate * 100 if stats else 100:.1f}%

#### Issue Breakdown
- Critical: {stats.critical_issues if stats else 0}
- High: {stats.high_issues if stats else 0}
- Medium: {stats.medium_issues if stats else 0}
- Low: {stats.low_issues if stats else 0}
"""

    def _build_annotations(
        self,
        checkpoint_result: "CheckpointResult",
    ) -> list[dict[str, Any]]:
        """Build check run annotations for issues."""
        annotations = []
        validation = checkpoint_result.validation_result

        if not validation or not validation.results:
            return annotations

        for result in validation.results:
            if result.success:
                continue

            # Map severity to annotation level
            level_map = {
                "critical": "failure",
                "high": "failure",
                "medium": "warning",
                "low": "notice",
            }
            level = level_map.get(result.severity or "low", "notice")

            annotation = {
                "path": checkpoint_result.data_asset or "data",
                "start_line": 1,
                "end_line": 1,
                "annotation_level": level,
                "message": result.message or f"{result.validator_name}: {result.issue_type}",
                "title": f"[{result.severity.upper() if result.severity else 'ISSUE'}] {result.validator_name}",
            }

            annotations.append(annotation)

        return annotations

    def validate_config(self) -> list[str]:
        """Validate configuration."""
        errors = []

        if self._config.min_severity_to_fail not in ("low", "medium", "high", "critical"):
            errors.append(f"Invalid min_severity_to_fail: {self._config.min_severity_to_fail}")

        return errors
