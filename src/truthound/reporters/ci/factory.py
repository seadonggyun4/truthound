"""CI Reporter Factory.

This module provides factory functions for creating CI reporters,
including auto-detection of the current CI platform.
"""

from __future__ import annotations

from typing import Any, Callable, Type

from truthound.reporters.ci.base import BaseCIReporter, CIPlatform


# Type alias for reporter class
CIReporterClass = Type[BaseCIReporter]

# Registry of CI reporter classes
_ci_reporter_registry: dict[str, CIReporterClass] = {}


def register_ci_reporter(name: str) -> Callable[[CIReporterClass], CIReporterClass]:
    """Decorator to register a CI reporter class.

    Args:
        name: Name to register the reporter under.

    Returns:
        Decorator function.

    Example:
        >>> @register_ci_reporter("my_ci")
        ... class MyCIReporter(BaseCIReporter):
        ...     pass
    """

    def decorator(cls: CIReporterClass) -> CIReporterClass:
        _ci_reporter_registry[name] = cls
        return cls

    return decorator


def get_ci_reporter(
    platform: str | CIPlatform | None = None,
    **kwargs: Any,
) -> BaseCIReporter:
    """Create a CI reporter for the specified or detected platform.

    This is the main entry point for creating CI reporters. It handles
    auto-detection of the CI platform and creates an appropriate reporter.

    Args:
        platform: CI platform name or CIPlatform enum. If None, auto-detects.
            Options: "github", "gitlab", "jenkins", "azure", "circleci",
                    "bitbucket", "auto", or CIPlatform enum value.
        **kwargs: Configuration options to pass to the reporter.

    Returns:
        Configured CI reporter instance.

    Raises:
        ValueError: If the platform is not recognized.

    Example:
        >>> # Auto-detect platform
        >>> reporter = get_ci_reporter()
        >>> reporter.report_to_ci(result)
        >>>
        >>> # Specify platform
        >>> reporter = get_ci_reporter("github", emoji_enabled=False)
        >>> reporter.report_to_ci(result)
        >>>
        >>> # Use enum
        >>> reporter = get_ci_reporter(CIPlatform.GITLAB_CI)
    """
    # Import detection here to avoid circular imports
    from truthound.reporters.ci.detection import detect_ci_platform

    # Handle auto-detection
    if platform is None or platform == "auto":
        detected = detect_ci_platform()
        if detected is None:
            # Not in CI, return a generic console-like reporter
            from truthound.reporters.ci.github import GitHubActionsReporter
            return GitHubActionsReporter(**kwargs)
        platform = detected

    # Convert enum to string for lookup
    if isinstance(platform, CIPlatform):
        platform_str = platform.value
    else:
        platform_str = str(platform).lower().strip()

    # Check custom registry first
    if platform_str in _ci_reporter_registry:
        return _ci_reporter_registry[platform_str](**kwargs)

    # Built-in reporters
    if platform_str in ("github", "github_actions", "github-actions"):
        from truthound.reporters.ci.github import GitHubActionsReporter
        return GitHubActionsReporter(**kwargs)

    elif platform_str in ("gitlab", "gitlab_ci", "gitlab-ci"):
        from truthound.reporters.ci.gitlab import GitLabCIReporter
        return GitLabCIReporter(**kwargs)

    elif platform_str == "jenkins":
        from truthound.reporters.ci.jenkins import JenkinsReporter
        return JenkinsReporter(**kwargs)

    elif platform_str in ("azure", "azure_devops", "azure-devops", "azdo", "vsts"):
        from truthound.reporters.ci.azure import AzureDevOpsReporter
        return AzureDevOpsReporter(**kwargs)

    elif platform_str in ("circleci", "circle", "circle_ci", "circle-ci"):
        from truthound.reporters.ci.circleci import CircleCIReporter
        return CircleCIReporter(**kwargs)

    elif platform_str in ("bitbucket", "bitbucket_pipelines", "bitbucket-pipelines"):
        from truthound.reporters.ci.bitbucket import BitbucketPipelinesReporter
        return BitbucketPipelinesReporter(**kwargs)

    elif platform_str == "generic":
        # Use GitHub Actions reporter as a reasonable default
        from truthound.reporters.ci.github import GitHubActionsReporter
        return GitHubActionsReporter(**kwargs)

    else:
        available = list(_ci_reporter_registry.keys()) + [
            "github", "gitlab", "jenkins", "azure", "circleci", "bitbucket"
        ]
        raise ValueError(
            f"Unknown CI platform: {platform_str}. "
            f"Available: {', '.join(sorted(set(available)))}"
        )


def list_ci_platforms() -> list[str]:
    """List all available CI platforms.

    Returns:
        List of platform names.
    """
    built_in = ["github", "gitlab", "jenkins", "azure", "circleci", "bitbucket"]
    custom = list(_ci_reporter_registry.keys())
    return sorted(set(built_in + custom))


def get_reporter_for_platform(platform: CIPlatform) -> CIReporterClass | None:
    """Get the reporter class for a specific platform.

    Args:
        platform: The CI platform.

    Returns:
        Reporter class, or None if not found.
    """
    platform_to_reporter: dict[CIPlatform, Callable[[], CIReporterClass]] = {
        CIPlatform.GITHUB_ACTIONS: lambda: __import__(
            "truthound.reporters.ci.github", fromlist=["GitHubActionsReporter"]
        ).GitHubActionsReporter,
        CIPlatform.GITLAB_CI: lambda: __import__(
            "truthound.reporters.ci.gitlab", fromlist=["GitLabCIReporter"]
        ).GitLabCIReporter,
        CIPlatform.JENKINS: lambda: __import__(
            "truthound.reporters.ci.jenkins", fromlist=["JenkinsReporter"]
        ).JenkinsReporter,
        CIPlatform.AZURE_DEVOPS: lambda: __import__(
            "truthound.reporters.ci.azure", fromlist=["AzureDevOpsReporter"]
        ).AzureDevOpsReporter,
        CIPlatform.CIRCLECI: lambda: __import__(
            "truthound.reporters.ci.circleci", fromlist=["CircleCIReporter"]
        ).CircleCIReporter,
        CIPlatform.BITBUCKET: lambda: __import__(
            "truthound.reporters.ci.bitbucket", fromlist=["BitbucketPipelinesReporter"]
        ).BitbucketPipelinesReporter,
    }

    loader = platform_to_reporter.get(platform)
    if loader:
        return loader()
    return None
