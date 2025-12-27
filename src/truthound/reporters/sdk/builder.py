"""Reporter builder utilities for quick reporter creation.

This module provides convenient ways to create custom reporters:
- Decorator-based reporter creation
- Fluent builder pattern
- Pre-configured reporter templates

Example:
    >>> # Simple decorator-based reporter
    >>> @create_reporter("my_format", extension=".myf")
    ... def render_my_format(result, config):
    ...     return f"Status: {result.status.value}"
    >>>
    >>> # Using the builder pattern
    >>> reporter_class = (
    ...     ReporterBuilder("custom")
    ...     .with_extension(".custom")
    ...     .with_content_type("text/custom")
    ...     .with_config_class(MyConfig)
    ...     .with_renderer(my_render_function)
    ...     .with_mixin(FormattingMixin)
    ...     .build()
    ... )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Type, TypeVar

from truthound.reporters.base import (
    BaseReporter,
    ReporterConfig,
    ValidationReporter,
)
from truthound.reporters.factory import register_reporter

if TYPE_CHECKING:
    from truthound.stores.results import ValidationResult


ConfigT = TypeVar("ConfigT", bound=ReporterConfig)
T = TypeVar("T")


# =============================================================================
# Decorator-Based Reporter Creation
# =============================================================================


def create_reporter(
    name: str,
    *,
    extension: str = ".txt",
    content_type: str = "text/plain",
    config_class: Type[ReporterConfig] | None = None,
    register: bool = True,
) -> Callable[[Callable[[Any, ReporterConfig], str]], Type[BaseReporter[Any, Any]]]:
    """Decorator to create a reporter from a render function.

    This is the simplest way to create a custom reporter. Just define
    a function that takes a result and config, and returns a string.

    Args:
        name: Reporter name (used in get_reporter).
        extension: File extension for output files.
        content_type: MIME content type.
        config_class: Optional custom config class.
        register: Whether to register with the factory.

    Returns:
        Decorator that creates a reporter class.

    Example:
        >>> @create_reporter("simple_text", extension=".txt")
        ... def render_simple(result, config):
        ...     lines = [f"Validation Result: {result.status.value}"]
        ...     for r in result.results:
        ...         if not r.success:
        ...             lines.append(f"  - {r.validator_name}: {r.message}")
        ...     return "\\n".join(lines)
        >>>
        >>> # Now usable via factory
        >>> reporter = get_reporter("simple_text")
        >>> output = reporter.render(validation_result)
    """

    def decorator(
        render_func: Callable[[Any, ReporterConfig], str],
    ) -> Type[BaseReporter[Any, Any]]:
        # Determine config class
        actual_config_class = config_class or ReporterConfig

        # Create the reporter class dynamically
        class GeneratedReporter(BaseReporter[ReporterConfig, Any]):
            __doc__ = render_func.__doc__ or f"Auto-generated reporter: {name}"

            # Class attributes
            name = name  # type: ignore
            file_extension = extension
            content_type = content_type

            @classmethod
            def _default_config(cls) -> ReporterConfig:
                return actual_config_class()

            def render(self, data: Any) -> str:
                return render_func(data, self._config)

        # Set class name
        GeneratedReporter.__name__ = f"{name.title().replace('_', '')}Reporter"
        GeneratedReporter.__qualname__ = GeneratedReporter.__name__

        # Register with factory
        if register:
            register_reporter(name)(GeneratedReporter)

        return GeneratedReporter

    return decorator


def create_validation_reporter(
    name: str,
    *,
    extension: str = ".txt",
    content_type: str = "text/plain",
    config_class: Type[ReporterConfig] | None = None,
    register: bool = True,
) -> Callable[
    [Callable[["ValidationResult", ReporterConfig], str]],
    Type[ValidationReporter[Any]],
]:
    """Decorator to create a validation reporter from a render function.

    Similar to create_reporter but specialized for ValidationResult input.

    Args:
        name: Reporter name.
        extension: File extension.
        content_type: MIME content type.
        config_class: Optional custom config class.
        register: Whether to register with factory.

    Returns:
        Decorator that creates a ValidationReporter class.

    Example:
        >>> @create_validation_reporter("severity_summary", extension=".txt")
        ... def render_severity_summary(result, config):
        ...     counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}
        ...     for r in result.results:
        ...         if not r.success and r.severity:
        ...             counts[r.severity.lower()] = counts.get(r.severity.lower(), 0) + 1
        ...     return "\\n".join(f"{k}: {v}" for k, v in counts.items())
    """

    def decorator(
        render_func: Callable[["ValidationResult", ReporterConfig], str],
    ) -> Type[ValidationReporter[Any]]:
        actual_config_class = config_class or ReporterConfig

        class GeneratedValidationReporter(ValidationReporter[ReporterConfig]):
            __doc__ = render_func.__doc__ or f"Auto-generated validation reporter: {name}"

            # Class attributes - avoid name collision
            reporter_name = name
            file_extension = extension
            content_type = content_type

            @classmethod
            def _default_config(cls) -> ReporterConfig:
                return actual_config_class()

            def render(self, data: "ValidationResult") -> str:
                return render_func(data, self._config)

        # Override the name attribute properly
        GeneratedValidationReporter.name = name  # type: ignore

        GeneratedValidationReporter.__name__ = f"{name.title().replace('_', '')}Reporter"
        GeneratedValidationReporter.__qualname__ = GeneratedValidationReporter.__name__

        if register:
            register_reporter(name)(GeneratedValidationReporter)

        return GeneratedValidationReporter

    return decorator


# =============================================================================
# Builder Pattern
# =============================================================================


@dataclass
class ReporterBuilder:
    """Fluent builder for creating custom reporter classes.

    Provides a step-by-step way to configure and build reporter classes
    with mixins, custom configs, and render functions.

    Example:
        >>> from truthound.reporters.sdk import ReporterBuilder, FormattingMixin
        >>>
        >>> # Build a custom reporter with mixins
        >>> MyReporter = (
        ...     ReporterBuilder("custom_report")
        ...     .with_extension(".rpt")
        ...     .with_content_type("text/plain")
        ...     .with_mixin(FormattingMixin)
        ...     .with_renderer(lambda self, data: f"Report: {data.status}")
        ...     .build()
        ... )
        >>>
        >>> # Create instance and use
        >>> reporter = MyReporter()
        >>> output = reporter.render(validation_result)
    """

    name: str
    extension: str = ".txt"
    content_type: str = "text/plain"
    config_class: Type[ReporterConfig] = field(default=ReporterConfig)
    base_class: Type[BaseReporter[Any, Any]] = field(default=BaseReporter)  # type: ignore
    mixins: list[type] = field(default_factory=list)
    renderer: Callable[[Any, Any], str] | None = None
    post_processors: list[Callable[[str], str]] = field(default_factory=list)
    register_name: str | None = None
    class_attributes: dict[str, Any] = field(default_factory=dict)

    def with_extension(self, extension: str) -> "ReporterBuilder":
        """Set the file extension.

        Args:
            extension: File extension (e.g., ".json").

        Returns:
            Self for chaining.
        """
        self.extension = extension
        return self

    def with_content_type(self, content_type: str) -> "ReporterBuilder":
        """Set the MIME content type.

        Args:
            content_type: Content type (e.g., "application/json").

        Returns:
            Self for chaining.
        """
        self.content_type = content_type
        return self

    def with_config_class(self, config_class: Type[ReporterConfig]) -> "ReporterBuilder":
        """Set the configuration class.

        Args:
            config_class: Configuration dataclass.

        Returns:
            Self for chaining.
        """
        self.config_class = config_class
        return self

    def with_base_class(
        self,
        base_class: Type[BaseReporter[Any, Any]],
    ) -> "ReporterBuilder":
        """Set the base class.

        Args:
            base_class: Base reporter class.

        Returns:
            Self for chaining.
        """
        self.base_class = base_class
        return self

    def for_validation_result(self) -> "ReporterBuilder":
        """Configure for ValidationResult input.

        Returns:
            Self for chaining.
        """
        self.base_class = ValidationReporter  # type: ignore
        return self

    def with_mixin(self, mixin: type) -> "ReporterBuilder":
        """Add a mixin class.

        Args:
            mixin: Mixin class to add.

        Returns:
            Self for chaining.
        """
        self.mixins.append(mixin)
        return self

    def with_mixins(self, *mixins: type) -> "ReporterBuilder":
        """Add multiple mixin classes.

        Args:
            *mixins: Mixin classes to add.

        Returns:
            Self for chaining.
        """
        self.mixins.extend(mixins)
        return self

    def with_renderer(
        self,
        renderer: Callable[[Any, Any], str],
    ) -> "ReporterBuilder":
        """Set the render function.

        The function receives (self, data) where self is the reporter instance.

        Args:
            renderer: Render function.

        Returns:
            Self for chaining.
        """
        self.renderer = renderer
        return self

    def with_post_processor(
        self,
        processor: Callable[[str], str],
    ) -> "ReporterBuilder":
        """Add a post-processor for the rendered output.

        Args:
            processor: Function that transforms the output.

        Returns:
            Self for chaining.
        """
        self.post_processors.append(processor)
        return self

    def with_attribute(self, name: str, value: Any) -> "ReporterBuilder":
        """Add a class attribute.

        Args:
            name: Attribute name.
            value: Attribute value.

        Returns:
            Self for chaining.
        """
        self.class_attributes[name] = value
        return self

    def register_as(self, name: str) -> "ReporterBuilder":
        """Set name for factory registration.

        Args:
            name: Name to register under (None to skip registration).

        Returns:
            Self for chaining.
        """
        self.register_name = name
        return self

    def build(self) -> Type[BaseReporter[Any, Any]]:
        """Build the reporter class.

        Returns:
            Generated reporter class.

        Raises:
            ValueError: If no renderer is set.
        """
        if self.renderer is None:
            raise ValueError("Renderer function must be set via with_renderer()")

        # Capture builder state for closure
        name = self.name
        extension = self.extension
        content_type = self.content_type
        config_class = self.config_class
        renderer = self.renderer
        post_processors = list(self.post_processors)
        class_attrs = dict(self.class_attributes)

        # Build base classes tuple
        bases = tuple(self.mixins) + (self.base_class,)

        # Create class dictionary
        class_dict: dict[str, Any] = {
            "name": name,
            "file_extension": extension,
            "content_type": content_type,
            **class_attrs,
        }

        # Add _default_config classmethod
        def _default_config(cls: type) -> ReporterConfig:
            return config_class()

        class_dict["_default_config"] = classmethod(_default_config)

        # Add render method
        def render(self: Any, data: Any) -> str:
            result = renderer(self, data)
            for processor in post_processors:
                result = processor(result)
            return result

        class_dict["render"] = render

        # Create the class
        class_name = f"{name.title().replace('_', '')}Reporter"
        reporter_class = type(class_name, bases, class_dict)

        # Register if requested
        if self.register_name:
            register_reporter(self.register_name)(reporter_class)

        return reporter_class


# =============================================================================
# Pre-Built Reporter Templates
# =============================================================================


def create_line_based_reporter(
    name: str,
    line_formatter: Callable[["ValidatorResult", int], str],
    header: str | Callable[["ValidationResult"], str] | None = None,
    footer: str | Callable[["ValidationResult"], str] | None = None,
    separator: str = "\n",
    include_passed: bool = False,
    extension: str = ".txt",
    register: bool = True,
) -> Type[ValidationReporter[Any]]:
    """Create a reporter that formats each result as a line.

    Args:
        name: Reporter name.
        line_formatter: Function to format each ValidatorResult.
        header: Optional header (string or function).
        footer: Optional footer (string or function).
        separator: Line separator.
        include_passed: Whether to include passed results.
        extension: File extension.
        register: Whether to register with factory.

    Returns:
        Generated reporter class.

    Example:
        >>> def format_line(result, index):
        ...     status = "✓" if result.success else "✗"
        ...     return f"{index}. [{status}] {result.validator_name}: {result.message}"
        >>>
        >>> MyReporter = create_line_based_reporter(
        ...     "checklist",
        ...     line_formatter=format_line,
        ...     header="Validation Checklist:\\n",
        ... )
    """

    @create_validation_reporter(name, extension=extension, register=register)
    def render_lines(result: "ValidationResult", config: ReporterConfig) -> str:
        lines = []

        # Add header
        if header:
            if callable(header):
                lines.append(header(result))
            else:
                lines.append(header)

        # Format each result
        index = 1
        for validator_result in result.results:
            if not include_passed and validator_result.success:
                continue
            lines.append(line_formatter(validator_result, index))
            index += 1

        # Add footer
        if footer:
            if callable(footer):
                lines.append(footer(result))
            else:
                lines.append(footer)

        return separator.join(lines)

    return render_lines  # type: ignore


def create_structured_reporter(
    name: str,
    structure_builder: Callable[["ValidationResult"], dict[str, Any]],
    serializer: Callable[[dict[str, Any]], str] = None,  # type: ignore
    extension: str = ".json",
    content_type: str = "application/json",
    register: bool = True,
) -> Type[ValidationReporter[Any]]:
    """Create a reporter that builds a structured output.

    Args:
        name: Reporter name.
        structure_builder: Function to build output structure.
        serializer: Function to serialize structure to string (default: JSON).
        extension: File extension.
        content_type: MIME content type.
        register: Whether to register with factory.

    Returns:
        Generated reporter class.

    Example:
        >>> def build_structure(result):
        ...     return {
        ...         "status": result.status.value,
        ...         "issues": [
        ...             {"name": r.validator_name, "message": r.message}
        ...             for r in result.results if not r.success
        ...         ]
        ...     }
        >>>
        >>> MyReporter = create_structured_reporter(
        ...     "simple_json",
        ...     structure_builder=build_structure,
        ... )
    """
    import json

    if serializer is None:
        def default_serializer(data: dict[str, Any]) -> str:
            return json.dumps(data, indent=2, default=str)
        serializer = default_serializer

    @create_validation_reporter(
        name,
        extension=extension,
        content_type=content_type,
        register=register,
    )
    def render_structured(result: "ValidationResult", config: ReporterConfig) -> str:
        structure = structure_builder(result)
        return serializer(structure)

    return render_structured  # type: ignore
