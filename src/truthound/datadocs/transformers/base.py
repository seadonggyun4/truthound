"""Base classes and protocols for transformers.

This module defines the core abstractions for data transformers
in the report generation pipeline.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, runtime_checkable

from truthound.datadocs.engine.context import ReportContext


@dataclass
class TransformResult:
    """Result of a transformation operation.

    Attributes:
        context: The transformed context.
        modified: Whether the context was actually modified.
        changes: Description of changes made.
        warnings: Any warnings generated during transformation.
    """
    context: ReportContext
    modified: bool = False
    changes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@runtime_checkable
class Transformer(Protocol):
    """Protocol for data transformers.

    Transformers receive a ReportContext and return a (possibly modified)
    ReportContext. They should be pure functions with no side effects.

    Example implementation:
        class MyTransformer:
            def transform(self, ctx: ReportContext) -> ReportContext:
                # Modify and return a new context
                return ctx.with_option("my_option", "value")
    """

    def transform(self, ctx: ReportContext) -> ReportContext:
        """Transform the report context.

        Args:
            ctx: Input context.

        Returns:
            Transformed context (may be the same instance if unchanged).
        """
        ...


class BaseTransformer(ABC):
    """Abstract base class for transformers.

    Provides common functionality and ensures consistent interface.
    Subclasses implement the _do_transform method.
    """

    def __init__(self, name: str | None = None) -> None:
        """Initialize the transformer.

        Args:
            name: Optional name for this transformer instance.
        """
        self._name = name or self.__class__.__name__

    @property
    def name(self) -> str:
        """Get the transformer name."""
        return self._name

    def transform(self, ctx: ReportContext) -> ReportContext:
        """Transform the report context.

        This method wraps _do_transform with common processing:
        - Adds trace entry
        - Handles errors gracefully

        Args:
            ctx: Input context.

        Returns:
            Transformed context.
        """
        try:
            result = self._do_transform(ctx)

            # If result is TransformResult, extract context
            if isinstance(result, TransformResult):
                return result.context

            return result

        except Exception as e:
            # Add error information to context and continue
            error_info = {
                "transformer": self._name,
                "error": str(e),
                "type": type(e).__name__,
            }
            return ctx.with_option(
                "_transform_errors",
                [*ctx.get_option("_transform_errors", []), error_info]
            )

    @abstractmethod
    def _do_transform(self, ctx: ReportContext) -> ReportContext | TransformResult:
        """Perform the actual transformation.

        Subclasses implement this method.

        Args:
            ctx: Input context.

        Returns:
            Transformed context or TransformResult.
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self._name!r})"


class ChainedTransformer(BaseTransformer):
    """A transformer that chains multiple transformers together.

    This allows composing transformers into a single unit.

    Example:
        chained = ChainedTransformer([
            I18nTransformer(locale="ko"),
            FilterTransformer(sections=["overview"]),
        ])
        ctx = chained.transform(ctx)
    """

    def __init__(
        self,
        transformers: list[Transformer],
        name: str | None = None,
    ) -> None:
        """Initialize with a list of transformers.

        Args:
            transformers: List of transformers to chain.
            name: Optional name for this chain.
        """
        super().__init__(name=name or "ChainedTransformer")
        self._transformers = list(transformers)

    @property
    def transformers(self) -> list[Transformer]:
        """Get the list of transformers."""
        return list(self._transformers)

    def add(self, transformer: Transformer) -> "ChainedTransformer":
        """Add a transformer to the chain.

        Args:
            transformer: Transformer to add.

        Returns:
            Self for chaining.
        """
        self._transformers.append(transformer)
        return self

    def _do_transform(self, ctx: ReportContext) -> ReportContext:
        """Apply all transformers in order.

        Args:
            ctx: Input context.

        Returns:
            Transformed context after all transformers.
        """
        for transformer in self._transformers:
            ctx = transformer.transform(ctx)
        return ctx


class ConditionalTransformer(BaseTransformer):
    """A transformer that conditionally applies based on a predicate.

    Example:
        # Only apply i18n if locale is not English
        conditional = ConditionalTransformer(
            transformer=I18nTransformer(),
            condition=lambda ctx: ctx.locale != "en",
        )
    """

    def __init__(
        self,
        transformer: Transformer,
        condition: Callable[[ReportContext], bool],
        name: str | None = None,
    ) -> None:
        """Initialize with a transformer and condition.

        Args:
            transformer: The transformer to conditionally apply.
            condition: Predicate that returns True to apply transformer.
            name: Optional name.
        """
        super().__init__(name=name or "ConditionalTransformer")
        self._transformer = transformer
        self._condition = condition

    def _do_transform(self, ctx: ReportContext) -> ReportContext:
        """Apply transformer if condition is met.

        Args:
            ctx: Input context.

        Returns:
            Transformed context (or unchanged if condition fails).
        """
        if self._condition(ctx):
            return self._transformer.transform(ctx)
        return ctx


class NoOpTransformer(BaseTransformer):
    """A transformer that does nothing (passthrough).

    Useful for testing and as a placeholder.
    """

    def __init__(self, name: str | None = None) -> None:
        super().__init__(name=name or "NoOpTransformer")

    def _do_transform(self, ctx: ReportContext) -> ReportContext:
        return ctx


class LambdaTransformer(BaseTransformer):
    """A transformer created from a lambda/function.

    Example:
        transformer = LambdaTransformer(
            lambda ctx: ctx.with_option("processed", True),
            name="SetProcessedFlag",
        )
    """

    def __init__(
        self,
        func: Callable[[ReportContext], ReportContext],
        name: str | None = None,
    ) -> None:
        """Initialize with a transformation function.

        Args:
            func: Function that takes and returns a ReportContext.
            name: Optional name for this transformer.
        """
        super().__init__(name=name or "LambdaTransformer")
        self._func = func

    def _do_transform(self, ctx: ReportContext) -> ReportContext:
        return self._func(ctx)
