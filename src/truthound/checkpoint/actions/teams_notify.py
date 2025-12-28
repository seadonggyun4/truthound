"""Microsoft Teams notification action.

This module provides Microsoft Teams integration for checkpoint notifications
using Incoming Webhooks and Adaptive Cards.

Features:
    - Adaptive Card formatting for rich message display
    - Multiple connector types (Incoming Webhook, Power Automate)
    - Customizable message templates
    - Action buttons for quick navigation
    - @mention support for users and channels
    - Thread/reply support for conversation continuity

Example:
    >>> from truthound.checkpoint.actions import TeamsNotification
    >>>
    >>> action = TeamsNotification(
    ...     webhook_url="https://outlook.office.com/webhook/...",
    ...     notify_on="failure",
    ...     include_actions=True,
    ...     mention_on_failure=["user@example.com"],
    ... )
    >>> result = action.execute(checkpoint_result)

References:
    - Adaptive Cards: https://adaptivecards.io/
    - Teams Webhooks: https://docs.microsoft.com/en-us/microsoftteams/platform/webhooks-and-connectors/
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from truthound.checkpoint.actions.base import (
    ActionConfig,
    ActionResult,
    ActionStatus,
    BaseAction,
    NotifyCondition,
)

if TYPE_CHECKING:
    from truthound.checkpoint.checkpoint import CheckpointResult


# =============================================================================
# Enums and Constants
# =============================================================================


class TeamsConnectorType(str, Enum):
    """Types of Teams connectors supported."""

    INCOMING_WEBHOOK = "incoming_webhook"
    POWER_AUTOMATE = "power_automate"
    LOGIC_APPS = "logic_apps"

    def __str__(self) -> str:
        return self.value


class AdaptiveCardVersion(str, Enum):
    """Adaptive Card schema versions."""

    V1_0 = "1.0"
    V1_2 = "1.2"
    V1_3 = "1.3"
    V1_4 = "1.4"
    V1_5 = "1.5"

    def __str__(self) -> str:
        return self.value


class MessageTheme(str, Enum):
    """Pre-defined message themes."""

    DEFAULT = "default"
    MINIMAL = "minimal"
    DETAILED = "detailed"
    COMPACT = "compact"

    def __str__(self) -> str:
        return self.value


class CardContainerStyle(str, Enum):
    """Container styles for Adaptive Cards."""

    DEFAULT = "default"
    EMPHASIS = "emphasis"
    GOOD = "good"
    ATTENTION = "attention"
    WARNING = "warning"
    ACCENT = "accent"

    def __str__(self) -> str:
        return self.value


class TextWeight(str, Enum):
    """Text weight options."""

    DEFAULT = "default"
    LIGHTER = "lighter"
    BOLDER = "bolder"

    def __str__(self) -> str:
        return self.value


class TextSize(str, Enum):
    """Text size options."""

    DEFAULT = "default"
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"
    EXTRA_LARGE = "extraLarge"

    def __str__(self) -> str:
        return self.value


class TextColor(str, Enum):
    """Text color options."""

    DEFAULT = "default"
    DARK = "dark"
    LIGHT = "light"
    ACCENT = "accent"
    GOOD = "good"
    WARNING = "warning"
    ATTENTION = "attention"

    def __str__(self) -> str:
        return self.value


# =============================================================================
# Adaptive Card Builder (Builder Pattern)
# =============================================================================


@runtime_checkable
class CardElement(Protocol):
    """Protocol for Adaptive Card elements."""

    def to_dict(self) -> dict[str, Any]:
        """Convert element to dictionary representation."""
        ...


class AdaptiveCardBuilder:
    """Fluent builder for creating Adaptive Cards.

    This builder provides a type-safe, fluent API for constructing
    Microsoft Adaptive Cards with proper schema validation.

    Example:
        >>> card = (
        ...     AdaptiveCardBuilder()
        ...     .set_version("1.4")
        ...     .add_text_block("Hello World", weight="bolder", size="large")
        ...     .add_fact_set([("Name", "John"), ("Status", "Active")])
        ...     .add_action_open_url("View Details", "https://example.com")
        ...     .build()
        ... )
    """

    def __init__(self, version: str | AdaptiveCardVersion = AdaptiveCardVersion.V1_4) -> None:
        """Initialize the builder.

        Args:
            version: Adaptive Card schema version to use.
        """
        self._version = str(version)
        self._body: list[dict[str, Any]] = []
        self._actions: list[dict[str, Any]] = []
        self._fallback_text: str | None = None
        self._speak: str | None = None
        self._ms_teams: dict[str, Any] = {}
        self._min_height: str | None = None
        self._vertical_content_alignment: str | None = None

    def set_version(self, version: str | AdaptiveCardVersion) -> "AdaptiveCardBuilder":
        """Set the Adaptive Card schema version.

        Args:
            version: Schema version string (e.g., "1.4").

        Returns:
            Self for method chaining.
        """
        self._version = str(version)
        return self

    def set_fallback_text(self, text: str) -> "AdaptiveCardBuilder":
        """Set fallback text for clients that don't support Adaptive Cards.

        Args:
            text: Plain text fallback.

        Returns:
            Self for method chaining.
        """
        self._fallback_text = text
        return self

    def set_speak(self, speak: str) -> "AdaptiveCardBuilder":
        """Set speech text for accessibility.

        Args:
            speak: SSML or plain text for speech.

        Returns:
            Self for method chaining.
        """
        self._speak = speak
        return self

    def set_min_height(self, height: str) -> "AdaptiveCardBuilder":
        """Set minimum height of the card.

        Args:
            height: CSS height value (e.g., "200px").

        Returns:
            Self for method chaining.
        """
        self._min_height = height
        return self

    def enable_full_width(self) -> "AdaptiveCardBuilder":
        """Enable full-width display in Teams.

        Returns:
            Self for method chaining.
        """
        self._ms_teams["width"] = "Full"
        return self

    # -------------------------------------------------------------------------
    # Text Elements
    # -------------------------------------------------------------------------

    def add_text_block(
        self,
        text: str,
        *,
        weight: str | TextWeight = TextWeight.DEFAULT,
        size: str | TextSize = TextSize.DEFAULT,
        color: str | TextColor = TextColor.DEFAULT,
        wrap: bool = True,
        is_subtle: bool = False,
        max_lines: int | None = None,
        horizontal_alignment: str = "left",
        spacing: str = "default",
        separator: bool = False,
    ) -> "AdaptiveCardBuilder":
        """Add a text block element.

        Args:
            text: Text content (supports Markdown in some contexts).
            weight: Font weight.
            size: Font size.
            color: Text color.
            wrap: Whether text should wrap.
            is_subtle: Whether to display as subtle/muted.
            max_lines: Maximum number of lines to display.
            horizontal_alignment: Text alignment (left, center, right).
            spacing: Spacing above this element.
            separator: Whether to show a separator line above.

        Returns:
            Self for method chaining.
        """
        element: dict[str, Any] = {
            "type": "TextBlock",
            "text": text,
            "wrap": wrap,
        }

        if str(weight) != "default":
            element["weight"] = str(weight)
        if str(size) != "default":
            element["size"] = str(size)
        if str(color) != "default":
            element["color"] = str(color)
        if is_subtle:
            element["isSubtle"] = True
        if max_lines is not None:
            element["maxLines"] = max_lines
        if horizontal_alignment != "left":
            element["horizontalAlignment"] = horizontal_alignment
        if spacing != "default":
            element["spacing"] = spacing
        if separator:
            element["separator"] = True

        self._body.append(element)
        return self

    def add_rich_text_block(
        self,
        inlines: list[dict[str, Any]],
        *,
        horizontal_alignment: str = "left",
        spacing: str = "default",
    ) -> "AdaptiveCardBuilder":
        """Add a rich text block with inline elements.

        Args:
            inlines: List of inline text elements.
            horizontal_alignment: Text alignment.
            spacing: Spacing above this element.

        Returns:
            Self for method chaining.
        """
        element: dict[str, Any] = {
            "type": "RichTextBlock",
            "inlines": inlines,
        }

        if horizontal_alignment != "left":
            element["horizontalAlignment"] = horizontal_alignment
        if spacing != "default":
            element["spacing"] = spacing

        self._body.append(element)
        return self

    # -------------------------------------------------------------------------
    # Container Elements
    # -------------------------------------------------------------------------

    def add_container(
        self,
        items: list[dict[str, Any]],
        *,
        style: str | CardContainerStyle = CardContainerStyle.DEFAULT,
        bleed: bool = False,
        min_height: str | None = None,
        spacing: str = "default",
        separator: bool = False,
    ) -> "AdaptiveCardBuilder":
        """Add a container element.

        Args:
            items: Child elements in the container.
            style: Container style.
            bleed: Whether to bleed to card edges.
            min_height: Minimum container height.
            spacing: Spacing above this element.
            separator: Whether to show a separator.

        Returns:
            Self for method chaining.
        """
        element: dict[str, Any] = {
            "type": "Container",
            "items": items,
        }

        if str(style) != "default":
            element["style"] = str(style)
        if bleed:
            element["bleed"] = True
        if min_height:
            element["minHeight"] = min_height
        if spacing != "default":
            element["spacing"] = spacing
        if separator:
            element["separator"] = True

        self._body.append(element)
        return self

    def add_column_set(
        self,
        columns: list[dict[str, Any]],
        *,
        spacing: str = "default",
        separator: bool = False,
    ) -> "AdaptiveCardBuilder":
        """Add a column set for horizontal layout.

        Args:
            columns: List of column definitions.
            spacing: Spacing above this element.
            separator: Whether to show a separator.

        Returns:
            Self for method chaining.
        """
        element: dict[str, Any] = {
            "type": "ColumnSet",
            "columns": columns,
        }

        if spacing != "default":
            element["spacing"] = spacing
        if separator:
            element["separator"] = True

        self._body.append(element)
        return self

    # -------------------------------------------------------------------------
    # Data Display Elements
    # -------------------------------------------------------------------------

    def add_fact_set(
        self,
        facts: list[tuple[str, str]],
        *,
        spacing: str = "default",
        separator: bool = False,
    ) -> "AdaptiveCardBuilder":
        """Add a fact set for key-value display.

        Args:
            facts: List of (title, value) tuples.
            spacing: Spacing above this element.
            separator: Whether to show a separator.

        Returns:
            Self for method chaining.
        """
        element: dict[str, Any] = {
            "type": "FactSet",
            "facts": [{"title": title, "value": value} for title, value in facts],
        }

        if spacing != "default":
            element["spacing"] = spacing
        if separator:
            element["separator"] = True

        self._body.append(element)
        return self

    def add_image(
        self,
        url: str,
        *,
        alt_text: str = "",
        size: str = "auto",
        style: str = "default",
        horizontal_alignment: str = "left",
        spacing: str = "default",
    ) -> "AdaptiveCardBuilder":
        """Add an image element.

        Args:
            url: Image URL.
            alt_text: Alternative text for accessibility.
            size: Image size (auto, stretch, small, medium, large).
            style: Image style (default, person for circular).
            horizontal_alignment: Image alignment.
            spacing: Spacing above this element.

        Returns:
            Self for method chaining.
        """
        element: dict[str, Any] = {
            "type": "Image",
            "url": url,
        }

        if alt_text:
            element["altText"] = alt_text
        if size != "auto":
            element["size"] = size
        if style != "default":
            element["style"] = style
        if horizontal_alignment != "left":
            element["horizontalAlignment"] = horizontal_alignment
        if spacing != "default":
            element["spacing"] = spacing

        self._body.append(element)
        return self

    def add_image_set(
        self,
        images: list[dict[str, Any]],
        *,
        image_size: str = "medium",
        spacing: str = "default",
    ) -> "AdaptiveCardBuilder":
        """Add an image set element.

        Args:
            images: List of image definitions.
            image_size: Size for all images.
            spacing: Spacing above this element.

        Returns:
            Self for method chaining.
        """
        element: dict[str, Any] = {
            "type": "ImageSet",
            "images": images,
            "imageSize": image_size,
        }

        if spacing != "default":
            element["spacing"] = spacing

        self._body.append(element)
        return self

    # -------------------------------------------------------------------------
    # Action Elements
    # -------------------------------------------------------------------------

    def add_action_open_url(
        self,
        title: str,
        url: str,
        *,
        icon_url: str | None = None,
        tooltip: str | None = None,
        style: str = "default",
    ) -> "AdaptiveCardBuilder":
        """Add an action that opens a URL.

        Args:
            title: Button text.
            url: URL to open.
            icon_url: Optional icon URL.
            tooltip: Hover tooltip text.
            style: Button style (default, positive, destructive).

        Returns:
            Self for method chaining.
        """
        action: dict[str, Any] = {
            "type": "Action.OpenUrl",
            "title": title,
            "url": url,
        }

        if icon_url:
            action["iconUrl"] = icon_url
        if tooltip:
            action["tooltip"] = tooltip
        if style != "default":
            action["style"] = style

        self._actions.append(action)
        return self

    def add_action_submit(
        self,
        title: str,
        data: dict[str, Any] | None = None,
        *,
        style: str = "default",
    ) -> "AdaptiveCardBuilder":
        """Add an action that submits data.

        Args:
            title: Button text.
            data: Data to submit.
            style: Button style.

        Returns:
            Self for method chaining.
        """
        action: dict[str, Any] = {
            "type": "Action.Submit",
            "title": title,
        }

        if data:
            action["data"] = data
        if style != "default":
            action["style"] = style

        self._actions.append(action)
        return self

    def add_action_show_card(
        self,
        title: str,
        card: dict[str, Any],
        *,
        style: str = "default",
    ) -> "AdaptiveCardBuilder":
        """Add an action that shows a nested card.

        Args:
            title: Button text.
            card: Nested Adaptive Card definition.
            style: Button style.

        Returns:
            Self for method chaining.
        """
        action: dict[str, Any] = {
            "type": "Action.ShowCard",
            "title": title,
            "card": card,
        }

        if style != "default":
            action["style"] = style

        self._actions.append(action)
        return self

    def add_action_toggle_visibility(
        self,
        title: str,
        target_elements: list[str | dict[str, Any]],
        *,
        style: str = "default",
    ) -> "AdaptiveCardBuilder":
        """Add an action that toggles element visibility.

        Args:
            title: Button text.
            target_elements: IDs or targets to toggle.
            style: Button style.

        Returns:
            Self for method chaining.
        """
        action: dict[str, Any] = {
            "type": "Action.ToggleVisibility",
            "title": title,
            "targetElements": target_elements,
        }

        if style != "default":
            action["style"] = style

        self._actions.append(action)
        return self

    # -------------------------------------------------------------------------
    # Raw Element Addition
    # -------------------------------------------------------------------------

    def add_element(self, element: dict[str, Any]) -> "AdaptiveCardBuilder":
        """Add a raw element to the card body.

        Args:
            element: Element definition dictionary.

        Returns:
            Self for method chaining.
        """
        self._body.append(element)
        return self

    def add_action(self, action: dict[str, Any]) -> "AdaptiveCardBuilder":
        """Add a raw action to the card.

        Args:
            action: Action definition dictionary.

        Returns:
            Self for method chaining.
        """
        self._actions.append(action)
        return self

    # -------------------------------------------------------------------------
    # Build
    # -------------------------------------------------------------------------

    def build(self) -> dict[str, Any]:
        """Build the final Adaptive Card.

        Returns:
            Complete Adaptive Card as a dictionary.
        """
        card: dict[str, Any] = {
            "type": "AdaptiveCard",
            "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
            "version": self._version,
            "body": self._body,
        }

        if self._actions:
            card["actions"] = self._actions

        if self._fallback_text:
            card["fallbackText"] = self._fallback_text

        if self._speak:
            card["speak"] = self._speak

        if self._min_height:
            card["minHeight"] = self._min_height

        if self._vertical_content_alignment:
            card["verticalContentAlignment"] = self._vertical_content_alignment

        if self._ms_teams:
            card["msteams"] = self._ms_teams

        return card

    def build_message_card(self) -> dict[str, Any]:
        """Build a complete Teams message containing the Adaptive Card.

        Returns:
            Teams message payload with the Adaptive Card.
        """
        return {
            "type": "message",
            "attachments": [
                {
                    "contentType": "application/vnd.microsoft.card.adaptive",
                    "contentUrl": None,
                    "content": self.build(),
                }
            ],
        }


# =============================================================================
# Message Template System
# =============================================================================


class MessageTemplate(ABC):
    """Abstract base class for message templates."""

    @abstractmethod
    def render(
        self,
        checkpoint_result: "CheckpointResult",
        config: "TeamsConfig",
    ) -> dict[str, Any]:
        """Render the template with the given data.

        Args:
            checkpoint_result: Validation result data.
            config: Teams notification configuration.

        Returns:
            Complete Teams message payload.
        """
        pass


class DefaultTemplate(MessageTemplate):
    """Default message template with full details."""

    # Status configuration
    STATUS_CONFIG: dict[str, dict[str, str]] = {
        "success": {
            "color": "good",
            "emoji": "✅",
            "title": "Validation Passed",
            "accent_color": "#28a745",
        },
        "failure": {
            "color": "attention",
            "emoji": "❌",
            "title": "Validation Failed",
            "accent_color": "#dc3545",
        },
        "error": {
            "color": "attention",
            "emoji": "⚠️",
            "title": "Validation Error",
            "accent_color": "#dc3545",
        },
        "warning": {
            "color": "warning",
            "emoji": "⚠️",
            "title": "Validation Warning",
            "accent_color": "#ffc107",
        },
    }

    def render(
        self,
        checkpoint_result: "CheckpointResult",
        config: "TeamsConfig",
    ) -> dict[str, Any]:
        """Render the default template."""
        status = checkpoint_result.status.value
        status_config = self.STATUS_CONFIG.get(
            status, {"color": "default", "emoji": "❓", "title": "Unknown", "accent_color": "#6c757d"}
        )

        validation = checkpoint_result.validation_result
        stats = validation.statistics if validation else None

        builder = AdaptiveCardBuilder(config.card_version)

        if config.full_width:
            builder.enable_full_width()

        # Build mentions string
        mentions_text = ""
        mentions_entities: list[dict[str, Any]] = []
        if status in ("failure", "error") and config.mention_on_failure:
            for mention in config.mention_on_failure:
                mention_id = mention.get("id", mention) if isinstance(mention, dict) else mention
                mention_name = mention.get("name", mention_id) if isinstance(mention, dict) else mention_id
                mentions_text += f"<at>{mention_name}</at> "
                mentions_entities.append({
                    "type": "mention",
                    "text": f"<at>{mention_name}</at>",
                    "mentioned": {
                        "id": mention_id,
                        "name": mention_name,
                    },
                })

        # Header with status
        header_text = f"{status_config['emoji']} **{status_config['title']}**"
        if mentions_text:
            header_text = f"{mentions_text.strip()} {header_text}"

        builder.add_text_block(
            header_text,
            weight=TextWeight.BOLDER,
            size=TextSize.LARGE,
            color=TextColor.ATTENTION if status in ("failure", "error") else TextColor.DEFAULT,
        )

        # Checkpoint name
        builder.add_text_block(
            f"Checkpoint: **{checkpoint_result.checkpoint_name}**",
            size=TextSize.MEDIUM,
            spacing="small",
        )

        # Facts section
        facts: list[tuple[str, str]] = [
            ("Data Asset", checkpoint_result.data_asset or "N/A"),
            ("Run ID", checkpoint_result.run_id[:12] + "..." if len(checkpoint_result.run_id) > 12 else checkpoint_result.run_id),
            ("Run Time", checkpoint_result.run_time.strftime("%Y-%m-%d %H:%M:%S")),
            ("Duration", f"{checkpoint_result.duration_ms:.1f}ms"),
        ]

        if stats:
            facts.extend([
                ("Total Issues", str(stats.total_issues)),
                ("Pass Rate", f"{stats.pass_rate * 100:.1f}%"),
            ])

        builder.add_fact_set(facts, spacing="medium", separator=True)

        # Issue breakdown if there are issues
        if stats and stats.total_issues > 0 and config.include_details:
            breakdown_text = (
                f"🔴 Critical: {stats.critical_issues} | "
                f"🟠 High: {stats.high_issues} | "
                f"🟡 Medium: {stats.medium_issues} | "
                f"🔵 Low: {stats.low_issues}"
            )
            builder.add_text_block(
                breakdown_text,
                size=TextSize.SMALL,
                is_subtle=True,
                spacing="small",
            )

        # Action buttons
        if config.include_actions:
            if config.dashboard_url:
                builder.add_action_open_url(
                    "View Dashboard",
                    config.dashboard_url.format(
                        run_id=checkpoint_result.run_id,
                        checkpoint=checkpoint_result.checkpoint_name,
                    ),
                    style="positive",
                )

            if config.details_url:
                builder.add_action_open_url(
                    "View Details",
                    config.details_url.format(
                        run_id=checkpoint_result.run_id,
                        checkpoint=checkpoint_result.checkpoint_name,
                    ),
                )

        # Build the card
        card = builder.build()

        # Add mentions to msteams section if any
        if mentions_entities:
            card.setdefault("msteams", {})["entities"] = mentions_entities

        return {
            "type": "message",
            "attachments": [
                {
                    "contentType": "application/vnd.microsoft.card.adaptive",
                    "contentUrl": None,
                    "content": card,
                }
            ],
        }


class MinimalTemplate(MessageTemplate):
    """Minimal template with just status and key info."""

    def render(
        self,
        checkpoint_result: "CheckpointResult",
        config: "TeamsConfig",
    ) -> dict[str, Any]:
        """Render minimal template."""
        status = checkpoint_result.status.value
        validation = checkpoint_result.validation_result
        stats = validation.statistics if validation else None

        status_emoji = {"success": "✅", "failure": "❌", "error": "⚠️", "warning": "⚠️"}.get(status, "❓")
        issues_text = f" - {stats.total_issues} issues" if stats and stats.total_issues > 0 else ""

        builder = AdaptiveCardBuilder(config.card_version)
        builder.add_text_block(
            f"{status_emoji} **{checkpoint_result.checkpoint_name}** ({status.upper()}){issues_text}",
            wrap=True,
        )

        if stats:
            builder.add_text_block(
                f"Pass rate: {stats.pass_rate * 100:.1f}%",
                size=TextSize.SMALL,
                is_subtle=True,
            )

        return builder.build_message_card()


class DetailedTemplate(MessageTemplate):
    """Detailed template with expandable sections."""

    def render(
        self,
        checkpoint_result: "CheckpointResult",
        config: "TeamsConfig",
    ) -> dict[str, Any]:
        """Render detailed template with expandable sections."""
        default = DefaultTemplate()
        base_message = default.render(checkpoint_result, config)

        # Add additional details card for show/hide
        validation = checkpoint_result.validation_result
        if validation and validation.results and config.include_details:
            # Build details card
            details_builder = AdaptiveCardBuilder()

            # Show first N issue details
            max_issues = min(5, len(validation.results))
            for i, result in enumerate(validation.results[:max_issues]):
                issue_text = f"**{result.validator_name}** on `{result.column or 'N/A'}`"
                if hasattr(result, "message"):
                    issue_text += f": {result.message}"

                details_builder.add_text_block(
                    issue_text,
                    size=TextSize.SMALL,
                    wrap=True,
                    spacing="small" if i > 0 else "none",
                )

            if len(validation.results) > max_issues:
                details_builder.add_text_block(
                    f"... and {len(validation.results) - max_issues} more issues",
                    size=TextSize.SMALL,
                    is_subtle=True,
                )

            # Add show card action to main card
            content = base_message["attachments"][0]["content"]
            content.setdefault("actions", []).append({
                "type": "Action.ShowCard",
                "title": "Show Issue Details",
                "card": details_builder.build(),
            })

        return base_message


class CompactTemplate(MessageTemplate):
    """Compact template for high-volume notifications."""

    def render(
        self,
        checkpoint_result: "CheckpointResult",
        config: "TeamsConfig",
    ) -> dict[str, Any]:
        """Render compact template."""
        status = checkpoint_result.status.value
        validation = checkpoint_result.validation_result
        stats = validation.statistics if validation else None

        status_emoji = {"success": "✅", "failure": "❌", "error": "⚠️", "warning": "⚠️"}.get(status, "❓")

        columns = [
            {
                "type": "Column",
                "width": "auto",
                "items": [
                    {"type": "TextBlock", "text": status_emoji, "size": "large"},
                ],
            },
            {
                "type": "Column",
                "width": "stretch",
                "items": [
                    {
                        "type": "TextBlock",
                        "text": f"**{checkpoint_result.checkpoint_name}**",
                        "weight": "bolder",
                    },
                    {
                        "type": "TextBlock",
                        "text": f"{checkpoint_result.data_asset or 'N/A'} | {status.upper()}",
                        "size": "small",
                        "isSubtle": True,
                        "spacing": "none",
                    },
                ],
            },
        ]

        if stats:
            columns.append({
                "type": "Column",
                "width": "auto",
                "items": [
                    {
                        "type": "TextBlock",
                        "text": f"{stats.total_issues}",
                        "size": "extraLarge",
                        "weight": "bolder",
                        "color": "attention" if stats.total_issues > 0 else "good",
                    },
                    {
                        "type": "TextBlock",
                        "text": "issues",
                        "size": "small",
                        "isSubtle": True,
                        "spacing": "none",
                    },
                ],
                "verticalContentAlignment": "center",
            })

        builder = AdaptiveCardBuilder(config.card_version)
        builder.add_column_set(columns)

        return builder.build_message_card()


# Template registry
_TEMPLATE_REGISTRY: dict[str | MessageTheme, type[MessageTemplate]] = {
    MessageTheme.DEFAULT: DefaultTemplate,
    MessageTheme.MINIMAL: MinimalTemplate,
    MessageTheme.DETAILED: DetailedTemplate,
    MessageTheme.COMPACT: CompactTemplate,
    "default": DefaultTemplate,
    "minimal": MinimalTemplate,
    "detailed": DetailedTemplate,
    "compact": CompactTemplate,
}


def get_template(theme: str | MessageTheme) -> MessageTemplate:
    """Get a template instance by theme name.

    Args:
        theme: Theme name or MessageTheme enum.

    Returns:
        Template instance.

    Raises:
        ValueError: If theme is not found.
    """
    template_class = _TEMPLATE_REGISTRY.get(theme)
    if template_class is None:
        raise ValueError(f"Unknown theme: {theme}. Available: {list(_TEMPLATE_REGISTRY.keys())}")
    return template_class()


def register_template(name: str, template_class: type[MessageTemplate]) -> None:
    """Register a custom template.

    Args:
        name: Template name.
        template_class: Template class to register.
    """
    _TEMPLATE_REGISTRY[name] = template_class


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class TeamsConfig(ActionConfig):
    """Configuration for Microsoft Teams notification action.

    Attributes:
        webhook_url: Teams Incoming Webhook URL.
        connector_type: Type of connector being used.
        theme: Message template theme to use.
        card_version: Adaptive Card version.
        full_width: Enable full-width card display.
        include_details: Include detailed statistics.
        include_actions: Include action buttons.
        mention_on_failure: Users to @mention on failure.
        dashboard_url: URL template for dashboard link.
        details_url: URL template for details link.
        custom_template: Custom MessageTemplate instance.
        custom_payload: Completely custom payload (overrides everything).
        proxy: HTTP proxy URL.
        verify_ssl: Whether to verify SSL certificates.
        thread_id: Thread ID for reply (conversation continuity).
    """

    webhook_url: str = ""
    connector_type: TeamsConnectorType | str = TeamsConnectorType.INCOMING_WEBHOOK
    theme: MessageTheme | str = MessageTheme.DEFAULT
    card_version: AdaptiveCardVersion | str = AdaptiveCardVersion.V1_4
    full_width: bool = True
    include_details: bool = True
    include_actions: bool = True
    mention_on_failure: list[str | dict[str, str]] = field(default_factory=list)
    dashboard_url: str | None = None
    details_url: str | None = None
    custom_template: MessageTemplate | None = None
    custom_payload: dict[str, Any] | None = None
    proxy: str | None = None
    verify_ssl: bool = True
    thread_id: str | None = None
    notify_on: NotifyCondition | str = NotifyCondition.FAILURE

    def __post_init__(self) -> None:
        """Convert string enums to proper types."""
        super().__post_init__()

        if isinstance(self.connector_type, str):
            self.connector_type = TeamsConnectorType(self.connector_type)
        if isinstance(self.theme, str) and self.theme in MessageTheme.__members__.values():
            self.theme = MessageTheme(self.theme)
        if isinstance(self.card_version, str):
            self.card_version = AdaptiveCardVersion(self.card_version)


# =============================================================================
# HTTP Client Abstraction
# =============================================================================


class TeamsHTTPClient:
    """HTTP client for Teams API requests.

    Abstracts HTTP operations for easier testing and future enhancements.
    """

    def __init__(
        self,
        timeout: int = 30,
        proxy: str | None = None,
        verify_ssl: bool = True,
    ) -> None:
        """Initialize the HTTP client.

        Args:
            timeout: Request timeout in seconds.
            proxy: Proxy URL.
            verify_ssl: Whether to verify SSL.
        """
        self._timeout = timeout
        self._proxy = proxy
        self._verify_ssl = verify_ssl

    def post(
        self,
        url: str,
        payload: dict[str, Any],
        headers: dict[str, str] | None = None,
    ) -> tuple[int, str]:
        """Send a POST request.

        Args:
            url: Request URL.
            payload: JSON payload.
            headers: Optional headers.

        Returns:
            Tuple of (status_code, response_body).

        Raises:
            Exception: On request failure.
        """
        import urllib.error
        import urllib.request
        import ssl

        default_headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if headers:
            default_headers.update(headers)

        data = json.dumps(payload).encode("utf-8")

        request = urllib.request.Request(
            url,
            data=data,
            headers=default_headers,
            method="POST",
        )

        # Handle proxy
        if self._proxy:
            proxy_handler = urllib.request.ProxyHandler({
                "http": self._proxy,
                "https": self._proxy,
            })
            opener = urllib.request.build_opener(proxy_handler)
        else:
            opener = urllib.request.build_opener()

        # Handle SSL verification
        context: ssl.SSLContext | None = None
        if not self._verify_ssl:
            context = ssl.create_default_context()
            context.check_hostname = False
            context.verify_mode = ssl.CERT_NONE

        try:
            with opener.open(request, timeout=self._timeout, context=context) as response:
                status_code = response.getcode()
                body = response.read().decode("utf-8")
                return status_code, body
        except urllib.error.HTTPError as e:
            return e.code, e.read().decode("utf-8") if e.fp else str(e)


# =============================================================================
# Main Action Class
# =============================================================================


class TeamsNotification(BaseAction[TeamsConfig]):
    """Action to send Microsoft Teams notifications.

    Sends formatted Adaptive Card messages to Teams channels via
    Incoming Webhooks with rich formatting and interactive elements.

    Features:
        - Rich Adaptive Card formatting
        - Multiple message themes (default, minimal, detailed, compact)
        - @mention support
        - Action buttons for navigation
        - Thread/reply support
        - Custom template support
        - Power Automate/Logic Apps connector support

    Example:
        >>> action = TeamsNotification(
        ...     webhook_url="https://outlook.office.com/webhook/...",
        ...     notify_on="failure",
        ...     theme="detailed",
        ...     mention_on_failure=["user@example.com"],
        ...     dashboard_url="https://dashboard.example.com/runs/{run_id}",
        ... )
        >>> result = action.execute(checkpoint_result)

    Custom Template Example:
        >>> class MyTemplate(MessageTemplate):
        ...     def render(self, checkpoint_result, config):
        ...         builder = AdaptiveCardBuilder()
        ...         builder.add_text_block("Custom message!")
        ...         return builder.build_message_card()
        >>>
        >>> action = TeamsNotification(
        ...     webhook_url="...",
        ...     custom_template=MyTemplate(),
        ... )
    """

    action_type = "teams_notification"

    # Webhook URL patterns for validation
    WEBHOOK_PATTERNS = [
        r"^https://[a-z0-9-]+\.webhook\.office\.com/",
        r"^https://outlook\.office\.com/webhook/",
        r"^https://[a-z0-9-]+\.logic\.azure\.com/",
        r"^https://prod-\d+\.[a-z]+\.logic\.azure\.com/",
    ]

    @classmethod
    def _default_config(cls) -> TeamsConfig:
        """Create default configuration."""
        return TeamsConfig()

    def _execute(self, checkpoint_result: "CheckpointResult") -> ActionResult:
        """Send Teams notification.

        Args:
            checkpoint_result: The checkpoint result to notify about.

        Returns:
            ActionResult with execution status.
        """
        config = self._config

        if not config.webhook_url:
            return ActionResult(
                action_name=self.name,
                action_type=self.action_type,
                status=ActionStatus.ERROR,
                message="No webhook URL configured",
                error="webhook_url is required",
            )

        # Build message payload
        try:
            payload = self._build_payload(checkpoint_result)
        except Exception as e:
            return ActionResult(
                action_name=self.name,
                action_type=self.action_type,
                status=ActionStatus.ERROR,
                message="Failed to build message payload",
                error=str(e),
            )

        # Send to Teams
        client = TeamsHTTPClient(
            timeout=config.timeout_seconds,
            proxy=config.proxy,
            verify_ssl=config.verify_ssl,
        )

        try:
            status_code, response_body = client.post(config.webhook_url, payload)

            # Teams returns "1" on success for webhooks
            if status_code in (200, 201, 202) or response_body == "1":
                return ActionResult(
                    action_name=self.name,
                    action_type=self.action_type,
                    status=ActionStatus.SUCCESS,
                    message="Teams notification sent",
                    details={
                        "status_code": status_code,
                        "response": response_body[:200] if response_body else None,
                        "theme": str(config.theme),
                        "connector_type": str(config.connector_type),
                    },
                )
            else:
                return ActionResult(
                    action_name=self.name,
                    action_type=self.action_type,
                    status=ActionStatus.ERROR,
                    message=f"Teams webhook returned error: {status_code}",
                    error=response_body[:500] if response_body else "Unknown error",
                )

        except Exception as e:
            return ActionResult(
                action_name=self.name,
                action_type=self.action_type,
                status=ActionStatus.ERROR,
                message="Failed to send Teams notification",
                error=str(e),
            )

    def _build_payload(self, checkpoint_result: "CheckpointResult") -> dict[str, Any]:
        """Build the Teams message payload.

        Args:
            checkpoint_result: The checkpoint result.

        Returns:
            Message payload dictionary.
        """
        config = self._config

        # Use custom payload if provided
        if config.custom_payload:
            return self._substitute_placeholders(config.custom_payload, checkpoint_result)

        # Use custom template if provided
        if config.custom_template:
            return config.custom_template.render(checkpoint_result, config)

        # Use theme-based template
        template = get_template(config.theme)
        return template.render(checkpoint_result, config)

    def _substitute_placeholders(
        self,
        payload: dict[str, Any],
        checkpoint_result: "CheckpointResult",
    ) -> dict[str, Any]:
        """Substitute placeholders in custom payload.

        Args:
            payload: Payload with placeholders.
            checkpoint_result: Data for substitution.

        Returns:
            Payload with substituted values.
        """
        validation = checkpoint_result.validation_result
        stats = validation.statistics if validation else None

        substitutions = {
            "${checkpoint}": checkpoint_result.checkpoint_name,
            "${status}": checkpoint_result.status.value,
            "${run_id}": checkpoint_result.run_id,
            "${data_asset}": checkpoint_result.data_asset or "",
            "${run_time}": checkpoint_result.run_time.isoformat(),
            "${duration_ms}": str(checkpoint_result.duration_ms),
            "${total_issues}": str(stats.total_issues) if stats else "0",
            "${critical_issues}": str(stats.critical_issues) if stats else "0",
            "${high_issues}": str(stats.high_issues) if stats else "0",
            "${medium_issues}": str(stats.medium_issues) if stats else "0",
            "${low_issues}": str(stats.low_issues) if stats else "0",
            "${pass_rate}": f"{stats.pass_rate * 100:.1f}" if stats else "100.0",
        }

        def substitute(obj: Any) -> Any:
            if isinstance(obj, str):
                for placeholder, value in substitutions.items():
                    obj = obj.replace(placeholder, value)
                return obj
            elif isinstance(obj, dict):
                return {k: substitute(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [substitute(item) for item in obj]
            return obj

        return substitute(payload)

    def validate_config(self) -> list[str]:
        """Validate configuration.

        Returns:
            List of validation error messages.
        """
        errors: list[str] = []

        if not self._config.webhook_url:
            errors.append("webhook_url is required")
        else:
            # Validate webhook URL pattern
            is_valid = any(
                re.match(pattern, self._config.webhook_url)
                for pattern in self.WEBHOOK_PATTERNS
            )
            if not is_valid:
                errors.append(
                    "webhook_url does not appear to be a valid Teams webhook URL. "
                    "Expected patterns: outlook.office.com/webhook/*, *.webhook.office.com/*, "
                    "*.logic.azure.com/*"
                )

        # Validate theme
        if isinstance(self._config.theme, str) and self._config.theme not in _TEMPLATE_REGISTRY:
            errors.append(f"Unknown theme: {self._config.theme}")

        return errors


# =============================================================================
# Convenience Functions
# =============================================================================


def create_teams_notification(
    webhook_url: str,
    *,
    notify_on: str | NotifyCondition = NotifyCondition.FAILURE,
    theme: str | MessageTheme = MessageTheme.DEFAULT,
    **kwargs: Any,
) -> TeamsNotification:
    """Create a Teams notification action with common defaults.

    Args:
        webhook_url: Teams webhook URL.
        notify_on: When to notify.
        theme: Message theme.
        **kwargs: Additional configuration options.

    Returns:
        Configured TeamsNotification instance.
    """
    return TeamsNotification(
        webhook_url=webhook_url,
        notify_on=notify_on,
        theme=theme,
        **kwargs,
    )


def create_failure_alert(
    webhook_url: str,
    mention_users: list[str] | None = None,
    dashboard_url: str | None = None,
) -> TeamsNotification:
    """Create a Teams notification for failure alerts.

    Pre-configured for failure notifications with mentions and dashboard link.

    Args:
        webhook_url: Teams webhook URL.
        mention_users: Users to @mention on failure.
        dashboard_url: Dashboard URL template.

    Returns:
        Configured TeamsNotification for failures.
    """
    return TeamsNotification(
        webhook_url=webhook_url,
        notify_on=NotifyCondition.FAILURE_OR_ERROR,
        theme=MessageTheme.DETAILED,
        mention_on_failure=mention_users or [],
        dashboard_url=dashboard_url,
        include_actions=True,
    )


def create_summary_notification(
    webhook_url: str,
    *,
    notify_on: str | NotifyCondition = NotifyCondition.ALWAYS,
) -> TeamsNotification:
    """Create a Teams notification for summary reports.

    Pre-configured for compact summary notifications.

    Args:
        webhook_url: Teams webhook URL.
        notify_on: When to notify.

    Returns:
        Configured TeamsNotification for summaries.
    """
    return TeamsNotification(
        webhook_url=webhook_url,
        notify_on=notify_on,
        theme=MessageTheme.COMPACT,
        include_details=False,
        include_actions=False,
    )
