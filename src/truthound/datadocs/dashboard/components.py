"""Reusable UI components for the Reflex dashboard.

This module provides styled components for building the dashboard UI.
All components are designed to work with Reflex's reactive system.
"""

from __future__ import annotations

from typing import Any, Callable

# Note: These are placeholder implementations that will work with Reflex
# when the dashboard extra is installed. The actual Reflex imports and
# component implementations are conditional.


def _check_reflex_installed() -> None:
    """Check if Reflex is installed."""
    try:
        import reflex  # noqa: F401
    except ImportError:
        raise ImportError(
            "Dashboard requires Reflex. "
            "Install with: pip install truthound[dashboard]"
        )


# =============================================================================
# Component Builders (Reflex-compatible)
# =============================================================================


def header(
    title: str = "Truthound Dashboard",
    subtitle: str = "",
    on_toggle_sidebar: Callable | None = None,
    on_toggle_theme: Callable | None = None,
) -> Any:
    """Create a dashboard header component.

    Args:
        title: Dashboard title
        subtitle: Dashboard subtitle
        on_toggle_sidebar: Callback for sidebar toggle
        on_toggle_theme: Callback for theme toggle

    Returns:
        Reflex component
    """
    _check_reflex_installed()
    import reflex as rx

    return rx.box(
        rx.hstack(
            rx.button(
                rx.icon("menu"),
                on_click=on_toggle_sidebar,
                variant="ghost",
                size="sm",
            ),
            rx.vstack(
                rx.heading(title, size="lg"),
                rx.text(subtitle, color="gray.500", size="sm") if subtitle else rx.fragment(),
                align_items="start",
                spacing="0",
            ),
            rx.spacer(),
            rx.button(
                rx.icon("sun"),
                on_click=on_toggle_theme,
                variant="ghost",
                size="sm",
            ),
            width="100%",
            padding="4",
            border_bottom="1px solid",
            border_color="gray.200",
        ),
    )


def sidebar(
    items: list[dict[str, Any]],
    active_item: str = "",
    on_select: Callable | None = None,
) -> Any:
    """Create a navigation sidebar component.

    Args:
        items: List of navigation items with 'id', 'label', 'icon'
        active_item: Currently active item ID
        on_select: Callback when item is selected

    Returns:
        Reflex component
    """
    _check_reflex_installed()
    import reflex as rx

    nav_items = []
    for item in items:
        is_active = item["id"] == active_item
        nav_items.append(
            rx.button(
                rx.icon(item.get("icon", "circle")),
                rx.text(item["label"]),
                width="100%",
                justify_content="start",
                variant="ghost" if not is_active else "solid",
                color_scheme="blue" if is_active else "gray",
                on_click=lambda i=item["id"]: on_select(i) if on_select else None,
            )
        )

    return rx.box(
        rx.vstack(
            *nav_items,
            spacing="2",
            padding="4",
            width="250px",
            border_right="1px solid",
            border_color="gray.200",
            height="100vh",
        ),
    )


def metric_card(
    label: str,
    value: str | int | float,
    icon: str = "activity",
    change: float | None = None,
    color_scheme: str = "blue",
) -> Any:
    """Create a metric display card.

    Args:
        label: Metric label
        value: Metric value
        icon: Icon name
        change: Optional percent change
        color_scheme: Color scheme for the card

    Returns:
        Reflex component
    """
    _check_reflex_installed()
    import reflex as rx

    formatted_value = value
    if isinstance(value, float):
        if value < 1:
            formatted_value = f"{value:.1%}"
        else:
            formatted_value = f"{value:,.2f}"
    elif isinstance(value, int):
        formatted_value = f"{value:,}"

    change_element = rx.fragment()
    if change is not None:
        change_color = "green.500" if change >= 0 else "red.500"
        change_icon = "trending_up" if change >= 0 else "trending_down"
        change_element = rx.hstack(
            rx.icon(change_icon, size="sm"),
            rx.text(f"{change:+.1%}", size="sm"),
            color=change_color,
        )

    return rx.box(
        rx.vstack(
            rx.hstack(
                rx.icon(icon, color=f"{color_scheme}.500"),
                rx.text(label, color="gray.500", size="sm"),
            ),
            rx.heading(str(formatted_value), size="2xl"),
            change_element,
            align_items="start",
            spacing="1",
        ),
        padding="4",
        border_radius="lg",
        border="1px solid",
        border_color="gray.200",
        bg="white",
        _hover={"shadow": "md"},
    )


def chart_container(
    title: str,
    chart_element: Any,
    subtitle: str = "",
    height: str = "300px",
) -> Any:
    """Create a container for charts.

    Args:
        title: Chart title
        chart_element: The chart component
        subtitle: Optional subtitle
        height: Chart height

    Returns:
        Reflex component
    """
    _check_reflex_installed()
    import reflex as rx

    return rx.box(
        rx.vstack(
            rx.heading(title, size="md"),
            rx.text(subtitle, color="gray.500", size="sm") if subtitle else rx.fragment(),
            rx.box(
                chart_element,
                height=height,
                width="100%",
            ),
            width="100%",
            align_items="start",
            spacing="2",
        ),
        padding="4",
        border_radius="lg",
        border="1px solid",
        border_color="gray.200",
        bg="white",
    )


def column_card(
    name: str,
    dtype: str,
    inferred_type: str,
    null_ratio: float,
    unique_ratio: float,
    distinct_count: int,
    on_click: Callable | None = None,
    distribution: dict[str, Any] | None = None,
    patterns: list[dict[str, Any]] | None = None,
) -> Any:
    """Create a column information card.

    Args:
        name: Column name
        dtype: Physical data type
        inferred_type: Inferred semantic type
        null_ratio: Null value ratio
        unique_ratio: Unique value ratio
        distinct_count: Count of distinct values
        on_click: Click handler
        distribution: Optional distribution stats
        patterns: Optional detected patterns

    Returns:
        Reflex component
    """
    _check_reflex_installed()
    import reflex as rx

    # Determine quality color
    quality_color = "green" if null_ratio < 0.05 else "yellow" if null_ratio < 0.2 else "red"

    # Type badge color
    type_colors = {
        "integer": "blue",
        "float": "blue",
        "string": "yellow",
        "datetime": "green",
        "boolean": "purple",
        "email": "pink",
        "url": "cyan",
        "phone": "orange",
    }
    type_color = type_colors.get(inferred_type.lower(), "gray")

    stats_items = []
    if distribution:
        if distribution.get("mean") is not None:
            stats_items.append(rx.text(f"Mean: {distribution['mean']:.2f}", size="xs"))
        if distribution.get("min") is not None:
            stats_items.append(rx.text(f"Range: [{distribution['min']}, {distribution['max']}]", size="xs"))

    pattern_badges = []
    if patterns:
        for p in patterns[:3]:
            pattern_badges.append(
                rx.badge(p.get("pattern", ""), color_scheme="purple", size="sm")
            )

    return rx.box(
        rx.vstack(
            rx.hstack(
                rx.heading(name, size="md"),
                rx.badge(inferred_type, color_scheme=type_color),
                width="100%",
                justify_content="space-between",
            ),
            rx.text(f"Type: {dtype}", color="gray.500", size="sm"),
            rx.hstack(
                rx.vstack(
                    rx.text("Null", size="xs", color="gray.500"),
                    rx.text(f"{null_ratio:.1%}", font_weight="bold", color=f"{quality_color}.500"),
                    spacing="0",
                ),
                rx.vstack(
                    rx.text("Unique", size="xs", color="gray.500"),
                    rx.text(f"{unique_ratio:.1%}", font_weight="bold"),
                    spacing="0",
                ),
                rx.vstack(
                    rx.text("Distinct", size="xs", color="gray.500"),
                    rx.text(f"{distinct_count:,}", font_weight="bold"),
                    spacing="0",
                ),
                width="100%",
                justify_content="space-around",
                padding_y="2",
            ),
            rx.hstack(*stats_items, spacing="2") if stats_items else rx.fragment(),
            rx.hstack(*pattern_badges, spacing="1") if pattern_badges else rx.fragment(),
            width="100%",
            align_items="start",
            spacing="2",
        ),
        padding="4",
        border_radius="lg",
        border="1px solid",
        border_color="gray.200",
        bg="white",
        cursor="pointer",
        _hover={"shadow": "md", "border_color": "blue.300"},
        on_click=on_click,
    )


def data_table(
    headers: list[str],
    rows: list[list[Any]],
    on_row_click: Callable | None = None,
    striped: bool = True,
    hoverable: bool = True,
) -> Any:
    """Create a data table component.

    Args:
        headers: Table headers
        rows: Table rows
        on_row_click: Row click handler
        striped: Enable striped rows
        hoverable: Enable row hover effect

    Returns:
        Reflex component
    """
    _check_reflex_installed()
    import reflex as rx

    header_cells = [rx.table.column_header_cell(h) for h in headers]

    body_rows = []
    for i, row in enumerate(rows):
        cells = [rx.table.cell(str(cell)) for cell in row]
        body_rows.append(
            rx.table.row(
                *cells,
                on_click=lambda idx=i: on_row_click(idx) if on_row_click else None,
            )
        )

    return rx.table.root(
        rx.table.header(
            rx.table.row(*header_cells),
        ),
        rx.table.body(*body_rows),
        width="100%",
    )


def alert_banner(
    title: str,
    message: str,
    type: str = "info",
    dismissible: bool = True,
    on_dismiss: Callable | None = None,
) -> Any:
    """Create an alert banner component.

    Args:
        title: Alert title
        message: Alert message
        type: Alert type (info, warning, error, success)
        dismissible: Allow dismissing
        on_dismiss: Dismiss callback

    Returns:
        Reflex component
    """
    _check_reflex_installed()
    import reflex as rx

    color_schemes = {
        "info": "blue",
        "warning": "yellow",
        "error": "red",
        "success": "green",
    }
    color_scheme = color_schemes.get(type, "gray")

    icons = {
        "info": "info",
        "warning": "alert_triangle",
        "error": "x_circle",
        "success": "check_circle",
    }
    icon = icons.get(type, "info")

    dismiss_button = rx.fragment()
    if dismissible:
        dismiss_button = rx.button(
            rx.icon("x"),
            on_click=on_dismiss,
            variant="ghost",
            size="sm",
        )

    return rx.box(
        rx.hstack(
            rx.icon(icon, color=f"{color_scheme}.500"),
            rx.vstack(
                rx.text(title, font_weight="bold"),
                rx.text(message, size="sm", color="gray.600"),
                align_items="start",
                spacing="0",
            ),
            rx.spacer(),
            dismiss_button,
            width="100%",
        ),
        padding="4",
        border_radius="md",
        bg=f"{color_scheme}.50",
        border="1px solid",
        border_color=f"{color_scheme}.200",
    )


def search_input(
    placeholder: str = "Search...",
    value: str = "",
    on_change: Callable | None = None,
) -> Any:
    """Create a search input component.

    Args:
        placeholder: Placeholder text
        value: Current value
        on_change: Change handler

    Returns:
        Reflex component
    """
    _check_reflex_installed()
    import reflex as rx

    return rx.hstack(
        rx.icon("search", color="gray.400"),
        rx.input(
            placeholder=placeholder,
            value=value,
            on_change=on_change,
            variant="unstyled",
            flex="1",
        ),
        padding="2",
        padding_x="3",
        border="1px solid",
        border_color="gray.200",
        border_radius="md",
        _focus_within={"border_color": "blue.500"},
    )


def filter_dropdown(
    label: str,
    options: list[dict[str, str]],
    value: str | list[str] = "",
    on_change: Callable | None = None,
    multi: bool = False,
) -> Any:
    """Create a filter dropdown component.

    Args:
        label: Dropdown label
        options: List of options with 'value' and 'label'
        value: Current value(s)
        on_change: Change handler
        multi: Allow multiple selection

    Returns:
        Reflex component
    """
    _check_reflex_installed()
    import reflex as rx

    option_elements = [
        rx.option(opt["label"], value=opt["value"])
        for opt in options
    ]

    return rx.vstack(
        rx.text(label, size="sm", color="gray.500"),
        rx.select(
            *option_elements,
            value=value,
            on_change=on_change,
            width="100%",
        ),
        align_items="start",
        spacing="1",
    )


def loading_spinner(
    text: str = "Loading...",
    size: str = "md",
) -> Any:
    """Create a loading spinner component.

    Args:
        text: Loading text
        size: Spinner size

    Returns:
        Reflex component
    """
    _check_reflex_installed()
    import reflex as rx

    return rx.center(
        rx.vstack(
            rx.spinner(size=size),
            rx.text(text, color="gray.500"),
            spacing="2",
        ),
        height="200px",
    )


def empty_state(
    icon: str = "inbox",
    title: str = "No data",
    message: str = "There is no data to display",
    action_label: str | None = None,
    on_action: Callable | None = None,
) -> Any:
    """Create an empty state component.

    Args:
        icon: Icon name
        title: Title text
        message: Message text
        action_label: Optional action button label
        on_action: Action button callback

    Returns:
        Reflex component
    """
    _check_reflex_installed()
    import reflex as rx

    action_button = rx.fragment()
    if action_label and on_action:
        action_button = rx.button(action_label, on_click=on_action)

    return rx.center(
        rx.vstack(
            rx.icon(icon, size="xl", color="gray.300"),
            rx.heading(title, size="md", color="gray.500"),
            rx.text(message, color="gray.400", text_align="center"),
            action_button,
            spacing="2",
            padding="8",
        ),
        height="300px",
    )
