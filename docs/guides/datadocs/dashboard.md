# Dashboard (Stage 2)

Truthound Data Docs provides a Reflex-based interactive dashboard.

## Installation

```bash
pip install truthound[dashboard]
```

## Quick Start

### CLI

```bash
# Run with profile
truthound dashboard --profile profile.json

# Custom port
truthound dashboard --profile profile.json --port 3000

# Allow external access
truthound dashboard --profile profile.json --host 0.0.0.0
```

### Python API

```python
from truthound.datadocs.dashboard import launch_dashboard

launch_dashboard(
    profile_path="profile.json",
    port=8080,
    host="localhost",
    title="My Dashboard",
    debug=False,
)
```

## DashboardConfig

A data class for dashboard configuration.

```python
from truthound.datadocs.dashboard import DashboardConfig

config = DashboardConfig(
    # Server settings
    host="localhost",
    port=8080,
    debug=False,

    # Theme
    theme="light",            # "light" or "dark"
    primary_color="blue",

    # Feature toggles
    show_raw_data=True,
    show_correlations=True,
    show_patterns=True,
    enable_export=True,

    # Data
    profile_path="profile.json",     # Profile file path
    profile_data=None,               # Or profile dictionary

    # Branding
    title="Truthound Dashboard",
    logo_url=None,
)
```

## DashboardApp

The dashboard application class.

```python
from truthound.datadocs.dashboard import DashboardApp, DashboardConfig

# Create with configuration
config = DashboardConfig(
    profile_path="profile.json",
    title="My Dashboard",
    port=8080,
)
app = DashboardApp(config)

# Load profile
app.load_profile(profile_path="profile.json")
# Or
app.load_profile(profile_data=profile_dict)

# Run server
app.run(host="localhost", port=8080, debug=False)
```

## Dashboard Structure

### Pages

The dashboard consists of 3 main pages:

#### 1. Overview

- **Metrics Grid**: Rows, Columns, Memory, Quality Score
- **Alert List**: Data quality issues

#### 2. Columns

- **Search**: Search by column name
- **Column Card Grid**: Detailed information for each column
  - Data type badge
  - Null/Unique/Distinct ratios

#### 3. Quality

- **Overall Quality Score**: Large display
- **Quality Analysis Description**

### UI Features

- **Sidebar**: Page navigation
- **Theme Toggle**: Light/dark mode switching
- **Responsive**: Mobile/tablet support

## State Management

The dashboard uses Reflex's state management.

```python
# Internal State class (for reference)
class State(rx.State):
    # Profile data
    profile_data: dict = {}
    row_count: int = 0
    column_count: int = 0
    memory_bytes: int = 0
    quality_score: float = 100.0
    columns: list = []
    correlations: list = []
    alerts: list = []

    # UI state
    sidebar_open: bool = True
    active_tab: str = "overview"
    selected_column: str = ""
    search_query: str = ""
    theme: str = "light"
    is_loading: bool = True

    # Actions
    def load_profile(self, data: dict) -> None: ...
    def toggle_sidebar(self) -> None: ...
    def set_tab(self, tab: str) -> None: ...
    def select_column(self, column: str) -> None: ...
    def set_search(self, query: str) -> None: ...
    def toggle_theme(self) -> None: ...

    # Computed properties
    @rx.var
    def filtered_columns(self) -> list: ...
    @rx.var
    def format_memory(self) -> str: ...
```

## Convenience Functions

### launch_dashboard

```python
from truthound.datadocs.dashboard import launch_dashboard

launch_dashboard(
    profile_path="profile.json",  # Or None
    profile_data=None,            # Or profile dictionary
    port=8080,
    host="localhost",
    title="Truthound Dashboard",
    debug=False,
)
```

### create_app

```python
from truthound.datadocs.dashboard import create_app, DashboardConfig

# Create with default settings
app = create_app(profile_path="profile.json")

# Create with custom settings
config = DashboardConfig(
    profile_path="profile.json",
    title="Custom Dashboard",
    theme="dark",
)
app = create_app(config=config)

# Run server
app.run()
```

## CLI Options

```bash
truthound dashboard [OPTIONS]
```

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--profile` | `-p` | None | Profile JSON file path |
| `--port` | | 8080 | Server port |
| `--host` | | "localhost" | Server host |
| `--title` | `-t` | "Truthound Dashboard" | Dashboard title |
| `--debug` | | False | Debug mode |

## Requirements

Dashboard functionality requires the Reflex package:

```bash
pip install truthound[dashboard]
```

If Reflex is not installed:

```python
from truthound.datadocs.dashboard import launch_dashboard

# ImportError: Dashboard requires Reflex.
# Install with: pip install truthound[dashboard]
```

## API Reference

### DashboardConfig

```python
@dataclass
class DashboardConfig:
    # Server settings
    host: str = "localhost"
    port: int = 8080
    debug: bool = False

    # Theme
    theme: str = "light"
    primary_color: str = "blue"

    # Feature toggles
    show_raw_data: bool = True
    show_correlations: bool = True
    show_patterns: bool = True
    enable_export: bool = True

    # Data
    profile_path: str | None = None
    profile_data: dict[str, Any] | None = None

    # Branding
    title: str = "Truthound Dashboard"
    logo_url: str | None = None
```

### DashboardApp

```python
class DashboardApp:
    def __init__(self, config: DashboardConfig | None = None) -> None:
        """Initialize dashboard application."""
        ...

    def load_profile(
        self,
        profile_path: str | Path | None = None,
        profile_data: dict | None = None,
    ) -> None:
        """Load profile data."""
        ...

    def run(
        self,
        host: str | None = None,
        port: int | None = None,
        debug: bool | None = None,
    ) -> None:
        """Run dashboard server."""
        ...
```

### launch_dashboard

```python
def launch_dashboard(
    profile_path: str | Path | None = None,
    profile_data: dict | None = None,
    port: int = 8080,
    host: str = "localhost",
    title: str = "Truthound Dashboard",
    debug: bool = False,
) -> None:
    """Launch interactive dashboard."""
    ...
```

### create_app

```python
def create_app(
    profile_path: str | Path | None = None,
    profile_data: dict | None = None,
    config: DashboardConfig | None = None,
) -> DashboardApp:
    """Create dashboard application instance."""
    ...
```

## See Also

- [HTML Reports](html-reports.md) - Static HTML reports
- [Themes](themes.md) - Theme customization
- [truthound-dashboard](https://github.com/seadonggyun4/truthound-dashboard) - Separate dashboard repository
