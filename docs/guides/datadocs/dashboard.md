# Optional Data Docs profile viewer

Core includes a Reflex-based local viewer over profile data. HTML/PDF report
generation does not require this UI. Install Truthound's `dashboard` extra to
provide its optional Reflex dependency.

```python
from truthound.datadocs import launch_dashboard

launch_dashboard(
    profile_path="profile.json",
    host="localhost",
    port=8080,
    title="Local profile inspection",
)
```

The launcher accepts `profile_path`, `profile_data`, `port`, `host`, `title` and
`debug`; file input takes precedence when a path is provided. Its UI includes
profile, column, pattern and correlation views and a profile JSON download.

To construct a wrapper without immediately starting the server:

```python
from truthound.datadocs import create_dashboard_app, get_dashboard_config

Config = get_dashboard_config()
viewer = create_dashboard_app(config=Config(
    host="localhost", port=8080, profile_path="profile.json",
))
# viewer.run() starts the optional server.
```

This returns a `DashboardApp` wrapper, not report HTML. In development sources,
`run()` prepares a private temporary project and runs its compiled frontend and
native backend on one HTTP port. Overrides for host, port and debug are forwarded;
debug selects server logging. Keep the UI local; it has no authentication boundary.

## Explicit preparation and managed lifecycle in development sources

Install `pip install '.[dashboard]'` from the checkout and provide Node/npm.
The supported Reflex range is `>=0.8.25,<0.9`; local runtime checks use 0.8.25.
The extra explicitly includes SQLAlchemy's asyncio support and the Uvicorn ASGI
server, including WebSocket support. Node 20.20.2/npm 10.8.2 were used locally.
Preparation installs the frontend packages through npm and builds the native app;
this is separate from server start and does not deploy remotely. Reflex telemetry
is disabled in the generated project. Configuration is JSON data, including titles.

```python
from pathlib import Path
from truthound.datadocs.dashboard import DashboardConfig, ReflexDashboardServer

# Parent directory must exist; preparation refuses an existing project directory.
server = ReflexDashboardServer(Path("private-viewer-project"), DashboardConfig(
    host="127.0.0.1", port=8080, profile_path="profile.json",
))
server.prepare()  # explicit build; default timeout 300 s, maximum 600 s
try:
    server.start()  # default timeout 10 s, maximum 30 s; no dependency installation
    print(server.status())  # running, ready, url, exit_code
    server.wait()
finally:
    server.close()
```

Relative operator-selected profile paths are resolved when constructing the server;
the child runs inside its private project. `stop()` permits restart using the
same prepared configuration. Configuration changes are refused and require a
fresh project. A changed profile file is also refused. The owner closes its process group; hard owner death also ends the
native child. Keep the prepared project private and out of repositories/packages:
it contains profile values in configuration and compiled assets. UI flags
`show_raw_data`, `show_correlations`, `show_patterns` and `enable_export` control
display/download controls; they do not protect values from an authorized browser.
Local tests cover real page compilation, profile state, HTTP readiness, restart,
MCP release and owner death. Remote deployment remains unverified.

Use `HTMLReportBuilder`, `generate_html_report` and `generate_validation_report`
for independent report artifacts. See [HTML reports](html-reports.md) and the
[Core interface overview](../../dashboard/index.md) for the separate Streamlit UI.
