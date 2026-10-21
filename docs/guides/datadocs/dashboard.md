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
`debug`; file input takes precedence when a path is provided. Its UI has profile,
column, pattern and correlation views. Source presence does not prove that all
controls operate correctly with the current Reflex version.

To construct a wrapper without immediately starting the server:

```python
from truthound.datadocs import create_dashboard_app, get_dashboard_config

Config = get_dashboard_config()
viewer = create_dashboard_app(config=Config(
    host="localhost", port=8080, profile_path="profile.json",
))
# viewer.run() starts the optional server.
```

This returns a `DashboardApp` wrapper, not report HTML. The current server call
does not forward `debug`. Configuration flags do not by themselves prove
implemented authentication/export controls. Do not expose the UI publicly or
assume authentication is enforced. GUI startup and dependency compatibility
were not tested by this documentation-only change.

Use `HTMLReportBuilder`, `generate_html_report` and `generate_validation_report`
for independent report artifacts. See [HTML reports](html-reports.md) and the
[Core interface overview](../../dashboard/index.md) for the separate Streamlit UI.
