# Optional interactive Core interfaces

Truthound includes optional local interfaces for exploring profile results.
They are not required for HTML/PDF report generation or the MCP adapter.

| Interface | Dependency | Entry point |
| --- | --- | --- |
| Data Docs profile viewer | Reflex, installed by the `dashboard` extra | `truthound dashboard` |
| Interactive profiler | Separately installed Streamlit | `truthound.profiler.dashboard.run_dashboard` |

## Data Docs profile viewer

```bash
pip install "truthound[dashboard]==3.2.0"
```

See the [CLI reference](../cli/dashboard.md) and
[Python guide](../guides/datadocs/dashboard.md). Its default address is
`localhost` and its default port is 8080.

## Interactive profiler

The `dashboard` extra installs Reflex, not Streamlit. For this separate UI:

```bash
pip install "truthound==3.2.0" streamlit
```

```python
from truthound.profiler.dashboard import DashboardConfig, run_dashboard

run_dashboard(DashboardConfig(host="localhost", port=8501))
```

The UI includes file selection/upload, profile views and export controls.
The launcher currently forwards host, port, theme and title to its generated
app; other configuration fields are not all forwarded. A field existing in a
configuration class does not prove runtime enforcement.

## Limits

This documentation change does not verify GUI startup or compatibility with
current Reflex/Streamlit versions. Validate the target environment before use.
Keep the server bound to localhost. Displayed values and uploaded files may be
sensitive; a `require_auth` configuration field does not establish an implemented
authentication boundary. Static report generation remains independent.
