# Optional interactive Core interfaces

Truthound includes optional local interfaces for exploring profile results.
They are not required for HTML/PDF report generation or the MCP adapter.

| Interface | Dependency | Entry point |
| --- | --- | --- |
| Data Docs profile viewer | Reflex, installed by the `dashboard` extra | `truthound dashboard` |
| Interactive profiler | Development `dashboard-streamlit` extra or separate Streamlit | `truthound.profiler.dashboard.run_dashboard` |

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
The development launcher forwards all dataclass settings as private JSON to a
static app entry point; titles are never interpolated into Python code. A field existing in a
configuration class does not prove runtime enforcement.

## Limits

Local development tests verify Streamlit startup, cleanup and title rendering,
and Reflex 0.8.25 page compilation, real HTTP lifecycle and owner cleanup.
Remote deployment compatibility remains unverified.
Keep the server bound to localhost. Displayed values and uploaded files may be
sensitive; a `require_auth` configuration field does not establish an implemented
authentication boundary. Static report generation remains independent.

## Managed process in development sources

From the development checkout, use `pip install '.[dashboard-streamlit]'`. `truthound.profiler.dashboard.server.DashboardServer` exposes `start()`,
`status()`, `stop()` and `close()` over a real Streamlit process. Readiness and process
liveness are separate, and the child exits when its owner disconnects. Startup
defaults to 10 seconds (maximum 30). Configuration is JSON data for a static entry
point; native UI authentication guarantees are unchanged. Approved MCP controls are
in the [guide](../guides/mcp.md#native-loopback-dashboard-lifecycle). This feature
is not yet included in published 3.2.0. Local actual-server/rendering regressions
cover this path. `truthound.datadocs.dashboard.ReflexDashboardServer` also manages
a private prepared Reflex project on one HTTP port. Its explicit `prepare()` needs
Node/npm and the development `dashboard` extra; starting the prepared server does
not install dependencies. See the [preparation guide](../guides/datadocs/dashboard.md).
