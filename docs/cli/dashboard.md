# Local profile viewer CLI

`truthound dashboard` launches Core's optional Reflex profile viewer. It is not
the command for static report generation or for starting the MCP server.

Install Truthound's `dashboard` extra in the environment that runs this command.

```bash
truthound dashboard --profile profile.json --host localhost --port 8080
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--profile`, `-p` | No file | Existing profile JSON input |
| `--host` | `localhost` | Requested bind address |
| `--port` | `8080` | Requested port |
| `--title`, `-t` | `Truthound Dashboard` | Page title |
| `--debug` | `false` | Development launcher selects server debug logging |

The command calls `truthound.datadocs.launch_dashboard`. Missing optional
dependencies cause an error. In development sources the launcher builds a private
temporary Reflex project using Node/npm, then starts its compiled frontend and
backend on the selected port. `debug` selects Uvicorn server logging. This fix is
not included in published 3.2.0; use `pip install '.[dashboard]'` from the checkout.

Use the separate `truthound docs generate` command for static output. See the
[Python guide](../guides/datadocs/dashboard.md) for programmatic access.
Keep the viewer on localhost and validate startup using non-sensitive input.
Local Reflex 0.8.25 compilation and native HTTP lifecycle are tested; remote
deployment remains unverified.
