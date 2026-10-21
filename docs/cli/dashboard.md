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
| `--debug` | `false` | Stored in launcher configuration |

The command calls `truthound.datadocs.launch_dashboard`. Missing optional
dependencies cause an error. The current launcher stores `debug` but does not
pass it into the underlying server call, so it must not be described as an
effective server debug switch.

Use the separate `truthound docs generate` command for static output. See the
[Python guide](../guides/datadocs/dashboard.md) for programmatic access.
Keep the viewer on localhost and validate startup using non-sensitive input.
The command signature is source-verified; current GUI/runtime compatibility is
not established by this documentation change.
