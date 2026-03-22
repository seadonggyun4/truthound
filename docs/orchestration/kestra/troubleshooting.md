---
title: Kestra Troubleshooting
---

# Kestra Troubleshooting

## The Script Task Runs But No Structured Result Appears

Check whether the flow is using the package's output handlers or whether the task only
prints raw console output. Structured downstream automation should rely on emitted
outputs, not on log scraping.

## A Generated Flow Compiles But Does Not Match Team Conventions

Treat the generator as a scaffold, not the only valid shape. Generate once, then adapt
namespace, retries, trigger policy, and variables to the team's actual operating model.

## A Script Works Locally But Not In Kestra

Validate:

- the runtime image includes the required package extras
- the target URI is reachable from the execution environment
- secrets and environment variables are available in the task context

## Related Pages

- [Scripts and Flow Templates](scripts-templates.md)
- [Shared Runtime Overview](../common/index.md)
