# Realtime Checkpoint Commands

Manage streaming validation checkpoints.

## Overview

Checkpoints allow you to save and restore the state of streaming validation pipelines. This is useful for:

- Resuming validation after a failure
- Auditing validation history
- Debugging streaming pipelines

## Commands

| Command | Description |
|---------|-------------|
| [`list`](list.md) | List available streaming validation checkpoints |
| [`show`](show.md) | Show details of a specific checkpoint |
| [`delete`](delete.md) | Delete a checkpoint |

## Quick Reference

```bash
# List all checkpoints
truthound realtime checkpoint list

# List checkpoints in a specific directory
truthound realtime checkpoint list --dir ./my_checkpoints

# Show checkpoint details
truthound realtime checkpoint show abc12345

# Delete a checkpoint
truthound realtime checkpoint delete abc12345

# Delete without confirmation
truthound realtime checkpoint delete abc12345 --force
```

## Checkpoint Storage

By default, checkpoints are stored in `./checkpoints/` directory. Each checkpoint is saved as a JSON file with the naming convention `checkpoint_{id}.json`.

## Related Commands

- [`realtime validate`](../validate.md) - Validate streaming data
- [`realtime monitor`](../monitor.md) - Monitor streaming pipeline
