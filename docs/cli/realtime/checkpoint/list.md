# realtime checkpoint list

List available streaming validation checkpoints.

## Synopsis

```bash
truthound realtime checkpoint list [OPTIONS]
```

## Description

Lists all available streaming validation checkpoints stored in the checkpoint directory. Displays checkpoint ID, creation time, number of batches processed, total records, and total issues.

## Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--dir`, `-d` | PATH | `./checkpoints` | Checkpoint directory |
| `--format`, `-f` | TEXT | `console` | Output format (`console`, `json`) |

## Examples

### List checkpoints in default directory

```bash
$ truthound realtime checkpoint list

Checkpoints in ./checkpoints
============================================================
ID           Created              Batches   Records    Issues
------------------------------------------------------------
abc12345     2026-01-15 10:30:00       50     50000       125
def67890     2026-01-15 11:45:00       30     30000        89
ghi11111     2026-01-15 14:20:00      100    100000       312

Total: 3 checkpoint(s)
```

### List checkpoints in custom directory

```bash
truthound realtime checkpoint list --dir ./my_checkpoints
```

### Output as JSON

```bash
$ truthound realtime checkpoint list --format json
[
  {
    "checkpoint_id": "abc12345",
    "created_at": "2026-01-15T10:30:00",
    "batch_count": 50,
    "total_records": 50000,
    "total_issues": 125
  },
  ...
]
```

## Output Fields

| Field | Description |
|-------|-------------|
| ID | Unique checkpoint identifier |
| Created | Timestamp when checkpoint was created |
| Batches | Number of batches processed |
| Records | Total number of records validated |
| Issues | Total number of validation issues found |

## Related Commands

- [`realtime checkpoint show`](show.md) - Show checkpoint details
- [`realtime checkpoint delete`](delete.md) - Delete a checkpoint
- [`realtime validate`](../validate.md) - Validate streaming data
