# realtime checkpoint show

Show details of a specific checkpoint.

## Synopsis

```bash
truthound realtime checkpoint show <CHECKPOINT_ID> [OPTIONS]
```

## Description

Displays detailed information about a specific streaming validation checkpoint, including stream position and state snapshot.

## Arguments

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `CHECKPOINT_ID` | TEXT | Yes | Checkpoint ID to show |

## Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--dir`, `-d` | PATH | `./checkpoints` | Checkpoint directory |

## Examples

### Show checkpoint details

```bash
$ truthound realtime checkpoint show abc12345

Checkpoint: abc12345
========================================
Created: 2026-01-15 10:30:00
Batches processed: 50
Total records: 50000
Total issues: 125

Stream Position:
  offset: 125000
  partition: 0
  topic: my_topic

State snapshot keys: ['validator_state', 'metrics', 'last_batch_id']
```

### Show checkpoint in custom directory

```bash
truthound realtime checkpoint show abc12345 --dir ./my_checkpoints
```

## Output Fields

| Field | Description |
|-------|-------------|
| Checkpoint | Unique checkpoint identifier |
| Created | Timestamp when checkpoint was created |
| Batches processed | Number of batches validated |
| Total records | Total number of records validated |
| Total issues | Total number of validation issues found |
| Stream Position | Current position in the stream (offset, partition, topic) |
| State snapshot keys | Keys stored in the checkpoint state |

## Error Handling

If the checkpoint is not found:

```bash
$ truthound realtime checkpoint show nonexistent
Error: Checkpoint 'nonexistent' not found
```

## Related Commands

- [`realtime checkpoint list`](list.md) - List all checkpoints
- [`realtime checkpoint delete`](delete.md) - Delete a checkpoint
