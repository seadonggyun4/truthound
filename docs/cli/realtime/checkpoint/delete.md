# realtime checkpoint delete

Delete a checkpoint.

## Synopsis

```bash
truthound realtime checkpoint delete <CHECKPOINT_ID> [OPTIONS]
```

## Description

Deletes a streaming validation checkpoint. By default, prompts for confirmation before deleting.

## Arguments

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `CHECKPOINT_ID` | TEXT | Yes | Checkpoint ID to delete |

## Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--dir`, `-d` | PATH | `./checkpoints` | Checkpoint directory |
| `--force`, `-f` | FLAG | `false` | Skip confirmation prompt |

## Examples

### Delete with confirmation

```bash
$ truthound realtime checkpoint delete abc12345
Delete checkpoint 'abc12345'? [y/N]: y
Checkpoint 'abc12345' deleted.
```

### Delete without confirmation

```bash
$ truthound realtime checkpoint delete abc12345 --force
Checkpoint 'abc12345' deleted.
```

### Delete from custom directory

```bash
truthound realtime checkpoint delete abc12345 --dir ./my_checkpoints
```

## Warning

!!! warning "Irreversible Operation"
    Deleting a checkpoint is irreversible. Once deleted, you cannot resume validation from that checkpoint. Use the `--force` flag with caution.

## Error Handling

If the checkpoint is not found:

```bash
$ truthound realtime checkpoint delete nonexistent
Error: Checkpoint 'nonexistent' not found
```

If the checkpoint directory doesn't exist:

```bash
$ truthound realtime checkpoint delete abc12345 --dir ./missing
Error: Checkpoint directory not found: ./missing
```

## Related Commands

- [`realtime checkpoint list`](list.md) - List all checkpoints
- [`realtime checkpoint show`](show.md) - Show checkpoint details
