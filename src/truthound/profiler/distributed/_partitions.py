"""Shared native row, column, hybrid and full-row hash partition plans."""
import math

from truthound.profiler.distributed.base import PartitionInfo, PartitionStrategy


def column_groups(columns, count, strategy):
    if strategy is PartitionStrategy.COLUMN_BASED:
        groups = min(count, len(columns))
    elif strategy is PartitionStrategy.HYBRID:
        groups = min(max(1, math.ceil(math.sqrt(count))), len(columns))
    else:
        groups = 1
    if not columns:
        return [[]]
    return [columns[index::groups] for index in range(groups)]


def expand(physical_count, columns, requested, strategy):
    groups = column_groups(columns, requested, strategy)
    plan = []
    for physical in range(physical_count):
        for selected in groups:
            plan.append(PartitionInfo(len(plan), physical_count * len(groups), columns=selected,
                                      metadata={'source_partition': physical, 'hash_partition': strategy is PartitionStrategy.HASH}))
    return plan


def polars_plan(rows, columns, count, strategy):
    if not isinstance(strategy, PartitionStrategy):
        raise ValueError('Unknown partition strategy')
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError('Partition count must be positive')
    groups = column_groups(columns, count, strategy)
    physical = 1 if strategy is PartitionStrategy.COLUMN_BASED else min(max(1, rows), max(1, math.ceil(count / len(groups))))
    plan = expand(physical, columns, count, strategy)
    for partition in plan:
        index = partition.metadata['source_partition']
        partition.start_row = rows * index // physical
        partition.end_row = rows * (index + 1) // physical
        partition.metadata['hash_count'] = physical
    return plan


def select_polars(frame, partition):
    if partition.metadata.get('hash_partition'):
        frame = frame.filter(frame.hash_rows(seed=0) % partition.metadata['hash_count'] == partition.metadata['source_partition'])
    else:
        frame = frame.slice(partition.start_row, partition.end_row - partition.start_row)
    return frame.select(partition.columns)
