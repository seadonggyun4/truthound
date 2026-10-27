"""Native sampling plans with collision-free columns and exact row budgets."""

import inspect
import random

import polars as pl


def names(frame, count):
    occupied = set(frame.collect_schema().names())
    result = []
    for index in range(count):
        candidate = f"__truthound_sampling_{index}"
        while candidate in occupied:
            candidate += "_"
        occupied.add(candidate)
        result.append(candidate)
    return result


def ordered(frame, size, seed, column=None):
    index, priority = names(frame, 2)
    value = pl.col(column) if column else pl.col(index)
    return (
        frame.with_row_index(index)
        .with_columns(value.hash(seed).alias(priority))
        .sort([priority, index])
        .head(size)
        .drop([index, priority])
    )


def stratified(frame, size, seed, column):
    index, priority, count, quota, rank = names(frame, 5)
    groups = frame.group_by(column, maintain_order=True).agg(pl.len().alias(count)).collect()
    total = groups[count].sum()
    counts = groups[count].to_list()
    quotas = [value * size // total for value in counts]
    remaining = size - sum(quotas)
    order = sorted(range(len(counts)), key=lambda item: -(counts[item] * size % total))
    for item in order[:remaining]:
        quotas[item] += 1
    selected = groups.with_columns(pl.Series(quota, quotas)).select(column, quota)
    null_option = "nulls_equal" if "nulls_equal" in inspect.signature(pl.LazyFrame.join).parameters else "join_nulls"
    return (
        frame.with_row_index(index)
        .with_columns(pl.col(index).hash(seed).alias(priority))
        .sort([priority, index])
        .with_columns(pl.int_range(pl.len()).over(column).alias(rank))
        .join(selected.lazy(), on=column, how="left", **{null_option: True})
        .filter(pl.col(rank) < pl.col(quota))
        .drop([index, priority, rank, quota])
    )


def reservoir(frame, size, seed):
    rng = random.Random(seed)
    rows = []
    observed = 0
    if hasattr(frame, "collect_batches"):
        chunks = frame.collect_batches(chunk_size=1024, maintain_order=True)
    else:
        # Older Polars has no incremental iterator. Its streaming query engine
        # may still materialize the result; this is not a hard heap bound.
        chunks = frame.collect(streaming=True).iter_slices(1024)
    try:
        for chunk in chunks:
            for row in chunk.iter_rows():
                if observed < size:
                    rows.append(row)
                else:
                    slot = rng.randrange(observed + 1)
                    if slot < size:
                        rows[slot] = row
                observed += 1
    finally:
        close = getattr(chunks, "close", None)
        if close is not None:
            close()
    return pl.DataFrame(rows, schema=frame.collect_schema(), orient="row").lazy(), observed
