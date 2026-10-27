"""Partition operations absent from individual native dataframe APIs.

Rows remain in the selected backend's workers. Round robin starts at zero in
each input partition/block and preserves every input row, including nulls.
"""


def _split_pandas(frame, count):
    return tuple(frame.iloc[index::count] for index in range(count))


def _concat_pandas(*frames):
    import pandas as pd

    return pd.concat(frames)


def round_robin_dask(data, count):
    import dask
    import dask.dataframe as dd

    pieces = [dask.delayed(_split_pandas, nout=count)(part, count) for part in data.to_delayed()]
    return dd.from_delayed(
        [dask.delayed(_concat_pandas)(*[part[index] for part in pieces]) for index in range(count)],
        meta=data._meta,
    )


def _split_arrow(table, count):
    import pyarrow as pa

    if count == 1:
        return table
    return tuple(
        table.take(pa.array(range(index, table.num_rows, count), type=pa.int64()))
        for index in range(count)
    )


def _concat_arrow(*tables):
    import pyarrow as pa

    return pa.concat_tables(tables)


def round_robin_ray(data, count):
    import ray

    source = data.to_arrow_refs()
    if not source:
        return data.repartition(count)
    split = ray.remote(_split_arrow).options(num_returns=count)
    pieces = []
    for block in source:
        result = split.remote(block, count)
        # Ray's num_returns=1 returns a single ref rather than a ref list.
        pieces.append([result] if count == 1 else result)
    merge = ray.remote(_concat_arrow)
    return ray.data.from_arrow_refs(
        [merge.remote(*[part[index] for part in pieces]) for index in range(count)]
    )


def round_robin_spark(data, count):
    def split(rows):
        for index, row in enumerate(rows):
            yield index % count, row

    partitioned = data.rdd.mapPartitions(split).partitionBy(count, lambda index: index).values()
    return data.sparkSession.createDataFrame(partitioned, data.schema)


def _range_samples(frame, column, count):
    import numpy as np

    values = frame[column].dropna().sort_values()
    positions = np.linspace(0, len(values) - 1, min(len(values), count + 1), dtype=int)
    return values.iloc[positions]


def _split_range(frame, column, count, bounds):
    import numpy as np
    import pandas as pd

    selected = frame[column].notna().to_numpy()
    buckets = np.full(len(frame), count - 1, dtype=np.int64)
    if selected.any():
        buckets[selected] = pd.Index(bounds).searchsorted(frame.loc[selected, column], side="right")
    return tuple(frame.iloc[np.flatnonzero(buckets == index)] for index in range(count))


def _concat_range(column, *frames):
    return _concat_pandas(*frames).sort_values(column, kind="stable", na_position="last")


def range_partition_dask(data, count, column, scheduler):
    """Plan with selected scheduler, allowing repeated keys and null rows.

    Only bounded per-partition key samples reach the driver. Raw rows remain in
    Dask tasks, including empty output partitions; no global scheduler changes.
    """
    import dask
    import dask.dataframe as dd
    import numpy as np

    samples = (
        data.map_partitions(_range_samples, column, count, meta=data._meta[column])
        .compute(scheduler=scheduler)
        .sort_values()
    )
    positions = np.linspace(0, len(samples) - 1, count + 1, dtype=int)[1:-1]
    bounds = samples.iloc[positions].tolist() if len(samples) else []
    pieces = [
        dask.delayed(_split_range, nout=count)(part, column, count, bounds)
        for part in data.to_delayed()
    ]
    return dd.from_delayed(
        [
            dask.delayed(_concat_range)(column, *[part[index] for part in pieces])
            for index in range(count)
        ],
        meta=data._meta,
    )
