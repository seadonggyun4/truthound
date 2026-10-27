"""Native distributed profiling must report global data, not partition estimates."""
import polars as pl
import pytest

from truthound.profiler.distributed import DistributedProfiler, PartitionStrategy


@pytest.mark.parametrize('partition', [PartitionStrategy.ROW_BASED, PartitionStrategy.COLUMN_BASED])
def test_local_global_counts_null_statistics_and_string_lengths(partition):
    frame = pl.DataFrame({'x': [1.0, -1.0, None, 3.0], 's': ['a', 'bbbb', None, 'a']})
    with DistributedProfiler.create(backend='local', partition_strategy=partition, num_partitions=3) as profiler:
        result = profiler.profile(frame)
    assert result.row_count == 4
    columns = {column.name: column for column in result.columns}
    assert columns['x'].distinct_count == 4
    assert columns['x'].distribution.mean == pytest.approx(1.0)
    assert columns['x'].distribution.std == pytest.approx(2.0)
    assert columns['s'].min_length == 1 and columns['s'].max_length == 4
    assert columns['s'].avg_length == pytest.approx(2.0)


def test_local_partition_error_cannot_be_successful_profile(monkeypatch):
    with DistributedProfiler.create(backend='local') as profiler:
        def fail(partition, frame):
            raise PermissionError('private failure')
        monkeypatch.setattr(profiler, '_profile_partition', fail)
        with pytest.raises(RuntimeError, match='partition'):
            profiler.profile(pl.DataFrame({'x': [1, 2]}))


def test_dask_global_statistics_match_native_values_and_restore_scheduler():
    import dask

    from truthound.profiler.distributed import DaskConfig
    before = dask.config.get('scheduler', None)
    with DistributedProfiler.create(backend='dask', backend_config=DaskConfig(scheduler='synchronous'), num_partitions=3) as profiler:
        result = profiler.profile(pl.DataFrame({'x': [1.0, -1.0, None, 3.0], 's': ['a', 'bbbb', None, 'a']}))
    columns = {column.name: column for column in result.columns}
    assert result.row_count == 4
    assert columns['x'].distinct_count == 4
    assert columns['x'].distribution.mean == pytest.approx(1.0)
    assert columns['x'].distribution.std == pytest.approx(2.0)
    assert columns['s'].avg_length == pytest.approx(2.0)
    assert dask.config.get('scheduler', None) == before


@pytest.mark.parametrize('backend', ['ray', 'spark'])
def test_actual_remote_profiler_global_contract(tmp_path, backend):
    from tests.verification_support import OwnedProbe
    from truthound.mcp.tables import decode
    runtime = OwnedProbe(backend, tmp_path / (backend + '-profiler.log'))
    try:
        assert runtime.receive()['ready'] is True
        observed = decode(runtime.call('profiler_global'))
        assert observed['existing_runtime_rows'] == 3
        assert observed['callback_sum'] == 6
        assert observed['sampled']['row_count'] == 2
        assert any(pattern['pattern'] == 'email' for pattern in observed['sampled']['columns'][0]['detected_patterns'])
        result = observed['profile']
        assert result['row_count'] == 4
        columns = {column['name']: column for column in result['columns']}
        assert columns['x']['distinct_count'] == 4
        assert columns['x']['null_count'] == 1
        assert columns['x']['distribution']['mean'] == pytest.approx(1)
        assert columns['x']['distribution']['std'] == pytest.approx(2)
        assert columns['s']['avg_length'] == pytest.approx(2)
    finally:
        runtime.close()


def test_dask_cleanup_attempts_all_owned_resources_and_preserves_failures():
    from truthound.profiler.distributed import DaskBackend
    backend = DaskBackend()
    attempts = []
    class Resource:
        def __init__(self, label):
            self.label = label
        def close(self):
            attempts.append(self.label)
            raise OSError('private close failure')
    backend._client = Resource('client')
    backend._cluster = Resource('cluster')
    with pytest.raises(ExceptionGroup) as failure:
        backend.shutdown()
    assert attempts == ['client', 'cluster']
    assert len(failure.value.exceptions) == 2


@pytest.mark.parametrize('backend', ['local', 'dask'])
def test_distributed_sampling_and_pattern_configuration_are_applied(backend):
    from truthound.profiler.distributed import DaskConfig
    settings = {'backend_config': DaskConfig(scheduler='synchronous')} if backend == 'dask' else {}
    frame = pl.DataFrame({'email': ['a@example.com', 'b@example.com', 'plain']})
    with DistributedProfiler.create(backend=backend, sample_size=2, include_patterns=True, **settings) as profiler:
        result = profiler.profile(frame)
        assert result.row_count == 2
        assert any(pattern.pattern == 'email' for pattern in result.columns[0].detected_patterns)
    with DistributedProfiler.create(backend=backend, include_patterns=False, **settings) as profiler:
        result = profiler.profile(frame)
        assert result.row_count == 3 and result.columns[0].detected_patterns == ()


def custom_partition_summary(partition, frame):
    from truthound.profiler.distributed import WorkerResult
    return WorkerResult(partition.partition_id, {}, frame.height, 0, metadata={'observed': frame['x'].sum()})


@pytest.mark.parametrize('backend', ['local', 'dask'])
def test_native_partition_callback_is_invoked(backend):
    from truthound.profiler.distributed import DaskConfig
    settings = {'backend_config': DaskConfig(scheduler='synchronous')} if backend == 'dask' else {}
    with DistributedProfiler.create(backend=backend, **settings) as profiler:
        native = profiler.backend
        frame = pl.DataFrame({'x': [1, 2, 3]})
        partitions = native.distribute_data(frame, num_partitions=2)
        results = native.map_partitions(custom_partition_summary, partitions, frame)
        assert sum(result.metadata['observed'] for result in results) == 6
        assert sum(result.row_count for result in results) == 3


@pytest.mark.parametrize('backend', ['local', 'dask'])
@pytest.mark.parametrize('strategy', list(PartitionStrategy))
def test_partition_policy_preserves_cells_and_applies_column_groups(backend, strategy):
    from truthound.profiler.distributed import DaskConfig, WorkerResult
    settings = {'backend_config': DaskConfig(scheduler='synchronous')} if backend == 'dask' else {}
    frame = pl.DataFrame({'x': [1, 1, 2, 3], 'y': [2, 2, 3, 4]})
    def inspect(partition, selected):
        return WorkerResult(partition.partition_id, {}, selected.height, 0, metadata={'columns': selected.columns, 'sum': sum(selected.sum().row(0)), 'pairs': selected.rows()})
    with DistributedProfiler.create(backend=backend, **settings) as profiler:
        partitions = profiler.backend.distribute_data(frame, num_partitions=4, strategy=strategy)
        results = profiler.backend.map_partitions(inspect, partitions, frame)
    assert sum(item.metadata['sum'] for item in results) == 18
    if strategy in {PartitionStrategy.COLUMN_BASED, PartitionStrategy.HYBRID}:
        assert all(len(item.metadata['columns']) == 1 for item in results)
    if strategy == PartitionStrategy.HASH:
        assert sum((1, 2) in item.metadata['pairs'] for item in results) == 1


@pytest.mark.parametrize('backend', ['ray', 'spark'])
def test_actual_remote_partition_policies(tmp_path, backend):
    from tests.verification_support import OwnedProbe
    from truthound.mcp.tables import decode
    runtime = OwnedProbe(backend, tmp_path / (backend + '-policies.log'))
    try:
        assert runtime.receive()['ready'] is True
        results = decode(runtime.call('profiler_policies'))
        for policy, value in results.items():
            assert value['sum'] == 18
            if policy in {'column_based', 'hybrid'}:
                assert value['max_columns'] == 1
            if policy == 'hash':
                assert value['duplicate_buckets'] == 1
    finally:
        runtime.close()


@pytest.mark.parametrize('backend', ['local', 'dask'])
def test_native_profiler_checkpoint_reopens_and_rejects_corruption(tmp_path, backend):
    from truthound.profiler.distributed import BackendConfig, DaskConfig
    config_type = DaskConfig if backend == 'dask' else BackendConfig
    settings = config_type(checkpoint_enabled=True, checkpoint_dir=str(tmp_path / 'checkpoints'))
    with DistributedProfiler.create(backend=backend, backend_config=settings) as profiler:
        profiler.profile(pl.DataFrame({'x': [1, None, 3]}))
        location = profiler.backend.checkpoint_path
    with DistributedProfiler.create(backend=backend) as reopened:
        partitions = reopened.backend.restore_checkpoint(location, num_partitions=2)
        results = reopened.backend.map_partitions(reopened._profile_partition, partitions, None)
        assert reopened.backend.aggregate_results(results)['x']['distinct_count'] == 3
        from pathlib import Path
        target = next((Path(location) / 'data').rglob('*.parquet'))
        target.write_bytes(target.read_bytes() + b'corruption')
        with pytest.raises(ValueError, match='integrity'):
            reopened.backend.restore_checkpoint(location)


def test_partition_explicit_retry_and_memory_budget_are_applied():
    from truthound.execution.distributed.policies import RetryableExecutionError
    from truthound.profiler.distributed import BackendConfig, LocalBackend, WorkerResult
    attempts = []
    def retry(partition, frame):
        attempts.append(partition.partition_id)
        if len(attempts) == 1:
            raise RetryableExecutionError('safe synthetic retry')
        return WorkerResult(partition.partition_id, {}, frame.height, 0)
    with LocalBackend(BackendConfig(retry_count=1, memory_per_worker='1m')) as backend:
        partitions = backend.distribute_data(pl.DataFrame({'x': [1]}), num_partitions=1)
        results = backend.map_partitions(retry, partitions, None)
        assert not results[0].errors and len(attempts) == 2
        partitions = backend.distribute_data(pl.DataFrame({'x': ['x' * (2 * 1024 * 1024)]}), num_partitions=1)
        results = backend.map_partitions(retry, partitions, None)
        assert results[0].errors == ['MemoryError'] and len(attempts) == 2


def test_partition_deadline_rejects_late_callback_without_replay():
    from truthound.profiler.distributed import BackendConfig, LocalBackend, WorkerResult
    calls = []
    def callback(partition, frame):
        calls.append(1)
        return WorkerResult(partition.partition_id, {}, frame.height, 0)
    with LocalBackend(BackendConfig(timeout_seconds=1e-12, retry_count=5)) as backend:
        partitions = backend.distribute_data(pl.DataFrame({'x': [1]}), num_partitions=1)
        results = backend.map_partitions(callback, partitions, None)
        assert results[0].errors == ['TimeoutError']
        assert len(calls) <= 1


@pytest.mark.parametrize('backend', ['ray', 'spark'])
def test_actual_remote_profiler_checkpoint_reopens(tmp_path, backend):
    from tests.verification_support import OwnedProbe
    from truthound.mcp.tables import decode
    runtime = OwnedProbe(backend, tmp_path / (backend + '-checkpoint.log'))
    try:
        assert runtime.receive()['ready'] is True
        assert decode(runtime.call('profiler_checkpoint')) == {
            'rows': 3, 'distinct': 3, 'corruption_blocked': True, 'existing_rows': 3,
        }
    finally:
        runtime.close()
