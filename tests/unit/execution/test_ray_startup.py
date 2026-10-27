"""Guard the SDK compatibility boundary without relaxing native readiness."""

import threading
from types import SimpleNamespace

import pytest

from truthound.execution.distributed import _ray_startup as startup

pytestmark = pytest.mark.contract


def node_record(**changes):
    from ray import NodeID
    from ray.core.generated.gcs_pb2 import GcsNodeInfo

    identity = NodeID.from_random()
    record = GcsNodeInfo(
        node_id=identity.binary(), state=GcsNodeInfo.GcsNodeState.ALIVE,
        node_manager_port=12345, runtime_env_agent_port=12346,
        metrics_agent_port=12347, metrics_export_port=12348,
        dashboard_agent_listen_port=12349,
        object_store_socket_name="owned-store", raylet_socket_name="owned-raylet",
        labels={"purpose": "synthetic"}, **changes,
    )
    return identity.hex(), record


@pytest.mark.parametrize("bad", ["missing", "dead", "wrong-id", "port", "duplicate"])
def test_missing_dead_mismatched_or_ambiguous_nodes_cannot_be_ready(bad):
    from ray.core.generated.gcs_pb2 import GcsNodeInfo

    identity, record = node_record()
    records = {"first": record}
    if bad == "missing":
        records = {}
    elif bad == "dead":
        record.state = GcsNodeInfo.GcsNodeState.DEAD
    elif bad == "wrong-id":
        record.node_id = b"0" * len(record.node_id)
    elif bad == "port":
        record.node_manager_port = 0
    else:
        records["second"] = record
    with pytest.raises(RuntimeError):
        startup._registered_node(SimpleNamespace(get_all_node_info=lambda **kw: records), identity, 1)


def test_alive_node_retains_sdk_fields_and_exact_selector():
    from ray.core.generated.gcs_pb2 import GcsNodeInfo

    identity, record = node_record()
    calls = []

    def query(**kwargs):
        calls.append(kwargs)
        return {"owned": record}

    observed = startup._registered_node(SimpleNamespace(get_all_node_info=query), identity, 0.25)
    assert observed == {
        "object_store_socket_name": "owned-store", "raylet_socket_name": "owned-raylet",
        "node_manager_port": 12345, "node_id": identity, "runtime_env_agent_port": 12346,
        "metrics_agent_port": 12347, "metrics_export_port": 12348,
        "dashboard_agent_listen_port": 12349, "labels": {"purpose": "synthetic"},
    }
    assert calls[0]["timeout"] == 0.25
    assert calls[0]["node_selectors"][0].node_id == record.node_id
    assert calls[0]["state_filter"] == GcsNodeInfo.GcsNodeState.ALIVE
    with pytest.raises(TimeoutError):
        startup._registered_node(SimpleNamespace(get_all_node_info=query), identity, 0)
    assert len(calls) == 1


@pytest.mark.parametrize("fail", [False, True])
def test_scoped_hooks_restore_and_other_threads_keep_original_lookup(monkeypatch, fail):
    from ray._private import node as node_module
    from ray._private import services

    events = []
    def original_lookup(address, identity):
        events.append((address, identity))
        return "original"

    def original_start(self):
        return "started"
    node_type = type("OwnedNode", (), {"start_ray_processes": original_start})
    monkeypatch.setattr(node_module, "Node", node_type)
    monkeypatch.setattr(services, "get_node", original_lookup)
    monkeypatch.setattr(startup, "_registered_node", lambda client, identity, remaining: "alive")
    owned = node_type()
    owned.gcs_address, owned._node_id = "127.0.0.1:12345", "owned"
    owned.get_gcs_client = lambda: object()
    try:
        with startup.local_node_lookup() as state:
            assert owned.start_ray_processes() == "started"
            assert services.get_node(owned.gcs_address, "owned") == "alive"
            result = []
            thread = threading.Thread(target=lambda: result.append(services.get_node(owned.gcs_address, "owned")))
            thread.start()
            thread.join(timeout=5)
            assert not thread.is_alive()
            assert result == ["original"] and state["queries"] == 1
            if fail:
                raise ValueError("synthetic startup failure")
    except ValueError:
        assert fail
    assert services.get_node is original_lookup
    assert node_type.start_ray_processes is original_start


@pytest.mark.parametrize("platform,version,address,guarded", [
    ("win32", "2.58.1", "local", True), ("win32", "2.58.1", None, True),
    ("linux", "2.58.1", "local", False), ("win32", "2.48.0", "local", False),
    ("win32", "2.58.1", "auto", False), ("win32", "2.58.1", "ray://synthetic", False),
])
def test_only_owned_windows_258_initialization_uses_guard(monkeypatch, platform, version, address, guarded):
    import contextlib

    import ray

    events = []

    @contextlib.contextmanager
    def guard():
        events.append("enter")
        yield
        events.append("exit")

    monkeypatch.setattr(startup, "sys", SimpleNamespace(platform=platform))
    monkeypatch.setattr(ray, "__version__", version)
    monkeypatch.setattr(ray, "init", lambda **kwargs: events.append(kwargs) or "context")
    monkeypatch.setattr(startup, "local_node_lookup", guard)
    assert startup.initialize_ray(address=address, num_cpus=2) == "context"
    assert events == (["enter", {"address": address, "num_cpus": 2}, "exit"] if guarded
                      else [{"address": address, "num_cpus": 2}])
