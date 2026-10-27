"""Scoped compatibility for owned Windows Ray 2.58 node discovery."""

from __future__ import annotations

import sys
import threading
import time
from contextlib import contextmanager

_STARTUP_LOCK = threading.RLock()


def _registered_node(client, node_id, remaining):
    from ray import NodeID
    from ray.core.generated.gcs_pb2 import GcsNodeInfo
    from ray.core.generated.gcs_service_pb2 import GetAllNodeInfoRequest

    if remaining <= 0:
        raise TimeoutError("Ray local node discovery exceeded its startup deadline")
    binary_id = NodeID.from_hex(node_id).binary()
    selector = GetAllNodeInfoRequest.NodeSelector(node_id=binary_id)
    records = client.get_all_node_info(
        timeout=remaining,
        node_selectors=[selector],
        state_filter=GcsNodeInfo.GcsNodeState.ALIVE,
    )
    matches = [
        node for node in records.values()
        if node.node_id == binary_id and node.state == GcsNodeInfo.GcsNodeState.ALIVE
    ]
    if len(matches) != 1:
        raise RuntimeError("Ray local node is not uniquely registered as alive")
    node = matches[0]
    if not 0 < node.node_manager_port < 65536:
        raise RuntimeError("Ray local node has no valid node manager port")
    return {
        "object_store_socket_name": node.object_store_socket_name,
        "raylet_socket_name": node.raylet_socket_name,
        "node_manager_port": node.node_manager_port,
        "node_id": node.node_id.hex(),
        "runtime_env_agent_port": node.runtime_env_agent_port,
        "metrics_agent_port": node.metrics_agent_port,
        "metrics_export_port": node.metrics_export_port,
        "dashboard_agent_listen_port": node.dashboard_agent_listen_port,
        "labels": dict(node.labels),
    }


@contextmanager
def local_node_lookup():
    """Use the owning Node's live client, restoring SDK hooks on every exit.

    Ray's existing startup loop still owns readiness polling. This avoids the
    temporary GlobalStateAccessor/GetNode path implicated in Windows native
    access violations, without manufacturing readiness or touching other threads.
    """
    from ray._private import node as node_module
    from ray._private import services

    with _STARTUP_LOCK:
        owner = threading.get_ident()
        original_start = node_module.Node.start_ray_processes
        original_lookup = services.get_node
        nodes = {}
        deadline = time.monotonic() + 30
        state = {"queries": 0}

        def start(node, *args, **kwargs):
            if threading.get_ident() == owner:
                nodes[(node.gcs_address, node._node_id)] = node
            return original_start(node, *args, **kwargs)

        def lookup(address, node_id):
            node = nodes.get((address, node_id))
            if threading.get_ident() != owner or node is None:
                return original_lookup(address, node_id)
            state["queries"] += 1
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Ray local node discovery exceeded its startup deadline")
            try:
                return _registered_node(node.get_gcs_client(), node_id, remaining)
            except RuntimeError:
                # Preserve the SDK GetNode readiness polling cadence and budget.
                time.sleep(min(1, max(0, deadline - time.monotonic())))
                raise

        node_module.Node.start_ray_processes = start
        services.get_node = lookup
        try:
            yield state
        finally:
            services.get_node = original_lookup
            node_module.Node.start_ray_processes = original_start
            nodes.clear()


def initialize_ray(**kwargs):
    """Initialize real Ray; only owned local Windows 2.58 startup is adapted."""
    import ray

    if (
        sys.platform == "win32"
        and ray.__version__.startswith("2.58.")
        and kwargs.get("address") in {None, "local"}
    ):
        with local_node_lookup():
            return ray.init(**kwargs)
    return ray.init(**kwargs)
