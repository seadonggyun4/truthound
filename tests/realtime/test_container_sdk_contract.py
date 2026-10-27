"""SDK boundaries for disposable native streaming test environments."""

import asyncio
import sys
import time
from threading import Lock
from types import ModuleType, SimpleNamespace

import pytest

from truthound.realtime.testing import containers

pytestmark = pytest.mark.contract


@pytest.mark.parametrize("host,address", [("127.0.0.1", "127.0.0.1"), ("::1", "[::1]")])
def test_redis_endpoint_uses_the_sdk_host_and_selected_mapped_port(host, address):
    owner = containers.RedisTestContainer(containers.RedisTestContainerConfig(port=6380))
    ports = []
    owner._container = SimpleNamespace(
        get_container_host_ip=lambda: host,
        get_exposed_port=lambda port: ports.append(port) or '15432',
    )
    assert owner.get_connection_url() == f'redis://{address}:15432'
    assert ports == [6380]


@pytest.mark.parametrize("disabled", [False, True])
def test_parallel_environment_start_initializes_the_native_reaper_once(monkeypatch, disabled):
    created = []
    lock = Lock()

    class Reaper:
        instance = None

        @classmethod
        def get_instance(cls):
            if cls.instance is None:
                with lock:
                    created.append(object())
                time.sleep(.05)
                cls.instance = object()
            return cls.instance

    class Native:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            if not disabled:
                Reaper.get_instance()

        def stop(self):
            pass

        def get_bootstrap_server(self):
            return '127.0.0.1:19092'

        def get_connection_url(self):
            return 'redis://127.0.0.1:16379'

        def get_container_host_ip(self):
            return '127.0.0.1'

        def get_exposed_port(self, port):
            return '16379'

    modules = {
        'testcontainers': {}, 'testcontainers.core': {},
        'testcontainers.core.container': {'Reaper': Reaper},
        'testcontainers.core.config': {'testcontainers_config': SimpleNamespace(ryuk_disabled=disabled)},
        'testcontainers.kafka': {'KafkaContainer': Native},
        'testcontainers.redis': {'RedisContainer': Native},
    }
    for name, attributes in modules.items():
        module = ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)

    async def run():
        environment = containers.StreamTestEnvironment(['kafka', 'redis'])
        try:
            await environment.start()
            assert len(created) == (0 if disabled else 1)
        finally:
            await environment.stop()

    asyncio.run(run())
