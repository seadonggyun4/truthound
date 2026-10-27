"""Run only disposable broker integration tests with loopback-only Python sockets."""

import ipaddress
import json
import os
import socket
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

TARGETS = (
    'tests/integration/realtime/test_kafka_adapter.py',
    'tests/integration/realtime/test_kinesis_adapter.py',
)


def local_connections(event, args):
    if event != 'socket.connect':
        return
    connection, address = args
    if connection.family not in (socket.AF_INET, socket.AF_INET6):
        return
    host = address[0]
    if host == 'localhost':
        return
    try:
        allowed = ipaddress.ip_address(host).is_loopback
    except ValueError:
        allowed = False
    if not allowed:
        raise PermissionError('BROKER_TEST_NON_LOOPBACK_CONNECTION')


def main():
    import aiobotocore  # noqa: F401
    import aiokafka  # noqa: F401
    import kafka  # noqa: F401
    import testcontainers  # noqa: F401

    output = Path('test-artifacts/brokers.xml')
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='truthound-broker-config-') as directory:
        empty = Path(directory) / 'empty'
        empty.write_text('', encoding='utf-8')
        os.environ.update(
            AWS_ACCESS_KEY_ID='test', AWS_SECRET_ACCESS_KEY='test',
            AWS_SESSION_TOKEN='test', AWS_DEFAULT_REGION='us-east-1',
            AWS_EC2_METADATA_DISABLED='true', AWS_SHARED_CREDENTIALS_FILE=str(empty),
            AWS_CONFIG_FILE=str(empty), TESTCONTAINERS_HOST_OVERRIDE='127.0.0.1',
        )
        sys.addaudithook(local_connections)
        result = pytest.main([
            '-q', *TARGETS, '--run-integration', '-p', 'no:cacheprovider',
            '--timeout=180', '--timeout-method=thread', '-o', 'faulthandler_timeout=90',
            '--junitxml=' + str(output),
        ])
    if result != 0:
        raise SystemExit(result)
    root = ET.parse(output).getroot()
    cases = list(root.iter('testcase'))
    assert cases and not list(root.iter('skipped'))
    assert not list(root.iter('failure')) and not list(root.iter('error'))
    print(json.dumps({'actual_broker_tests': len(cases), 'skipped': 0, 'loopback_only': True}))


if __name__ == '__main__':
    main()
