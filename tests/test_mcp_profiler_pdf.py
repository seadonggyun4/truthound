"""Actual optional PDF provider and installed native MCP binary artifacts."""
import base64
import hashlib
import subprocess
import sys
from pathlib import Path

import pytest

from tests import test_mcp_plugin_bindings as plugins
from tests.test_mcp_brokers import approved
from tests.test_mcp_profiler_services import call, prepare
from truthound.profiler.visualization import ProfileData, ReportConfig, ReportExporter

runtime = plugins.runtime
pytestmark = pytest.mark.contract


def native_loader(monkeypatch):
    if sys.platform == 'darwin' and Path('/opt/homebrew/lib/libgobject-2.0.dylib').exists():
        monkeypatch.setenv('DYLD_FALLBACK_LIBRARY_PATH', '/opt/homebrew/lib')


def test_actual_pdf_static_charts_and_atomic_failure(tmp_path, monkeypatch):
    native_loader(monkeypatch)
    profile = ProfileData(table_name='synthetic_pdf_table', row_count=3, column_count=1,
                          columns=[{'name': 'amount', 'data_type': 'integer', 'null_ratio': 0.0}])
    target = tmp_path / 'report.pdf'
    ReportExporter(ReportConfig(renderer='plotly')).to_file(profile, target, format='pdf')
    assert target.read_bytes().startswith(b'%PDF-')
    text = subprocess.run(['pdftotext', str(target), '-'], check=True, capture_output=True, text=True).stdout
    assert 'synthetic_pdf_table' in text and 'amount' in text
    assert 'Data Type Distribution' in text
    previous = target.read_bytes()
    import truthound.stores._filesystem_io as native_io
    def fail(path, payload):
        raise OSError('synthetic atomic publication failure')
    monkeypatch.setattr(native_io, 'publish', fail)
    with pytest.raises(OSError):
        ReportExporter.to_pdf('<h1>new content</h1>', target)
    assert target.read_bytes() == previous


def test_installed_native_pdf_export_and_fixed_slot_digest(runtime, monkeypatch):
    native_loader(monkeypatch)
    # Native DLL discovery is part of the preinstalled approved factory.
    runtime[2].with_name('pdf-native-mode').write_text('enabled')
    service = prepare(runtime, monkeypatch)
    _, profile = call(runtime, service, 'profile', profiler_id='custom', asset_id=runtime[4])
    response = approved(runtime, 'profiler_services_execute', binding_id='local', service_id=service,
                        command='visualization_export', visualizer_id='svg', export_format='pdf',
                        profile_id=profile['profile_id'], idempotency_key='native-pdf-export')
    assert runtime[0]._session.resources.get(response['data']['resource_id'], 'artifact').media_type == 'application/pdf'
    parts, offset = [], 0
    while offset is not None:
        result = plugins.invoke(runtime[0], 'artifacts_read', resource_id=response['data']['resource_id'], offset=offset)
        parts.append(base64.b64decode(result['data']['base64']))
        offset = result['data']['next_offset']
    payload = b''.join(parts)
    assert payload.startswith(b'%PDF-')
    _, saved = call(runtime, service, 'visualization_save', visualizer_id='svg', export_format='pdf',
                    visualization_slot_id='local', profile_id=profile['profile_id'])
    persisted = runtime[2].with_name('private-native-profile.html').read_bytes()
    assert persisted.startswith(b'%PDF-') and saved['sha256'] == hashlib.sha256(persisted).hexdigest()
    assert saved['bytes'] == len(persisted)
