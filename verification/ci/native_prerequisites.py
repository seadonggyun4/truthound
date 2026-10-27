"""Install native CI providers with a hash-checked, bounded CJK font download."""

import hashlib
import os
import subprocess
import sys
from pathlib import Path


def run(*args, **kwargs):
    return subprocess.run(args, check=True, **kwargs)


def main():
    options = ('-o', 'Acquire::Retries=3', '-o', 'Acquire::http::Timeout=30',
               '-o', 'Acquire::https::Timeout=30')
    run('sudo', 'apt-get', *options, 'update')
    package = 'fonts-noto-cjk'
    metadata = subprocess.check_output(
        ['apt-cache', 'show', '--no-all-versions', package], text=True
    )
    fields = dict(line.split(': ', 1) for line in metadata.splitlines() if ': ' in line and not line.startswith(' '))
    relative = fields['Filename']
    digest = fields['SHA256']
    if not relative.startswith('pool/') or '..' in relative.split('/') or len(digest) != 64:
        raise ValueError('INVALID_APT_FONT_METADATA')
    destination = Path(os.environ['RUNNER_TEMP']) / Path(relative).name
    # Both mirrors serve the exact package authenticated by APT's signed index.
    for mirror in ('https://archive.ubuntu.com/ubuntu/', 'https://azure.archive.ubuntu.com/ubuntu/'):
        result = subprocess.run([
            'curl', '--fail', '--location', '--silent', '--show-error',
            '--connect-timeout', '15', '--max-time', '120', '--speed-limit', '524288',
            '--speed-time', '20', '--retry', '2', '--retry-all-errors',
            '--output', str(destination), mirror + relative,
        ])
        if result.returncode == 0:
            if hashlib.sha256(destination.read_bytes()).hexdigest() != digest:
                raise ValueError('APT_FONT_HASH_MISMATCH')
            break
    else:
        raise RuntimeError('APT_FONT_DOWNLOAD_FAILED')
    run('sudo', 'apt-get', *options, 'install', '--no-install-recommends', '-y',
        str(destination), *sys.argv[1:])


if __name__ == '__main__':
    main()
