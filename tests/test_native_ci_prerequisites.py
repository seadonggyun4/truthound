"""The runner's APT route retains signed-index hash authentication."""

import hashlib
from types import SimpleNamespace

import pytest
from verification.ci import native_prerequisites as prerequisites

pytestmark = pytest.mark.contract


@pytest.mark.parametrize("tampered", [False, True])
def test_apt_mirror_font_bytes_must_match_signed_index_before_install(tmp_path, monkeypatch, tampered):
    payload = b"synthetic-font-package-authentication-fixture"
    relative = "pool/main/f/fonts-noto-cjk/synthetic-font.deb"
    metadata = "Filename: " + relative + "\nSHA256: " + hashlib.sha256(payload).hexdigest() + "\n"
    calls = []

    def execute(args, **kwargs):
        args = list(args)
        calls.append(args)
        if args[0] == "curl":
            assert args[-1] == "http://azure.archive.ubuntu.com/ubuntu/" + relative
            assert args[args.index("--max-time") + 1] == "120"
            assert args[args.index("--speed-limit") + 1] == "524288"
            (tmp_path / "synthetic-font.deb").write_bytes(b"tampered" if tampered else payload)
        return SimpleNamespace(returncode=0)

    monkeypatch.setenv("RUNNER_TEMP", str(tmp_path))
    monkeypatch.setattr(prerequisites, "sys", SimpleNamespace(argv=["prerequisites", "libpango-1.0-0"]))
    monkeypatch.setattr(prerequisites.subprocess, "check_output", lambda *args, **kwargs: metadata)
    monkeypatch.setattr(prerequisites.subprocess, "run", execute)
    if tampered:
        with pytest.raises(ValueError, match="APT_FONT_HASH_MISMATCH"):
            prerequisites.main()
        assert not any("dpkg" in args for args in calls)
    else:
        prerequisites.main()
        assert ["sudo", "dpkg", "--install", str(tmp_path / "synthetic-font.deb")] in calls
        assert "fonts-noto-cjk" not in calls[-1]
        assert calls[-1][-1] == "libpango-1.0-0"
