"""MODELS=host: the tagger from the shared model server matches the native one, and an unreachable host fails open.

The equality test needs a model server with WS /v1/tag/stream at MODEL_HOST_URL (Ask the Moon's model_server);
it skips when none is running.
"""
import socket
import sys
from pathlib import Path
from urllib.parse import urlparse

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from server import config, tagger  # noqa: E402
from server.tagger import RemoteTagger, Tagger  # noqa: E402


def _host_up() -> bool:
    """A model server with WS /v1/tag/stream answers at MODEL_HOST_URL."""
    u = urlparse(config.MODEL_HOST_URL)
    try:
        socket.create_connection((u.hostname, u.port or 80), timeout=0.3).close()
        RemoteTagger(domain="assistant").close()
        return True
    except Exception:
        return False


def _script(t):
    out = []
    t.push(["show", "me", "Tycho"]); out.append(t.poll())
    t.push(["Copernicus", "crater", "please"], from_=2); out.append(t.poll())
    t.endpoint(); out.append(t.poll())
    t.reset()
    t.push("my name is Maya and I live at 42 Oak Street".split(), endpoint=True); out.append(t.poll())
    out.append(t.tag_text("Copernicus is about 93 kilometres wide."))
    return [{k: v for k, v in d.items() if k != "ms"} for d in out]


@pytest.fixture
def host_mode(monkeypatch):
    monkeypatch.setattr(config, "MODELS", "host")
    monkeypatch.setitem(tagger._HOST, "down", False)


@pytest.mark.skipif(not _host_up(), reason="no model server with WS /v1/tag/stream at MODEL_HOST_URL")
def test_host_tagger_matches_native(host_mode, monkeypatch):
    remote = Tagger(domain="assistant")
    assert isinstance(remote, RemoteTagger) and remote.h is not None
    monkeypatch.setattr(config, "MODELS", "local")
    native = Tagger(domain="assistant")
    assert type(native) is Tagger
    try:
        assert _script(remote) == _script(native)
    finally:
        remote.close(), native.close()
    assert remote.h is None


@pytest.mark.skipif(not _host_up(), reason="no model server with WS /v1/tag/stream at MODEL_HOST_URL")
def test_host_bad_option_raises(host_mode):
    with pytest.raises(ValueError):
        RemoteTagger(domain="assistant", options={"no_such_option": "1"})


def test_unreachable_host_falls_back_to_local(host_mode, monkeypatch):
    s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()   # a port nobody listens on
    monkeypatch.setattr(config, "MODEL_HOST_URL", f"http://127.0.0.1:{port}")
    t = Tagger(domain="assistant")
    try:
        assert type(t) is Tagger and t.h is not None      # the native tagger, working
        assert tagger._HOST["down"]                         # and the host isn't retried for this process
        t.push(["hello"]); assert isinstance(t.poll(), dict)
    finally:
        t.close()
