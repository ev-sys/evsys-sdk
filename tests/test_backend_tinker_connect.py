"""The connect guard: a stalled client must surface as an error, not a hang."""
import pytest
from evsys_sdk.backends import tinker as tb


def test_a_stalled_client_raises_with_the_transport_hint(monkeypatch):
    import time
    monkeypatch.setattr(tb, "CONNECT_TIMEOUT_S", 0.3)
    monkeypatch.setattr(tb.tinker, "ServiceClient", lambda **k: time.sleep(30))
    with pytest.raises(RuntimeError, match="pyqwest"):
        tb._connect({})


def test_a_healthy_client_is_returned_unchanged(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(tb.tinker, "ServiceClient", lambda **k: sentinel)
    assert tb._connect({}) is sentinel
