"""SkyRL backend — the same training, on your own compute.

SkyRL serves the Tinker protocol, so the integration is a redirect rather than
a second client: point `TINKER_BASE_URL` at it and every client the run builds
(training, sampling, and harbor's rollout LLM) follows.
"""

from __future__ import annotations

import pytest

pytest.importorskip("tinker")

from evsys_sdk.backends.skyrl import SkyRLBackend
from evsys_sdk.backends.tinker import PROTOCOL_TINKER, TinkerBackend
from evsys_sdk.registry import get_backend, list_backends


class TestRegistration:
    def test_selectable_by_name_from_yaml(self):
        assert "skyrl" in list_backends()
        assert get_backend("skyrl") is SkyRLBackend

    def test_config_forbids_typos(self):
        with pytest.raises(Exception):
            SkyRLBackend.Config(base_url="http://x", prot="oops")


class TestProtocolCapability:
    """Algorithms must gate on what a backend can DO, not what it is called."""

    def test_both_backends_declare_the_tinker_protocol(self):
        assert TinkerBackend.protocol == PROTOCOL_TINKER
        assert SkyRLBackend.protocol == PROTOCOL_TINKER

    def test_algorithms_accept_any_backend_speaking_it(self):
        from evsys_sdk.algorithms.base import BaseAlgorithm

        class _Ctx:
            class backend:
                name = "skyrl"
                protocol = PROTOCOL_TINKER

        # the gate is the first thing train() does; getting past it is the test
        algo = BaseAlgorithm.__new__(BaseAlgorithm)
        with pytest.raises(Exception) as e:
            algo.train(_Ctx())
        assert "requires a backend speaking" not in str(e.value)

    def test_a_backend_without_the_protocol_is_rejected_by_capability(self):
        from evsys_sdk.algorithms.base import BaseAlgorithm

        class _Ctx:
            class backend:
                name = "local"

        algo = BaseAlgorithm.__new__(BaseAlgorithm)
        with pytest.raises(RuntimeError, match="protocol"):
            algo.train(_Ctx())


class TestPrepareRedirects:
    def _backend(self, monkeypatch, **kw):
        b = SkyRLBackend(health_check=False, **kw)
        monkeypatch.setattr("evsys_sdk.backends.skyrl._connect", lambda kwargs: kwargs)
        return b

    def test_base_url_is_exported_so_every_client_follows(self, monkeypatch):
        """The whole integration: clients are built with no arguments and read
        TINKER_BASE_URL, so setting it once redirects training AND rollouts."""
        monkeypatch.delenv("TINKER_BASE_URL", raising=False)
        b = self._backend(monkeypatch, base_url="http://gpu-box:8000")
        h = b.prepare(model={"name": "Qwen/Qwen3-4B"}, run_dir="/tmp/r")

        import os
        assert os.environ["TINKER_BASE_URL"] == "http://gpu-box:8000"
        assert h["base_url"] == "http://gpu-box:8000"
        assert h["protocol"] == PROTOCOL_TINKER
        assert h["model_name"] == "Qwen/Qwen3-4B"

    def test_a_trailing_slash_does_not_double_up(self, monkeypatch):
        b = self._backend(monkeypatch, base_url="http://gpu-box:8000/")
        b.prepare(model={"name": "m"}, run_dir="/tmp/r")

        import os
        assert os.environ["TINKER_BASE_URL"] == "http://gpu-box:8000"

    def test_a_placeholder_key_is_supplied(self, monkeypatch):
        """SkyRL authenticates nobody, but the tinker client will not construct
        without some key — users should not have to invent one."""
        monkeypatch.delenv("TINKER_API_KEY", raising=False)
        self._backend(monkeypatch).prepare(model={"name": "m"}, run_dir="/tmp/r")

        import os
        assert os.environ["TINKER_API_KEY"] == "tml-dummy"

    def test_a_real_key_is_left_alone(self, monkeypatch):
        monkeypatch.setenv("TINKER_API_KEY", "tml-mine")
        self._backend(monkeypatch).prepare(model={"name": "m"}, run_dir="/tmp/r")

        import os
        assert os.environ["TINKER_API_KEY"] == "tml-mine"


class TestHealthProbe:
    def test_an_absent_server_says_so_instead_of_stalling(self, monkeypatch):
        """Without this the tinker client retries a refused connection and the
        run just hangs — the failure mode that cost us an afternoon."""
        def boom(*a, **k):
            raise ConnectionRefusedError("nope")

        monkeypatch.setattr("evsys_sdk.backends.skyrl.urllib.request.urlopen", boom)
        with pytest.raises(RuntimeError, match="no SkyRL server reachable"):
            SkyRLBackend(base_url="http://localhost:8000").prepare(
                model={"name": "m"}, run_dir="/tmp/r")

    def test_an_http_error_still_counts_as_alive(self, monkeypatch):
        """A 404/401 proves something is listening; only a transport failure
        means 'no server'."""
        import urllib.error

        def http_error(*a, **k):
            raise urllib.error.HTTPError("u", 404, "nf", {}, None)  # type: ignore[arg-type]

        monkeypatch.setattr("evsys_sdk.backends.skyrl.urllib.request.urlopen", http_error)
        monkeypatch.setattr("evsys_sdk.backends.skyrl._connect", lambda kwargs: kwargs)
        h = SkyRLBackend().prepare(model={"name": "m"}, run_dir="/tmp/r")
        assert h["backend"] == "skyrl"
