"""Credential storage across providers.

One provider is not a supply chain: the spot SKU this project benchmarked on
stopped being offered mid-run. Being authenticated with several is the fix,
and the obstacle is that every provider invents its own file format.
"""
from __future__ import annotations

import json
import pytest

from evsys_sdk.compute import credentials


@pytest.fixture
def home(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    return tmp_path


class TestSaving:
    def test_json_provider_round_trips(self, home):
        p = credentials.save("primeintellect", api_key="pit_abc")
        assert json.loads(p.read_text()) == {"api_key": "pit_abc"}
        assert credentials.status()["primeintellect"] is True

    def test_oauth_pair_is_written_whole(self, home):
        p = credentials.save("verda", client_id="cid", client_secret="sec")
        assert json.loads(p.read_text()) == {"client_id": "cid", "client_secret": "sec"}

    def test_toml_provider(self, home):
        p = credentials.save("runpod", api_key="rp_1")
        assert p.read_text().strip() == 'api_key = "rp_1"'

    def test_raw_provider_writes_the_bare_token(self, home):
        p = credentials.save("lambda", api_key="secret-token")
        assert p.read_text().strip() == "secret-token"

    def test_written_private(self, home):
        """These keys rent GPUs on someone's credit card."""
        p = credentials.save("primeintellect", api_key="pit_abc")
        assert p.stat().st_mode & 0o777 == 0o600

    def test_parent_directory_is_created(self, home):
        p = credentials.save("verda", client_id="a", client_secret="b")
        assert p.parent.is_dir()


class TestRefusesToWriteSomethingBroken:
    def test_missing_field_raises(self, home):
        """A half-written credential fails at launch as 'resources
        unavailable' — hours later, looking exactly like a capacity shortage."""
        with pytest.raises(ValueError, match="client_secret"):
            credentials.save("verda", client_id="only-half")

    def test_empty_value_counts_as_missing(self, home):
        with pytest.raises(ValueError):
            credentials.save("primeintellect", api_key="")

    def test_unknown_provider_lists_the_known_ones(self, home):
        with pytest.raises(KeyError, match="verda"):
            credentials.save("nosuchcloud", api_key="x")


class TestStatus:
    def test_nothing_authenticated_on_a_clean_machine(self, home):
        assert not any(credentials.status().values())

    def test_partial_json_is_not_authenticated(self, home):
        spec = credentials.PROVIDERS["verda"]
        spec.file().parent.mkdir(parents=True, exist_ok=True)
        spec.file().write_text(json.dumps({"client_id": "a"}))
        assert credentials.status()["verda"] is False

    def test_corrupt_json_is_not_authenticated(self, home):
        spec = credentials.PROVIDERS["primeintellect"]
        spec.file().parent.mkdir(parents=True, exist_ok=True)
        spec.file().write_text("{not json")
        assert credentials.status()["primeintellect"] is False

    def test_spot_filter(self, home):
        credentials.save("primeintellect", api_key="a")
        credentials.save("lambda", api_key="b")
        assert credentials.authenticated(spot=True) == ["primeintellect"]
        assert credentials.authenticated(spot=False) == ["lambda"]
        assert credentials.authenticated() == ["lambda", "primeintellect"]


class TestCapabilityFlags:
    """Read from SkyPilot's own _CLOUD_UNSUPPORTED_FEATURES, and load-bearing:
    they decide whether a provider can be used unattended at all."""

    def test_verda_needs_a_tunnel(self):
        """No open ports, and the whole design is a server others connect to."""
        assert credentials.PROVIDERS["verda"].open_ports is False
        assert "TUNNEL" in credentials.report()

    def test_neither_broker_nor_verda_can_autostop(self):
        """Without autostop a crashed host bills until someone notices."""
        assert credentials.PROVIDERS["primeintellect"].autostop is False
        assert credentials.PROVIDERS["verda"].autostop is False

    def test_runpod_is_the_only_cheap_one_with_all_three(self):
        rp = credentials.PROVIDERS["runpod"]
        assert (rp.spot, rp.autostop, rp.open_ports) == (True, True, True)
