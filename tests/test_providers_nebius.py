"""Tests for the Nebius provider: JWT shape, transport plumbing, and the
capacity-advisor availability probe — all against fakes, no tenant."""
import base64
import json

import pytest

from evsys_sdk.compute import availability as av
from evsys_sdk.compute.providers_nebius import (PLATFORMS, NebiusProvider,
                                                make_jwt)


def _rsa_pem():
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pem = key.private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption()).decode()
    return key, pem


def _b64pad(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def test_make_jwt_shape_and_signature():
    key, pem = _rsa_pem()
    tok = make_jwt("serviceaccount-e00x", "publickey-e00y", pem)
    head_b64, claims_b64, sig_b64 = tok.split(".")
    header = json.loads(_b64pad(head_b64))
    claims = json.loads(_b64pad(claims_b64))
    assert header == {"alg": "RS256", "typ": "JWT", "kid": "publickey-e00y"}
    assert claims["iss"] == claims["sub"] == "serviceaccount-e00x"
    assert claims["exp"] - claims["iat"] == 300     # the documented 5 minutes
    # signature verifies against the public key
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import padding
    key.public_key().verify(_b64pad(sig_b64),
                            f"{head_b64}.{claims_b64}".encode(),
                            padding.PKCS1v15(), hashes.SHA256())


def test_reason_mapping():
    assert NebiusProvider._reason(429, "quota exceeded") == "no_capacity"
    assert NebiusProvider._reason(403, "permission denied") == "no_auth"
    assert NebiusProvider._reason(500, "boom") == "http_500"


def test_advice_paginates(monkeypatch, tmp_path):
    creds = tmp_path / "credentials.json"
    creds.write_text(json.dumps({"tenant_id": "tenant-1"}))
    p = NebiusProvider(credentials=str(creds))
    pages = {
        "": {"items": [{"a": 1}], "nextPageToken": "t2"},
        "t2": {"items": [{"a": 2}]},
    }
    calls = []

    def fake_call(path, body=None, method=None):
        calls.append(path)
        tok = path.split("pageToken=")[1] if "pageToken=" in path else ""
        return pages[tok]

    p._call = fake_call
    items = p.advice()
    assert items == [{"a": 1}, {"a": 2}]
    assert all("parentId=tenant-1" in c for c in calls)


def _advice_item(platform="gpu-h100-sxm", preset="8gpu-128vcpu-1600gb",
                 region="eu-north1", gpus=8, pre_avail=22, od_avail=24,
                 pre_level="AVAILABILITY_LEVEL_MEDIUM",
                 od_level="AVAILABILITY_LEVEL_HIGH"):
    return {
        "spec": {"region": region,
                 "computeInstance": {
                     "platform": platform,
                     "preset": {"name": preset,
                                "resources": {"gpuCount": gpus}}}},
        "status": {
            "onDemand": {"available": od_avail, "limit": 32,
                         "availabilityLevel": od_level,
                         "dataState": "DATA_STATE_FRESH"},
            "preemptible": {"available": pre_avail, "limit": 128,
                            "availabilityLevel": pre_level,
                            "dataState": "DATA_STATE_FRESH"},
        },
    }


@pytest.fixture
def probe(monkeypatch):
    monkeypatch.setattr(NebiusProvider, "advice",
                        lambda self: [_advice_item()])
    return av.NebiusAvailability()


def test_probe_maps_advisor_to_capacity(probe):
    caps = probe.check("H100", 8)
    spot = [c for c in caps if c.spot]
    od = [c for c in caps if not c.spot]
    assert spot and od
    s = spot[0]
    assert s.ok and s.provider == "nebius"
    assert s.sku == "gpu-h100-sxm/8gpu-128vcpu-1600gb"
    assert s.region == "eu-north1"
    # price is per-GPU-hour x count, from the published catalog
    assert s.usd_hr == pytest.approx(
        PLATFORMS["gpu-h100-sxm"]["usd_gpu_hr_spot"] * 8)
    assert od[0].usd_hr == pytest.approx(
        PLATFORMS["gpu-h100-sxm"]["usd_gpu_hr"] * 8)
    assert "22 launchable" in s.detail


def test_probe_limit_reached_is_unavailable(monkeypatch):
    monkeypatch.setattr(
        NebiusProvider, "advice",
        lambda self: [_advice_item(
            pre_avail=0, pre_level="AVAILABILITY_LEVEL_LIMIT_REACHED")])
    caps = av.NebiusAvailability().check("H100", 8, spot=True)
    assert caps[0].state == av.UNAVAILABLE


def test_probe_filters_gpu_count_and_region(probe):
    assert all(c.state == av.UNAVAILABLE          # "no matching offers"
               for c in probe.check("H200", 8))
    assert all(c.state == av.UNAVAILABLE
               for c in probe.check("H100", 1))
    assert probe.check("H100", 8, region="eu-north1")[0].ok
    assert all(c.state == av.UNAVAILABLE
               for c in probe.check("H100", 8, region="us-central1"))


def test_scan_includes_nebius():
    assert "nebius" in av.clouds()
