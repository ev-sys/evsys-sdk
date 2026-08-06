"""Nebius AI Cloud as a capacity provider.

Everything here is against Nebius's **REST gateway** — the HTTP/1.1+JSON
translation of their gRPC API, generated from the same protos
(https://docs.nebius.com/rest-api/): ``https://api.nebius.cloud/{service}/v1/…``.

What is different from Verda, learned from the docs before writing a line:

  * **Auth is a two-hop**: mint a short-lived RS256 JWT (iss=sub=service
    account id, kid=authorized public key id), exchange it at
    ``https://auth.eu.nebius.com/oauth2/token/exchange`` (RFC 8693 form body)
    for a 12-hour bearer token. No client_id/secret pair.
  * **SKUs are platform/preset pairs**, e.g. ``gpu-h100-sxm`` +
    ``8gpu-128vcpu-1600gb``. Prices are flat per GPU-hour per platform and
    published, not returned by an offers endpoint — the catalog below carries
    them.
  * **Availability is a real API** — the *capacity advisor*
    (``ResourceAdviceService``) reports, per (region, platform, preset),
    how many on-demand and preemptible VMs the tenant could launch right now
    plus a confidence level. Far stronger than Verda's boolean.
  * **Preemption = STOP, not delete.** A preempted VM keeps all attached
    disks; only dynamic IPs are lost. Spot VMs are created with
    ``recovery_policy: FAIL`` + ``preemptible: {on_preemption: STOP}``.

Transport is injected the same way as every provider here, so all of this is
testable without an account. Nothing in this module has been run against a
live Nebius tenant yet — the shapes follow docs.nebius.com and the published
protos (github.com/nebius/api), and the paths are constants for easy
correction on first live contact.
"""

from __future__ import annotations

import base64
import json
import pathlib
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from ..logger import get_logger
from .provider import LaunchFailed

log = get_logger(__name__)

API = "https://api.nebius.cloud"
TOKEN_URL = "https://auth.eu.nebius.com/oauth2/token/exchange"
CREDENTIALS = "~/.nebius/credentials.json"

#: Ubuntu 24.04 + CUDA 12 — supported on the Hopper/L40S platforms this
#: project rents (Blackwell platforms require the cuda13 family).
DEFAULT_IMAGE_FAMILY = "ubuntu24.04-cuda12"

#: The published catalog (docs.nebius.com/compute/virtual-machines/types +
#: /compute/resources/pricing, 2026-08). Prices are USD per GPU-hour and flat
#: across counts; regions are where the platform is sold. The capacity
#: advisor is the live truth — this table names what to ask it about.
PLATFORMS: dict[str, dict[str, Any]] = {
    "gpu-h100-sxm": {
        "gpu": "H100", "usd_gpu_hr": 3.85, "usd_gpu_hr_spot": 2.15,
        "regions": ["eu-north1"],
        "presets": {1: "1gpu-16vcpu-200gb", 8: "8gpu-128vcpu-1600gb"},
    },
    "gpu-h200-sxm": {
        "gpu": "H200", "usd_gpu_hr": 4.50, "usd_gpu_hr_spot": 2.45,
        "regions": ["eu-north1", "eu-west1", "us-central1"],
        "presets": {1: "1gpu-16vcpu-200gb", 8: "8gpu-128vcpu-1600gb"},
    },
    "gpu-b200-sxm": {
        "gpu": "B200", "usd_gpu_hr": 7.15, "usd_gpu_hr_spot": 3.95,
        "regions": ["eu-north1", "us-central1"],
        "presets": {1: "1gpu-20vcpu-224gb", 8: "8gpu-160vcpu-1792gb"},
    },
    "gpu-l40s-a": {
        "gpu": "L40S", "usd_gpu_hr": 1.35, "usd_gpu_hr_spot": 0.65,
        "regions": ["eu-north1"],
        "presets": {1: "1gpu-16vcpu-64gb"},
    },
}


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def make_jwt(service_account_id: str, public_key_id: str,
             private_key_pem: str, *, lifetime_s: int = 300) -> str:
    """The 5-minute RS256 JWT Nebius exchanges for a real access token.

    Implemented directly over ``cryptography`` (already a transitive dep)
    rather than pulling in PyJWT for one token shape.
    """
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import padding

    now = int(time.time())
    header = {"alg": "RS256", "typ": "JWT", "kid": public_key_id}
    claims = {"iss": service_account_id, "sub": service_account_id,
              "iat": now, "exp": now + lifetime_s}
    signing_input = (_b64url(json.dumps(header, separators=(",", ":")).encode())
                     + "." +
                     _b64url(json.dumps(claims, separators=(",", ":")).encode()))
    key = serialization.load_pem_private_key(private_key_pem.encode(),
                                             password=None)
    sig = key.sign(signing_input.encode(), padding.PKCS1v15(),
                   hashes.SHA256())
    return f"{signing_input}.{_b64url(sig)}"


class NebiusProvider:
    """Credentials, token exchange, and the REST transport (``_call``)."""

    name = "nebius"

    def __init__(self, credentials: str = CREDENTIALS):
        self.credentials = credentials
        self._cfg: dict[str, str] | None = None
        self._tok: str = ""
        self._tok_exp: float = 0.0

    # -- config ------------------------------------------------------------

    @property
    def cfg(self) -> dict[str, str]:
        if self._cfg is None:
            p = pathlib.Path(self.credentials).expanduser()
            self._cfg = json.loads(p.read_text())
        return self._cfg

    @property
    def project_id(self) -> str:
        return self.cfg.get("project_id", "")

    @property
    def tenant_id(self) -> str:
        return self.cfg.get("tenant_id", "")

    # -- transport ---------------------------------------------------------

    def _token(self) -> str:
        """Bearer token via JWT exchange, cached (12h lifetime, wide margin)."""
        if self._tok and time.time() < self._tok_exp:
            return self._tok
        jwt = make_jwt(self.cfg["service_account_id"],
                       self.cfg["public_key_id"], self.cfg["private_key"])
        body = urllib.parse.urlencode({
            "grant_type": "urn:ietf:params:oauth:grant-type:token-exchange",
            "requested_token_type":
                "urn:ietf:params:oauth:token-type:access_token",
            "subject_token": jwt,
            "subject_token_type": "urn:ietf:params:oauth:token-type:jwt",
        }).encode()
        req = urllib.request.Request(
            TOKEN_URL, data=body,
            headers={"Content-Type": "application/x-www-form-urlencoded"})
        with urllib.request.urlopen(req, timeout=30) as r:
            d = json.load(r)
        self._tok = d["access_token"]
        self._tok_exp = time.time() + max(int(d.get("expires_in", 43200)) - 300,
                                          60)
        return self._tok

    def _call(self, path: str, body: Any = None, method: str | None = None) -> Any:
        """``path`` is relative to the REST gateway, e.g.
        ``compute/v1/instances``. GET with params goes in the path itself."""
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(
            f"{API}/{path}", data=data, method=method,
            headers={"Authorization": f"Bearer {self._token()}",
                     "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                raw = resp.read().decode().strip()
        except urllib.error.HTTPError as e:
            detail = e.read().decode()[:300]
            raise LaunchFailed(f"{e.code} {detail}", self._reason(e.code, detail))
        return json.loads(raw) if raw else {}

    @staticmethod
    def _reason(code: int, detail: str) -> str:
        d = detail.lower()
        if "quota" in d or "capacity" in d or "resource_exhausted" in d:
            return "no_capacity"
        if "permission" in d or code in (401, 403):
            return "no_auth"
        return f"http_{code}"

    # -- capacity ----------------------------------------------------------

    def advice(self) -> list[dict]:
        """The capacity advisor: per (region, platform, preset), how many
        on-demand and preemptible VMs this tenant could launch right now."""
        out: list[dict] = []
        token = ""
        while True:
            q = f"capacity/v1/resourceAdvices?parentId={self.tenant_id}"
            if token:
                q += f"&pageToken={urllib.parse.quote(token)}"
            page = self._call(q) or {}
            out.extend(page.get("items") or [])
            token = page.get("nextPageToken") or ""
            if not token:
                return out


__all__ = ["API", "CREDENTIALS", "DEFAULT_IMAGE_FAMILY", "PLATFORMS",
           "NebiusProvider", "TOKEN_URL", "make_jwt"]
