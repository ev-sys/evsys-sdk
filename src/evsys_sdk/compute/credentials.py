"""Provider credentials, in one place, in one format.

Renting a GPU means having an account with whoever currently has one spare,
and "whoever" changes hour to hour — the spot A100 this project benchmarked on
stopped being offered *during* the run. One provider is not a supply chain, it
is a single point of failure, so the practical answer is to be authenticated
with several and let the launcher pick.

The obstacle is that each provider invents its own credential file: JSON here,
TOML there, a bare token in a file somewhere else, OAuth client pairs for the
one that used to be a different company. SkyPilot reads each in its own native
location, so this module does not replace any of that — it *writes* those
files from one uniform description, and reports which providers are actually
usable. One JSON in, whatever the provider wanted out.

    from evsys_sdk.compute import credentials
    credentials.save("verda", client_id="…", client_secret="…")
    credentials.status()      # -> {"verda": True, "primeintellect": True, …}

Files are written ``0600``. These are keys that rent GPUs by the hour on
someone's credit card; the umask default of world-readable is not appropriate.

**On brokers.** PrimeIntellect resells other people's hardware — the offers it
returns name datacrunch, lambdalabs, nebius, hyperstack and massedcompute as
providers. Authenticating directly with those means both a shorter price chain
and access to inventory the broker is not currently listing, which is exactly
the gap you notice when one shows an A100 and the other does not.
"""

from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass

from ..logger import get_logger

log = get_logger(__name__)

CREDENTIAL_MODE = 0o600


@dataclass(frozen=True)
class Provider:
    """How one provider wants to be told who you are."""

    name: str
    path: str
    fields: tuple[str, ...]
    fmt: str = "json"
    spot: bool = False
    autostop: bool = False
    open_ports: bool = True
    note: str = ""

    def file(self) -> pathlib.Path:
        return pathlib.Path(self.path).expanduser()

    def authenticated(self) -> bool:
        p = self.file()
        if not p.exists():
            return False
        if self.fmt != "json":
            return bool(p.read_text().strip())
        try:
            data = json.loads(p.read_text())
        except Exception:
            return False
        return all(data.get(f) for f in self.fields)


#: Capability flags are SkyPilot's own, read from each cloud's
#: ``_CLOUD_UNSUPPORTED_FEATURES``. They are not cosmetic:
#:
#:   * no ``spot`` means no cheap capacity, which is the entire cost argument;
#:   * no ``autostop`` means a crashed host leaves a GPU billing until someone
#:     notices (see ``max_lifetime_s`` on the SkyPilot target);
#:   * no ``open_ports`` means a served port cannot be exposed directly and
#:     needs an SSH tunnel — which matters here, because the whole design is a
#:     training *server* other processes connect to.
PROVIDERS: dict[str, Provider] = {
    p.name: p for p in [
        Provider("primeintellect", "~/.prime/config.json", ("api_key",),
                 spot=True, autostop=False, open_ports=True,
                 note="broker: resells datacrunch/lambdalabs/nebius/hyperstack/"
                      "massedcompute. No balance endpoint — an empty wallet "
                      "surfaces only as 'resources unavailable'."),
        Provider("verda", "~/.verda/config.json", ("client_id", "client_secret"),
                 spot=True, autostop=False, open_ports=False,
                 note="OAuth2 client credentials. Has /v1/balance, which "
                      "PrimeIntellect does not. No open ports: reach the "
                      "server over an SSH tunnel."),
        Provider("runpod", "~/.runpod/config.toml", ("api_key",), fmt="toml",
                 spot=True, autostop=True, open_ports=True,
                 note="Spot plus autostop plus open ports — the only one of "
                      "the cheap providers with all three."),
        Provider("vast", "~/.config/vastai/vast_api_key", ("api_key",), fmt="raw",
                 spot=True, autostop=True, open_ports=True,
                 note="Marketplace of individual hosts; reliability varies "
                      "by host, not by region."),
        Provider("nebius", "~/.nebius/credentials.json",
                 ("tenant_id", "service_account_id", "private_key"),
                 spot=True, autostop=True, open_ports=True,
                 note="Also appears inside PrimeIntellect's offers."),
        Provider("lambda", "~/.lambda_cloud/lambda_keys", ("api_key",), fmt="raw",
                 spot=False, autostop=True, open_ports=True,
                 note="On-demand only."),
        Provider("hyperbolic", "~/.hyperbolic/api_key", ("api_key",), fmt="raw",
                 spot=False, autostop=False, open_ports=False),
        Provider("fluidstack", "~/.fluidstack/api_key", ("api_key",), fmt="raw",
                 spot=False, autostop=True, open_ports=True),
        Provider("shadeform", "~/.shadeform/api_key", ("api_key",), fmt="raw",
                 spot=False, autostop=True, open_ports=True,
                 note="Itself an aggregator across other clouds."),
    ]
}


def save(provider: str, **fields: str) -> pathlib.Path:
    """Write ``fields`` where ``provider`` expects them, at mode 0600.

    Raises on an unknown provider or a missing required field rather than
    writing a file that will fail opaquely at launch — a half-written
    credential reads exactly like a capacity shortage, hours later.
    """
    try:
        spec = PROVIDERS[provider]
    except KeyError:
        raise KeyError(f"unknown provider {provider!r}; known: "
                       f"{', '.join(sorted(PROVIDERS))}") from None
    missing = [f for f in spec.fields if not fields.get(f)]
    if missing:
        raise ValueError(f"{provider} needs {', '.join(missing)}")

    path = spec.file()
    path.parent.mkdir(parents=True, exist_ok=True)
    if spec.fmt == "json":
        body = json.dumps({f: fields[f] for f in spec.fields}, indent=2) + "\n"
    elif spec.fmt == "toml":
        body = "".join(f'{f} = "{fields[f]}"\n' for f in spec.fields)
    else:
        body = fields[spec.fields[0]].strip() + "\n"
    # Create with the right mode rather than fixing it afterwards: a
    # world-readable moment is still a leak on a shared machine.
    path.touch(mode=CREDENTIAL_MODE, exist_ok=True)
    path.chmod(CREDENTIAL_MODE)
    path.write_text(body)
    log.info("[credentials] wrote %s for %s", path, provider)
    return path


def status() -> dict[str, bool]:
    """Which providers are authenticated, by name."""
    return {n: p.authenticated() for n, p in sorted(PROVIDERS.items())}


def authenticated(spot: bool | None = None) -> list[str]:
    """Authenticated providers, optionally only those offering spot."""
    return [n for n, p in sorted(PROVIDERS.items())
            if p.authenticated() and (spot is None or p.spot == spot)]


def report() -> str:
    """A table of who we can rent from, and what they can do."""
    lines = [f"{'provider':<16} {'auth':<6} {'spot':<6} {'autostop':<9} ports"]
    for name, p in sorted(PROVIDERS.items()):
        lines.append(f"{name:<16} {'yes' if p.authenticated() else 'no':<6} "
                     f"{'yes' if p.spot else 'no':<6} "
                     f"{'yes' if p.autostop else 'no':<9} "
                     f"{'yes' if p.open_ports else 'TUNNEL'}")
    return "\n".join(lines)


__all__ = ["PROVIDERS", "Provider", "authenticated", "report", "save", "status"]
