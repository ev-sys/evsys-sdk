"""Renting a machine, as a contract a provider can implement.

SkyPilot factors this well — every cloud implements ``run_instances``,
``wait_instances``, ``terminate_instances``, ``query_instances`` and the
generic provisioner drives them. We need a narrower version of the same idea:
no multi-node, no port management, no storage mounting, just *get me one
machine with a working GPU and let me put work on it*.

Why not simply use SkyPilot: its catalog is a pre-generated CSV that was wrong
about PrimeIntellect's H100 price by 65% and wrong about stock existing at all,
and its errors flatten three unrelated failures — no capacity, no funds,
unsupported feature — into one ``ResourcesUnavailableError``. Those are
different problems with different fixes, and hours went into telling them apart
by hand. This layer keeps them distinct.

The contract is five verbs::

    offers()      what can I buy right now, and for how much
    launch()      provision one machine, return a handle
    poll()        where is it: pending / running / gone
    terminate()   release it AND its disks
    orphans()     disks left behind by machines that died

A provider is roughly 100 lines. Everything above it — health gating, file
transfer, retries, teardown guarantees — is written once here.

Three behaviours are non-negotiable, each because getting it wrong cost real
money today:

  * **Availability must be authoritative per SKU.** Aggregate listings answer a
    different question (on-demand stock) and disagreed with spot deploys all
    afternoon.
  * **A host is not usable until CUDA initialises.** Two machines reported
    healthy GPUs through ``nvidia-smi`` while ``cuInit`` returned 802 on a dead
    NVSwitch fabric. Installing 27 GB onto one costs ten minutes.
  * **Terminating an instance does not release its disk.** Orphaned volumes
    bill at $0.08/hr each and accumulate silently with every preemption.
"""

from __future__ import annotations

import abc
import atexit
import os
import signal
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any, Iterable

from ..logger import get_logger
from . import reliability as rel

log = get_logger(__name__)

PENDING, RUNNING, GONE = "pending", "running", "gone"

#: Probe that decides whether a machine can actually run work. Deliberately
#: `cuInit`, not `nvidia-smi`: the latter lists GPUs on a host whose fabric
#: manager has aborted, and reports them as healthy.
CUDA_PROBE = (
    'python3 -c "'
    "import ctypes;cu=ctypes.CDLL('libcuda.so.1');rc=cu.cuInit(0);"
    "n=ctypes.c_int();cu.cuDeviceGetCount(ctypes.byref(n));"
    "print('USABLE' if rc==0 and n.value>0 else 'BROKEN')\""
)


@dataclass(frozen=True)
class Offer:
    """Something purchasable right now."""

    sku: str
    region: str
    gpu: str
    count: int
    usd_hr: float
    spot: bool

    @property
    def usd_per_gpu_hr(self) -> float:
        return self.usd_hr / max(self.count, 1)


@dataclass
class Machine:
    """A rented machine. ``ip`` is None until it reaches RUNNING."""

    id: str
    provider: str
    sku: str
    region: str
    gpu: str
    count: int
    usd_hr: float
    ip: str | None = None
    state: str = PENDING
    launched_at: float = field(default_factory=time.time)

    @property
    def uptime_s(self) -> float:
        return time.time() - self.launched_at


class LaunchFailed(RuntimeError):
    """Provisioning was refused. ``reason`` distinguishes the causes that
    matter: no_capacity, no_funds, unsupported, unknown. They are not
    interchangeable — capacity says something about the provider, an empty
    wallet says nothing at all."""

    def __init__(self, message: str, reason: str = "unknown"):
        super().__init__(message)
        self.reason = reason


class Provider(abc.ABC):
    """What a cloud must implement. Five methods."""

    name: str = ""

    @abc.abstractmethod
    def offers(self, gpu: str | None = None, count: int = 1,
               spot: bool | None = None) -> list[Offer]:
        """Purchasable options now, cheapest per GPU first."""

    @abc.abstractmethod
    def available(self, sku: str, region: str, spot: bool = True) -> bool:
        """Authoritative yes/no for ONE sku in ONE region.

        Must not be inferred from a listing. Providers answer this differently
        from "what exists", and the difference is where launches fail.
        """

    @abc.abstractmethod
    def launch(self, sku: str, region: str, *, spot: bool = True,
               name: str = "evsys") -> Machine:
        """Provision one machine. Raise :class:`LaunchFailed` with a reason."""

    @abc.abstractmethod
    def poll(self, machine: Machine) -> Machine:
        """Refresh state and ip in place; return the same object."""

    @abc.abstractmethod
    def terminate(self, machine: Machine) -> None:
        """Release the machine AND any disk it created. Never raises."""

    def orphans(self) -> list[tuple[str, str, float]]:
        """Detached disks still billing: (id, name, usd_month). Optional."""
        return []


# -- the generic layer, written once ----------------------------------------


def ssh(ip: str, cmd: str, key: str, timeout: float = 300) -> tuple[int, str]:
    """Run a command. Returns (rc, stdout). rc 255 means unreachable."""
    try:
        p = subprocess.run(
            ["ssh", "-i", key, "-o", "StrictHostKeyChecking=no",
             "-o", "UserKnownHostsFile=/dev/null", "-o", "ConnectTimeout=15",
             "-o", "BatchMode=yes", f"root@{ip}", cmd],
            capture_output=True, text=True, timeout=timeout)
        return p.returncode, p.stdout
    except subprocess.TimeoutExpired:
        return 255, ""
    except Exception:
        return 255, ""


def push(ip: str, files: Iterable[str], key: str, dest: str = "/root/") -> bool:
    """Copy files to a host. False on any failure.

    scp is atomic across its argument list: one missing local path fails the
    whole transfer. That is indistinguishable from a network problem in the
    return value, and it cost three acquire/teardown cycles - each one renting
    a GPU, passing the CUDA gate, failing the copy and releasing - before the
    cause was found. So check locally first and name the file.
    """
    files = list(files)
    if not files:
        return True
    absent = [f for f in files if not os.path.exists(f)]
    if absent:
        log.error("[provider] refusing to scp: %d local file(s) missing: %s",
                  len(absent), ", ".join(absent))
        return False
    try:
        p = subprocess.run(
            ["scp", "-i", key, "-o", "StrictHostKeyChecking=no",
             "-o", "UserKnownHostsFile=/dev/null", "-o", "ConnectTimeout=15",
             "-q", *files, f"root@{ip}:{dest}"],
            capture_output=True, text=True, timeout=600)
        return p.returncode == 0
    except Exception:
        return False


def pull(ip: str, remote: str, local: str, key: str) -> bool:
    try:
        p = subprocess.run(
            ["scp", "-i", key, "-o", "StrictHostKeyChecking=no",
             "-o", "UserKnownHostsFile=/dev/null", "-o", "ConnectTimeout=12",
             "-q", f"root@{ip}:{remote}", local],
            capture_output=True, text=True, timeout=300)
        return p.returncode == 0
    except Exception:
        return False


def wait_running(provider: Provider, machine: Machine, *, timeout_s: float = 600,
                 poll_s: float = 15) -> Machine:
    """Block until the machine has an ip, or raise."""
    end = time.time() + timeout_s
    while time.time() < end:
        provider.poll(machine)
        if machine.state == RUNNING and machine.ip:
            return machine
        if machine.state == GONE:
            raise LaunchFailed(f"{machine.id} disappeared before running", "gone")
        time.sleep(poll_s)
    raise LaunchFailed(f"{machine.id} never reached running", "timeout")


def wait_usable(ip: str, key: str, *, timeout_s: float = 300,
                poll_s: float = 12) -> bool:
    """True once CUDA initialises. False if the fabric is dead.

    This is the gate that separates "provisioned" from "usable". Skipping it
    means installing 27 GB onto a machine that cannot run a kernel.
    """
    end = time.time() + timeout_s
    while time.time() < end:
        rc, out = ssh(ip, CUDA_PROBE, key, timeout=45)
        if "USABLE" in out:
            return True
        if "BROKEN" in out:
            return False
        time.sleep(poll_s)
    return False


def acquire(provider: Provider, wants: list[tuple[str, str]], key: str, *,
            spot: bool = True, name: str = "evsys",
            attempts: int | None = None, wait_s: float = 45,
            gpu_of: dict[str, str] | None = None) -> Machine | None:
    """Keep trying until a machine is provisioned AND passes the CUDA gate.

    ``wants`` is (sku, region) in preference order. Availability listings lag
    reality — every SKU reported free was refused seconds later, repeatedly —
    so the listing only orders the attempts; the launch call is the authority.

    A machine that fails the gate is released and the search continues, because
    dead-on-arrival is common enough that stopping on it wastes the run. Every
    outcome is recorded to the reliability ledger.
    """
    n = 0
    while attempts is None or n < attempts:
        n += 1
        for sku, region in wants:
            try:
                if not provider.available(sku, region, spot=spot):
                    continue
            except Exception as e:
                log.debug("[provider] availability check failed for %s/%s: %s",
                          sku, region, e)
                continue
            try:
                m = provider.launch(sku, region, spot=spot, name=name)
            except LaunchFailed as e:
                rel.record(rel.LAUNCH_FAIL, provider=provider.name,
                           gpu=(gpu_of or {}).get(sku, sku), region=region,
                           reason=e.reason)
                continue
            try:
                wait_running(provider, m)
            except LaunchFailed:
                provider.terminate(m)
                continue
            if wait_usable(m.ip or "", key):
                rel.record(rel.LAUNCH_OK, provider=provider.name, gpu=m.gpu,
                           count=m.count, region=region, usd_hr=m.usd_hr,
                           wait_s=m.uptime_s)
                log.info("[provider] acquired %s %s in %s at $%.4f/hr",
                         m.sku, m.ip, region, m.usd_hr)
                return m
            # Provisioned, billed, and unusable — its own outcome.
            rel.record(rel.DOA, provider=provider.name, gpu=m.gpu, count=m.count,
                       region=region, usd_hr=m.usd_hr, reason="cuda_init_failed")
            log.warning("[provider] %s is dead on arrival — releasing", m.ip)
            provider.terminate(m)
        time.sleep(wait_s)
    return None


class Session:
    """A machine you cannot forget to release.

    Teardown runs on exit whatever happens, and records whether the machine was
    torn down or had already vanished — the difference matters, because only
    real preemptions inform the MTBF that sets snapshot cadence.
    """

    def __init__(self, provider: Provider, machine: Machine, key: str):
        self.provider, self.machine, self.key = provider, machine, key
        self._released = False
        # `with` only protects against exceptions and clean returns. A SIGTERM
        # - which is what `kill <launcher>` sends - unwinds nothing, so the
        # machine keeps billing with no process left to release it. That
        # happened: killing two launchers left an H200 and 7 volumes running,
        # ~$360/mo of storage alone. Register a last-resort release.
        atexit.register(self._release)
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                prev = signal.getsignal(sig)
                signal.signal(sig, self._on_signal(sig, prev))
            except (ValueError, OSError):
                pass       # not the main thread; atexit still covers us

    def _on_signal(self, sig, prev):
        def handler(signum, frame):
            log.warning("[provider] signal %s - releasing %s before exit",
                        signum, self.machine.id[:8])
            self._release()
            if callable(prev) and prev not in (signal.SIG_IGN, signal.SIG_DFL):
                prev(signum, frame)
            else:
                raise SystemExit(128 + signum)
        return handler

    def _release(self) -> None:
        """Idempotent teardown. Safe to call from __exit__, a signal, or atexit."""
        if self._released:
            return
        self._released = True
        try:
            self.provider.poll(self.machine)
        except Exception:
            pass
        gone = self.machine.state == GONE
        try:
            rel.record(rel.PREEMPTED if gone else rel.TORN_DOWN,
                       provider=self.provider.name, gpu=self.machine.gpu,
                       count=self.machine.count, region=self.machine.region,
                       uptime_s=self.machine.uptime_s, usd_hr=self.machine.usd_hr)
        except Exception:
            pass
        if not gone:
            try:
                self.provider.terminate(self.machine)
            except Exception as e:  # noqa: BLE001
                log.error("[provider] COULD NOT RELEASE %s: %s - it is still "
                          "billing", self.machine.id, e)

    def __enter__(self) -> Session:
        return self

    def run(self, cmd: str, timeout: float = 1800) -> tuple[int, str]:
        return ssh(self.machine.ip or "", cmd, self.key, timeout=timeout)

    def push(self, *files: str) -> bool:
        return push(self.machine.ip or "", files, self.key)

    def pull(self, remote: str, local: str) -> bool:
        return pull(self.machine.ip or "", remote, local, self.key)

    def alive(self) -> bool:
        self.provider.poll(self.machine)
        return self.machine.state == RUNNING

    def __exit__(self, *exc: Any) -> None:
        self._release()
        # Disks outlive their machines on every provider tested. Sweep even
        # when the machine vanished on its own, because that is exactly when
        # nothing else will.
        for vid, vname, cost in self.provider.orphans():
            log.info("[provider] releasing orphan disk %s (%s, $%.0f/mo)",
                     vname, vid[:8], cost)


__all__ = ["CUDA_PROBE", "GONE", "LaunchFailed", "Machine", "Offer", "PENDING",
           "Provider", "RUNNING", "Session", "acquire", "pull", "push", "ssh",
           "wait_running", "wait_usable"]
