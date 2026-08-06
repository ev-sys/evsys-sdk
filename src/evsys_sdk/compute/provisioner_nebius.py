"""Nebius's side of the router's Provisioner contract.

Structurally simpler than Verda's, because Nebius's API removes the two
hazards the Verda dance exists for:

  * **Disks attach at create time.** ``secondary_disks`` rides in the
    instance spec, so there is no shutdown → attach → boot sequence and no
    @reboot re-entry: cloud-init runs on the first (only) boot with the data
    disk already present.
  * **The device path is deterministic.** ``device_id`` on the attachment
    surfaces as ``/dev/disk/by-id/virtio-<device_id>`` in the guest — no
    lsblk guessing. The agent still mounts before mkfs, because a ``reuse``
    plan attaches a disk holding the very checkpoints it came to recover.
  * **Preemption = STOP, not delete** (docs: preemptible VMs). A preempted
    instance keeps its disks and turns up as ``STOPPED`` — which ``alive()``
    reports as dead so the router requeues; the surviving data disk then
    feeds a ``reuse`` plan on the replacement node.

The instance is created with ``recovery_policy: FAIL`` +
``preemptible: {on_preemption: STOP}`` when spot is requested — the only
combination Nebius allows for preemptible VMs.

`sku` here is the availability probe's ``platform/preset`` pair (e.g.
``gpu-h100-sxm/8gpu-128vcpu-1600gb``).

Transport is injected (``call``) like every provider; ``None`` builds the
real authenticated NebiusProvider transport. Not yet exercised against a
live tenant — shapes follow docs.nebius.com and github.com/nebius/api.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict

from ..logger import get_logger
from ..registry import register_provisioner
from . import availability as av
from .job_router import NodeHandle, VolumePlan
from .queue import Job

log = get_logger(__name__)

#: Stable guest device path for the data disk (virtio device_id).
DATA_DEVICE_ID = "evsysdata"

#: cloud-init user data (max 32 KiB at the API). A bare ``#!`` script is a
#: valid cloud-init payload; it runs once, as root, on first boot — with the
#: data disk already attached, so no reboot persistence is needed.
AGENT_TEMPLATE = """#!/bin/bash
exec > /var/log/evsys_agent.log 2>&1
set -x
DEV=/dev/disk/by-id/virtio-{device_id}
for i in $(seq 1 60); do [ -e "$DEV" ] && break; sleep 5; done
mkdir -p /data
# mount-first: mkfs only if the disk has no filesystem (reuse-safe)
if ! mount "$DEV" /data 2>/dev/null; then
  mkfs.ext4 -F "$DEV" && mount "$DEV" /data
fi
export EVSYS_SNAPSHOT_INTERVAL_S={interval}
export EVSYS_JOB_ID={job_id}
export EVSYS_RESUME_STEP={resume_step}
export EVSYS_CONFIG={config}
export EVSYS_STORE_DIR=/data/store
export EVSYS_PROVIDER=nebius
export EVSYS_VOLUME={volume}
export EVSYS_EVENTS_URL={events_url}
{payload}
"""


class NebiusProvisionerConfig(BaseModel):
    """The ``{kind: nebius, params: {...}}`` surface."""

    model_config = ConfigDict(extra="forbid")

    payload: str = "echo agent-payload-not-configured"
    project_id: str = ""
    """Parent for every resource. Empty -> the credentials file's."""
    subnet_id: str = ""
    """VPC subnet for the instance NIC. Empty -> first subnet in the
    project (looked up once at first provision)."""
    image_family: str = ""
    """Boot image family. Empty -> the platform-appropriate default."""
    volume_gb: int = 300
    boot_disk_gb: int = 200
    events_url: str = ""


@register_provisioner("nebius")
class NebiusProvisioner:
    """Implements the router's Provisioner protocol against Nebius's REST API."""

    name: ClassVar[str] = "nebius"
    Config: ClassVar[type] = NebiusProvisionerConfig

    def __init__(self, call: Callable[..., Any] | None = None,
                 payload: str = "echo agent-payload-not-configured",
                 project_id: str = "", subnet_id: str = "",
                 image_family: str = "", volume_gb: int = 300,
                 boot_disk_gb: int = 200, events_url: str = ""):
        if call is None:
            from .providers_nebius import NebiusProvider
            provider = NebiusProvider()
            call = provider._call
            project_id = project_id or provider.project_id
        self.call = call
        self.project_id = project_id
        self.payload = payload
        self.subnet_id = subnet_id
        self.image_family = image_family
        self.volume_gb = volume_gb
        self.boot_disk_gb = boot_disk_gb
        self.events_url = events_url

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def _resource_id(op: Any) -> str:
        """Operations carry the created resource's id; be liberal about
        where (proto3 JSON camelCase vs snake_case)."""
        if isinstance(op, str):
            return op
        for k in ("resourceId", "resource_id", "id"):
            if isinstance(op, dict) and op.get(k):
                return str(op[k])
        meta = (op or {}).get("metadata") or {}
        return str(meta.get("resourceId") or meta.get("resource_id") or "")

    def _subnet(self) -> str:
        if self.subnet_id:
            return self.subnet_id
        subs = self.call(f"vpc/v1/subnets?parentId={self.project_id}") or {}
        items = subs.get("items") or []
        if not items:
            raise RuntimeError(f"no subnets in project {self.project_id}")
        self.subnet_id = items[0]["metadata"]["id"]
        return self.subnet_id

    def _ensure_disk(self, job: Job, plan: VolumePlan) -> str:
        if plan.mode == "reuse":
            return plan.volume
        op = self.call("compute/v1/disks", {
            "metadata": {"parentId": self.project_id,
                         "name": f"evsys-{job.id}"},
            "spec": {"sizeGibibytes": self.volume_gb,
                     "type": "NETWORK_SSD"}}, "POST")
        return self._resource_id(op)

    def _script(self, job: Job, plan: VolumePlan, interval_s: float,
                volume: str) -> str:
        resume = plan.checkpoint.step if plan.checkpoint else 0
        return AGENT_TEMPLATE.format(
            device_id=DATA_DEVICE_ID, interval=int(interval_s),
            job_id=job.id, resume_step=resume, config=job.config,
            volume=volume, events_url=self.events_url, payload=self.payload)

    # -- Provisioner protocol ---------------------------------------------

    def provision(self, job: Job, capacity: av.Capacity, plan: VolumePlan,
                  snapshot_interval_s: float) -> NodeHandle | None:
        from .providers_nebius import DEFAULT_IMAGE_FAMILY
        platform, _, preset = (capacity.sku or "").partition("/")
        if not platform or not preset:
            log.warning("[nebius-prov] sku %r is not platform/preset",
                        capacity.sku)
            return None
        try:
            volume = self._ensure_disk(job, plan)
            spec: dict[str, Any] = {
                "resources": {"platform": platform, "preset": preset},
                "networkInterfaces": [{
                    "name": "eth0", "subnetId": self._subnet(),
                    "ipAddress": {}, "publicIpAddress": {}}],
                "bootDisk": {
                    "attachMode": "READ_WRITE",
                    "managedDisk": {
                        "name": f"evsys-os-{job.id}-{int(time.time())}",
                        "spec": {"sizeGibibytes": self.boot_disk_gb,
                                 "type": "NETWORK_SSD",
                                 "sourceImageFamily": {
                                     "imageFamily": self.image_family
                                     or DEFAULT_IMAGE_FAMILY}}}},
                "secondaryDisks": [{
                    "attachMode": "READ_WRITE",
                    "existingDisk": {"id": volume},
                    "deviceId": DATA_DEVICE_ID}],
                "cloudInitUserData": self._script(job, plan,
                                                  snapshot_interval_s, volume),
                "hostname": f"evsys-{job.id}"[:32].rstrip("-"),
            }
            if capacity.spot:
                # The only combination Nebius allows for preemptible VMs.
                spec["recoveryPolicy"] = "FAIL"
                spec["preemptible"] = {"onPreemption": "STOP"}
            op = self.call("compute/v1/instances", {
                "metadata": {"parentId": self.project_id,
                             "name": f"evsys-{job.id}"},
                "spec": spec}, "POST")
            iid = self._resource_id(op)
            if not iid:
                log.warning("[nebius-prov] create returned no id: %r", op)
                return None
            return NodeHandle(job_id=job.id, provider="nebius",
                              machine_id=iid, region=capacity.region or "",
                              gpu=capacity.gpu, volume=volume,
                              usd_hr=capacity.usd_hr or 0.0)
        except Exception as e:
            log.warning("[nebius-prov] provision failed for %s: %s", job.id, e)
            return None

    def alive(self, handle: NodeHandle) -> bool:
        """RUNNING/CREATING/STARTING are alive. STOPPED is how preemption
        manifests here (the VM is kept, its disks intact) — dead to the
        router, which requeues and reuses the surviving volume."""
        try:
            inst = self.call(f"compute/v1/instances/{handle.machine_id}") or {}
        except Exception as e:
            msg = str(e)
            if "404" in msg or "not_found" in msg.lower():
                return False
            log.warning("[nebius-prov] poll failed: %s — treating as alive "
                        "to avoid a spurious requeue", e)
            return True
        state = ((inst.get("status") or {}).get("state") or "").upper()
        return state in ("CREATING", "STARTING", "RUNNING", "UPDATING", "")

    def terminate(self, handle: NodeHandle, *, keep_volume: bool = True) -> None:
        """Delete the instance. The data disk is an ExistingDisk attachment,
        so deletion detaches and preserves it; the managed boot disk dies
        with the instance (no orphan sweep needed — unlike Verda)."""
        try:
            self.call(f"compute/v1/instances/{handle.machine_id}",
                      method="DELETE")
        except Exception as e:
            log.warning("[nebius-prov] terminate failed: %s", e)
        if not keep_volume and handle.volume:
            try:
                self.call(f"compute/v1/disks/{handle.volume}", method="DELETE")
            except Exception as e:
                log.warning("[nebius-prov] disk delete failed: %s", e)


__all__ = ["AGENT_TEMPLATE", "DATA_DEVICE_ID", "NebiusProvisioner",
           "NebiusProvisionerConfig"]
