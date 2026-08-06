"""Verda's side of the router's Provisioner contract — the proven dance.

Every semantic here was learned live (experiments/spot_limits/
results_e2e_checkpoint_restore.md), not read from docs:

  * ``PUT /volumes {action: attach}`` **requires the instance shut down** —
    the API answers "Instance should be shutdown". So attaching a surviving
    volume is create -> shutdown -> attach -> boot.
  * **Startup scripts run on first boot only.** The reboot after attach does
    not re-run them, so the agent template installs an ``@reboot`` cron on
    first boot; the post-attach boot enters through that.
  * The agent must **mount before mkfs** — formatting is only for a mount
    failure on a fresh volume. An unconditional mkfs destroys the very
    checkpoints a ``reuse`` plan came to recover.
  * Deleting an instance auto-detaches extra volumes; they survive and keep
    billing, which is why ``terminate(keep_volume=True)`` leaves the data
    volume alone and only the router's cleanup (job DONE + map forgotten)
    ever deletes one.

Transport is injected (``call``) exactly like the VerdaProvider tests mock
``_call``, so this file is testable without an account.
"""

from __future__ import annotations

import time
from typing import Any, Callable

from ..logger import get_logger
from . import availability as av
from .job_router import NodeHandle, VolumePlan
from .queue import Job

log = get_logger(__name__)

#: Ubuntu 22.04 + CUDA 12.8 — the image every campaign used. The full CUDA
#: toolkit is on it at /usr/local/cuda (off PATH — agents must export it).
DEFAULT_IMAGE = "aaaaaaaa-3dd9-4d09-9512-52d8032fff6e"

#: The reboot-persistent agent skeleton. The job payload (``{payload}``) is
#: whatever actually trains; this wrapper owns the volume, the cadence and
#: reboot survival. ``{interval}`` is the Young/Daly snapshot interval the
#: router computed.
AGENT_TEMPLATE = """#!/bin/bash
exec > /var/log/evsys_agent.log 2>&1
set -x
cat > /root/agent_run.sh <<'RUNEOF'
#!/bin/bash
exec >> /var/log/evsys_agent.log 2>&1
# mount-first: mkfs only if the volume has no filesystem (reuse-safe)
DEV=""
for i in $(seq 1 90); do
  DEV=$(lsblk -ndo NAME,TYPE,MOUNTPOINT | awk '$2=="disk" && NF==2 && $1!~"^(vda|sda|nvme0n1)$" {{print "/dev/"$1; exit}}')
  [ -n "$DEV" ] && break; sleep 5
done
mkdir -p /data
if [ -n "$DEV" ] && ! mount "$DEV" /data 2>/dev/null; then
  mkfs.ext4 -F "$DEV" && mount "$DEV" /data
fi
export EVSYS_SNAPSHOT_INTERVAL_S={interval}
export EVSYS_JOB_ID={job_id}
export EVSYS_RESUME_STEP={resume_step}
{payload}
RUNEOF
chmod +x /root/agent_run.sh
echo "@reboot root /root/agent_run.sh" > /etc/cron.d/evsys-agent
/root/agent_run.sh
"""


class VerdaProvisioner:
    """Implements the router's Provisioner protocol against Verda's API."""

    def __init__(self, call: Callable[..., Any], ssh_key_id: str,
                 payload: str = "echo agent-payload-not-configured",
                 image: str = DEFAULT_IMAGE, volume_gb: int = 300,
                 boot_wait_s: float = 20.0):
        """``call(path, body=None, method=None)`` is the Verda transport —
        the same shape VerdaProvider uses, injected so tests can fake it."""
        self.call = call
        self.ssh_key_id = ssh_key_id
        self.payload = payload
        self.image = image
        self.volume_gb = volume_gb
        self.boot_wait_s = boot_wait_s

    # -- helpers -----------------------------------------------------------

    def _script(self, job: Job, plan: VolumePlan, interval_s: float) -> str:
        resume = plan.checkpoint.step if plan.checkpoint else 0
        script = AGENT_TEMPLATE.format(interval=int(interval_s),
                                       job_id=job.id, resume_step=resume,
                                       payload=self.payload)
        sid = self.call("scripts", {"name": f"evsys-agent-{job.id}",
                                    "script": script}, "POST")
        return str(sid).strip().strip('"')

    def _ensure_volume(self, job: Job, plan: VolumePlan,
                       region: str) -> str:
        if plan.mode == "reuse":
            return plan.volume
        vid = self.call("volumes", {"name": f"evsys-{job.id}",
                                    "size": self.volume_gb, "type": "NVMe",
                                    "location_code": region}, "POST")
        return str(vid).strip().strip('"')

    # -- Provisioner protocol ---------------------------------------------

    def provision(self, job: Job, capacity: av.Capacity, plan: VolumePlan,
                  snapshot_interval_s: float) -> NodeHandle | None:
        region = capacity.region or ""
        try:
            sid = self._script(job, plan, snapshot_interval_s)
            volume = self._ensure_volume(job, plan, region)
            iid = self.call("instances", {
                "instance_type": capacity.sku, "image": self.image,
                "ssh_key_ids": [self.ssh_key_id], "startup_script_id": sid,
                "hostname": f"evsys-{job.id}"[:32],
                "description": f"evsys job {job.id}",
                "location_code": region, "is_spot": capacity.spot,
                "contract": "SPOT" if capacity.spot else "PAY_AS_YOU_GO",
                "os_volume": {"name": f"evsys-os-{job.id}-{int(time.time())}",
                              "size": 50}})
            iid = str(iid).strip().strip('"')
            if not iid:
                return None
            # The proven attach dance: the data volume can only attach to a
            # SHUT DOWN instance, and the post-attach boot re-enters the agent
            # through the @reboot cron the first boot installed.
            time.sleep(self.boot_wait_s)
            self.call("instances", {"id": iid, "action": "shutdown"}, "PUT")
            time.sleep(self.boot_wait_s)
            self.call("volumes", {"action": "attach", "id": volume,
                                  "instance_id": iid}, "PUT")
            self.call("instances", {"id": iid, "action": "boot"}, "PUT")
            return NodeHandle(job_id=job.id, provider="verda",
                              machine_id=iid, region=region,
                              gpu=capacity.gpu, volume=volume,
                              usd_hr=capacity.usd_hr or 0.0)
        except Exception as e:                             # noqa: BLE001
            log.warning("[verda-prov] provision failed for %s: %s", job.id, e)
            return None

    def alive(self, handle: NodeHandle) -> bool:
        try:
            for inst in self.call("instances") or []:
                if inst.get("id") == handle.machine_id:
                    return inst.get("status") not in ("discontinued", "error")
        except Exception as e:                             # noqa: BLE001
            log.warning("[verda-prov] poll failed: %s — treating as alive "
                        "to avoid a spurious requeue", e)
            return True
        return False

    def terminate(self, handle: NodeHandle, *, keep_volume: bool = True) -> None:
        try:
            self.call("instances", {"id": handle.machine_id,
                                    "action": "delete"}, "PUT")
        except Exception as e:                             # noqa: BLE001
            log.warning("[verda-prov] terminate failed: %s", e)
        if not keep_volume and handle.volume:
            try:
                self.call("volumes", {"action": "delete", "id": handle.volume,
                                      "is_permanent": True}, "PUT")
            except Exception as e:                         # noqa: BLE001
                log.warning("[verda-prov] volume delete failed: %s", e)


__all__ = ["AGENT_TEMPLATE", "DEFAULT_IMAGE", "VerdaProvisioner"]
