"""Tests for VerdaProvisioner — the proven shutdown->attach->boot semantics,
against a faked transport (same approach as test_providers_verda)."""
import pytest

from evsys_sdk.compute import availability as av
from evsys_sdk.compute.checkpoint_map import Checkpoint, StoreRef
from evsys_sdk.compute.job_router import NodeHandle, VolumePlan
from evsys_sdk.compute.provisioner_verda import (AGENT_TEMPLATE,
                                                 VerdaProvisioner)
from evsys_sdk.compute.queue import Job


def _cap(sku="1H100.80S.32V", region="FIN-02"):
    return av.Capacity(provider="verda", gpu="H100", count=1,
                       state="available", region=region, sku=sku, spot=True,
                       usd_hr=1.625)


class FakeApi:
    def __init__(self):
        self.log = []           # (path, body, method)
        self.instances = []

    def __call__(self, path, body=None, method=None):
        self.log.append((path, body, method))
        if path == "scripts":
            return "script-1"
        if path == "volumes" and method == "POST" or (path == "volumes" and body and "location_code" in (body or {})):
            return "vol-new"
        if path == "instances" and body and "instance_type" in body:
            return "inst-1"
        if path == "instances" and body is None:
            return self.instances
        return ""


@pytest.fixture
def prov():
    api = FakeApi()
    p = VerdaProvisioner(api, ssh_key_id="key-1",
                         payload="python3 /root/train.py", boot_wait_s=0)
    return p, api


def test_fresh_provision_creates_volume_and_does_attach_dance(prov):
    p, api = prov
    job = Job(config="c.yaml", model="m")
    h = p.provision(job, _cap(), VolumePlan("fresh"), 569.0)
    assert isinstance(h, NodeHandle)
    assert h.machine_id == "inst-1" and h.volume == "vol-new"
    ops = [(path, (body or {}).get("action"), m) for path, body, m in api.log]
    # order: volume create (id feeds the script env), script, instance,
    # then the dance: shutdown, attach, boot
    assert ops[0][0] == "volumes"
    assert ("scripts", None, "POST") == ops[1]
    assert ops[2][0] == "instances"
    assert ("instances", "shutdown", "PUT") in ops
    assert ("volumes", "attach", "PUT") in ops
    assert ("instances", "boot", "PUT") in ops
    # the dance is ordered: shutdown before attach before boot
    i_shut = ops.index(("instances", "shutdown", "PUT"))
    i_att = ops.index(("volumes", "attach", "PUT"))
    i_boot = ops.index(("instances", "boot", "PUT"))
    assert i_shut < i_att < i_boot


def test_reuse_plan_attaches_surviving_volume_without_creating(prov):
    p, api = prov
    job = Job(config="c.yaml", model="m")
    ck = Checkpoint(job_id=job.id, step=200,
                    store=StoreRef(kind="local_dir", provider="verda",
                                   volume="vol-old"),
                    base_key="base.evd", delta_key="step-200.evd")
    h = p.provision(job, _cap(), VolumePlan("reuse", volume="vol-old",
                                            checkpoint=ck), 569.0)
    assert h.volume == "vol-old"
    creates = [b for path, b, m in api.log
               if path == "volumes" and m == "POST"]
    assert creates == []                       # no new volume
    attach = [b for path, b, m in api.log
              if path == "volumes" and (b or {}).get("action") == "attach"]
    assert attach[0]["id"] == "vol-old"


def test_agent_script_is_reuse_safe_and_reboot_persistent(prov):
    p, api = prov
    job = Job(config="c.yaml", model="m")
    ck = Checkpoint(job_id=job.id, step=200,
                    store=StoreRef(kind="local_dir", provider="verda",
                                   volume="v"),
                    base_key="b", delta_key="d")
    p.provision(job, _cap(), VolumePlan("reuse", volume="v", checkpoint=ck),
                432.0)
    script = api.log[0][1]["script"]
    assert "@reboot" in script                     # survives the attach reboot
    assert "mkfs.ext4" in script and "! mount" not in script.replace(
        '&& ! mount', 'MARK')                      # sanity: template intact
    # mount is attempted BEFORE mkfs (reuse-safe)
    assert script.index("mount \"$DEV\" /data") < script.index("mkfs.ext4")
    assert "EVSYS_RESUME_STEP=200" in script
    assert "EVSYS_SNAPSHOT_INTERVAL_S=432" in script


def test_alive_and_terminate(prov):
    p, api = prov
    api.instances = [{"id": "inst-1", "status": "running"}]
    h = NodeHandle(job_id="j", provider="verda", machine_id="inst-1",
                   volume="vol-x")
    assert p.alive(h) is True
    api.instances = [{"id": "inst-1", "status": "discontinued"}]
    assert p.alive(h) is False
    api.instances = []
    assert p.alive(h) is False
    p.terminate(h, keep_volume=True)
    vol_deletes = [b for path, b, m in api.log
                   if path == "volumes" and (b or {}).get("action") == "delete"]
    assert vol_deletes == []                   # the checkpoints survive
    p.terminate(h, keep_volume=False)
    vol_deletes = [b for path, b, m in api.log
                   if path == "volumes" and (b or {}).get("action") == "delete"]
    assert vol_deletes and vol_deletes[0]["id"] == "vol-x"


def test_failed_launch_returns_none(prov):
    p, api = prov
    def boom(path, body=None, method=None):
        raise IOError("api down")
    p.call = boom
    job = Job(config="c.yaml", model="m")
    assert p.provision(job, _cap(), VolumePlan("fresh"), 500.0) is None
