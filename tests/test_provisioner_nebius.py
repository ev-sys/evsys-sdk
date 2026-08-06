"""Tests for NebiusProvisioner — create-time disk attach (no dance),
STOP-preemption liveness, disk-preserving terminate. Faked transport."""
import pytest

from evsys_sdk.compute import availability as av
from evsys_sdk.compute.checkpoint_map import Checkpoint, StoreRef
from evsys_sdk.compute.job_router import NodeHandle, VolumePlan, build_provisioner
from evsys_sdk.compute.provisioner_nebius import DATA_DEVICE_ID, NebiusProvisioner
from evsys_sdk.compute.queue import Job


def _cap(sku="gpu-h100-sxm/8gpu-128vcpu-1600gb", region="eu-north1",
         spot=True):
    return av.Capacity(provider="nebius", gpu="H100", count=8,
                       state="available", region=region, sku=sku, spot=spot,
                       usd_hr=17.2)


class FakeApi:
    def __init__(self):
        self.log = []                    # (path, body, method)
        self.instance = {"status": {"state": "RUNNING"}}
        self.fail_get = None

    def __call__(self, path, body=None, method=None):
        self.log.append((path, body, method))
        if path.startswith("vpc/v1/subnets"):
            return {"items": [{"metadata": {"id": "subnet-1"}}]}
        if path == "compute/v1/disks" and method == "POST":
            return {"resourceId": "disk-new"}
        if path == "compute/v1/instances" and method == "POST":
            return {"metadata": {"id": "op-1"}, "resourceId": "inst-1"}
        if path.startswith("compute/v1/instances/") and method is None:
            if self.fail_get:
                raise self.fail_get
            return self.instance
        return {}


@pytest.fixture
def prov():
    api = FakeApi()
    p = NebiusProvisioner(api, project_id="project-1",
                          payload="python3 /root/train.py")
    return p, api


def test_fresh_provision_creates_disk_and_attaches_at_create(prov):
    p, api = prov
    job = Job(config="c.yaml", model="m")
    h = p.provision(job, _cap(), VolumePlan("fresh"), 569.0)
    assert isinstance(h, NodeHandle)
    assert h.machine_id == "inst-1" and h.volume == "disk-new"
    assert h.provider == "nebius"
    # ONE instance call, disk already in its spec — no shutdown/attach dance
    inst = [b for path, b, m in api.log
            if path == "compute/v1/instances" and m == "POST"][0]
    sec = inst["spec"]["secondaryDisks"]
    assert sec[0]["existingDisk"]["id"] == "disk-new"
    assert sec[0]["deviceId"] == DATA_DEVICE_ID
    assert inst["metadata"]["parentId"] == "project-1"
    assert inst["spec"]["resources"] == {
        "platform": "gpu-h100-sxm", "preset": "8gpu-128vcpu-1600gb"}


def test_spot_gets_the_only_legal_preemptible_combo(prov):
    p, api = prov
    p.provision(Job(config="c.yaml", model="m"), _cap(spot=True),
                VolumePlan("fresh"), 500.0)
    spec = [b for path, b, m in api.log
            if path == "compute/v1/instances" and m == "POST"][0]["spec"]
    assert spec["recoveryPolicy"] == "FAIL"
    assert spec["preemptible"] == {"onPreemption": "STOP"}


def test_on_demand_omits_preemptible(prov):
    p, api = prov
    p.provision(Job(config="c.yaml", model="m"), _cap(spot=False),
                VolumePlan("fresh"), 500.0)
    spec = [b for path, b, m in api.log
            if path == "compute/v1/instances" and m == "POST"][0]["spec"]
    assert "preemptible" not in spec and "recoveryPolicy" not in spec


def test_reuse_plan_attaches_surviving_disk_without_creating(prov):
    p, api = prov
    job = Job(config="c.yaml", model="m")
    ck = Checkpoint(job_id=job.id, step=200,
                    store=StoreRef(kind="local_dir", provider="nebius",
                                   volume="disk-old"),
                    base_key="base.evd", delta_key="step-200.evd")
    h = p.provision(job, _cap(), VolumePlan("reuse", volume="disk-old",
                                            checkpoint=ck), 569.0)
    assert h.volume == "disk-old"
    assert not any(path == "compute/v1/disks" and m == "POST"
                   for path, b, m in api.log)
    spec = [b for path, b, m in api.log
            if path == "compute/v1/instances" and m == "POST"][0]["spec"]
    assert spec["secondaryDisks"][0]["existingDisk"]["id"] == "disk-old"
    # the agent resumes from the recorded step and mounts before mkfs
    script = spec["cloudInitUserData"]
    assert "EVSYS_RESUME_STEP=200" in script
    assert "EVSYS_PROVIDER=nebius" in script
    assert f"virtio-{DATA_DEVICE_ID}" in script
    assert script.index('mount "$DEV" /data') < script.index("mkfs.ext4")


def test_bad_sku_refuses(prov):
    p, api = prov
    assert p.provision(Job(config="c", model="m"), _cap(sku="justplatform"),
                       VolumePlan("fresh"), 100.0) is None


def test_alive_states(prov):
    p, api = prov
    h = NodeHandle(job_id="j", provider="nebius", machine_id="inst-1",
                   volume="disk-1")
    assert p.alive(h) is True
    # STOP is how preemption manifests on Nebius: dead to the router
    api.instance = {"status": {"state": "STOPPED"}}
    assert p.alive(h) is False
    api.fail_get = OSError("404 instance not found")
    assert p.alive(h) is False
    api.fail_get = OSError("timeout")           # transient -> assume alive
    assert p.alive(h) is True


def test_terminate_preserves_data_disk_by_default(prov):
    p, api = prov
    h = NodeHandle(job_id="j", provider="nebius", machine_id="inst-1",
                   volume="disk-1")
    p.terminate(h, keep_volume=True)
    deletes = [(path, m) for path, b, m in api.log if m == "DELETE"]
    assert deletes == [("compute/v1/instances/inst-1", "DELETE")]
    p.terminate(h, keep_volume=False)
    deletes = [(path, m) for path, b, m in api.log if m == "DELETE"]
    assert ("compute/v1/disks/disk-1", "DELETE") in deletes


def test_subnet_config_wins_over_lookup():
    api = FakeApi()
    p = NebiusProvisioner(api, project_id="project-1", subnet_id="subnet-x")
    p.provision(Job(config="c", model="m"), _cap(), VolumePlan("fresh"), 100.0)
    assert not any(path.startswith("vpc/") for path, b, m in api.log)
    spec = [b for path, b, m in api.log
            if path == "compute/v1/instances" and m == "POST"][0]["spec"]
    assert spec["networkInterfaces"][0]["subnetId"] == "subnet-x"


def test_registered_and_buildable_via_registry():
    import evsys_sdk.compute  # noqa: F401
    from evsys_sdk.registry import list_provisioners
    assert "nebius" in list_provisioners()
    api = FakeApi()
    p = build_provisioner("nebius", {"project_id": "project-9",
                                     "volume_gb": 150}, call=api)
    assert isinstance(p, NebiusProvisioner)
    assert p.project_id == "project-9" and p.volume_gb == 150
    with pytest.raises(Exception):
        build_provisioner("nebius", {"projject_id": "typo"}, call=api)
