"""The live e2e driver: the REAL JobRouter + VerdaProvisioner + TopicEvents,
running the SDK experiment yaml on a router-provisioned SkyRL node."""
import json
import os
import pathlib
import sys

sys.path.insert(0, "/home/user/evsys-sdk/src")

from evsys_sdk.compute.checkpoint_map import CheckpointMap
from evsys_sdk.compute.events_topic import TopicEvents
from evsys_sdk.compute.job_router import JobRouter
from evsys_sdk.compute.provisioner_verda import VerdaProvisioner
from evsys_sdk.compute.providers_verda import VerdaProvider
from evsys_sdk.compute.queue import Queue

SD = os.path.dirname(os.path.abspath(__file__))
TOPIC = open(f"{SD}/ntfy_topic.txt").read().strip()
SDK_URL = open(f"{SD}/sdk_url.txt").read().strip()

payload = open(f"{SD}/e2e_payload.sh").read().replace("__SDK_URL__", SDK_URL)

provider = VerdaProvider()
keys = provider._call("sshkeys")
assert keys, "no ssh key registered"

prov = VerdaProvisioner(
    provider._call, ssh_key_id=keys[0]["id"], payload=payload,
    events_url=f"https://ntfy.sh/{TOPIC}", volume_gb=100,
    boot_wait_s=90.0,   # give first boot time to finish cloud-init before the dance
)
router = JobRouter(
    Queue(path=f"{SD}/e2e_q.jsonl"),
    CheckpointMap(path=f"{SD}/e2e_map.jsonl"),
    prov,
    events=TopicEvents(f"https://ntfy.sh/{TOPIC}"),
    vendors=["verda"],
    handles_path=f"{SD}/e2e_handles.json",
    snapshot_cost_s=30.0, mtbf_s=3600.0,
    poll_s=45.0,
)

cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
if cmd == "submit":
    job = router.submit("/root/config.yaml", model="Qwen/Qwen3-4B-Instruct-2507",
                        gpus=["H100"], spot=None)
    print("submitted", job.id)
elif cmd == "tick":
    print(json.dumps(router.tick(), default=str))
elif cmd == "status":
    for j in router.queue.jobs():
        ck = router.cmap.latest(j.id)
        print(j.describe(), f"| ckpt step {ck.step}" if ck else "| no ckpt",
              f"| meta {ck.meta.get('state_path','')}" if ck else "")
    print("handles:", {k: v.machine_id for k, v in router.handles.items()})
elif cmd == "run":
    router.run(timeout_s=float(sys.argv[2]) if len(sys.argv) > 2 else 5400)
