"""Launch one autonomous benchmark box for a given card. No SSH: the boot script
installs deps, runs the 8B sweep-to-OOM + 9B probe, streams to ntfy, self-halts.
Usage: launch_card.py <sku> <region> <card_label> <price_hr> <8b_rungs> <9b_rungs>
"""
import base64, json, os, sys, time, urllib.request, urllib.error
SD = os.path.dirname(os.path.abspath(__file__))
API = "https://api.verda.com/v1"
sku, region, card, price, r8, r9 = sys.argv[1:7]
TOPIC = open(f"{SD}/ntfy_topic.txt").read().strip()
KID = open(f"{SD}/key_id.txt").read().strip()

def tok():
    b = json.dumps({"grant_type": "client_credentials", "client_id": os.environ["VERDA_CLIENT_ID"],
                    "client_secret": os.environ["VERDA_CLIENT_SECRET"]}).encode()
    r = urllib.request.Request(f"{API}/oauth2/token", data=b, headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=30))["access_token"]

def call(path, body=None, method=None):
    d = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(f"{API}/{path}", data=d, method=method,
        headers={"Authorization": f"Bearer {tok()}", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=40) as r: return r.read().decode()
    except urllib.error.HTTPError as e: return f"ERR {e.code} {e.read().decode()[:200]}"

b64 = base64.b64encode(open(f"{SD}/bench.py", "rb").read()).decode()
startup = r'''#!/bin/bash
exec > /var/log/evsys_bench.log 2>&1
set -x
export DEBIAN_FRONTEND=noninteractive HOME=/root
TOPIC="__TOPIC__"; NT="https://ntfy.sh/$TOPIC"; L=/var/log/evsys_bench.log
say(){ curl -sS -H "Title: $1" -d "$2" "$NT" >/dev/null 2>&1 || true; }
step(){ local t="$1"; shift; local s=$(date +%s); say "$t" start; "$@"; local rc=$?
  if [ $rc -ne 0 ]; then say "FAIL:$t" "rc=$rc $(( $(date +%s)-s ))s | $(tail -c 1000 $L | tr '\n' ' ')"; return 1; fi
  say "$t" "ok $(( $(date +%s)-s ))s"; }
say boot "__CARD__ up $(hostname); $(nvidia-smi -L 2>&1 | tr '\n' '|' | head -c 250)"
( sleep 18000; say deadman "5h cap"; /sbin/shutdown -h now ) &
echo __B64__ | base64 -d > /root/bench.py
(
  export PATH="/root/.local/bin:$PATH" HF_HUB_ENABLE_HF_TRANSFER=1 BENCH_TOPIC="$TOPIC"
  step uv bash -c 'curl -LsSf https://astral.sh/uv/install.sh | sh' || { /sbin/shutdown -h now; exit; }
  export PATH="/root/.local/bin:$PATH"; PY=/root/venv/bin/python
  step venv uv venv /root/venv --python 3.11 || { /sbin/shutdown -h now; exit; }
  step torch uv pip install --python $PY torch --index-url https://download.pytorch.org/whl/cu124 || { /sbin/shutdown -h now; exit; }
  step libs uv pip install --python $PY "transformers>=4.51" peft accelerate hf_transfer safetensors || { /sbin/shutdown -h now; exit; }
  say sweep8b "8B sweep on __CARD__ (to OOM)"
  CARD="__CARD__" PRICE_HR=__PRICE__ BENCH_MODEL="Qwen/Qwen3-8B" RUNGS="__R8__" $PY /root/bench.py
  say sweep9b "9B (Qwen3.5) probe on __CARD__"
  CARD="__CARD__" PRICE_HR=__PRICE__ BENCH_MODEL="Qwen/Qwen3.5-9B" RUNGS="__R9__" $PY /root/bench.py
  say ALL_DONE "__CARD__ complete — halting"
  /sbin/shutdown -h now
) &
echo STARTUP_RETURNED
'''
for k, v in {"__TOPIC__": TOPIC, "__CARD__": card, "__PRICE__": price, "__R8__": r8, "__R9__": r9, "__B64__": b64}.items():
    startup = startup.replace(k, v)
sid = call("scripts", {"name": f"evsys-{card}", "script": startup}, "POST").strip().strip('"')
av = call(f"instance-availability/{sku}?is_spot=true&location_code={region}").strip()
print(f"{card} {sku}/{region} spot avail:", av)
if av != "true":
    print("NOT AVAILABLE"); sys.exit(2)
vol = f"evsys-{card}-{int(time.time())}"
body = {"instance_type": sku, "image": "aaaaaaaa-3dd9-4d09-9512-52d8032fff6e",
        "ssh_key_ids": [KID], "startup_script_id": sid, "hostname": f"evsys-{card}".lower(),
        "description": f"evsys {card} autorun", "location_code": region, "is_spot": True,
        "contract": "SPOT", "os_volume": {"name": vol, "size": 300}}
iid = call("instances", body).strip().strip('"')
print("LAUNCHED:", iid, vol)
existing = json.load(open(f"{SD}/live_instances.json")) if os.path.exists(f"{SD}/live_instances.json") else []
existing.append({"id": iid, "volume": vol, "card": card, "script": sid})
json.dump(existing, open(f"{SD}/live_instances.json", "w"))
