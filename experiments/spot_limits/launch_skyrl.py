import base64, json, os, sys, time, urllib.request, urllib.error
SD=os.path.dirname(os.path.abspath(__file__)); API="https://api.verda.com/v1"
sku,region,card,price = sys.argv[1:5]   # price = PER-GPU $/hr (branch convention)
TOPIC=open(f"{SD}/ntfy_topic.txt").read().strip(); KID=open(f"{SD}/key_id.txt").read().strip()
def tok():
    b=json.dumps({"grant_type":"client_credentials","client_id":os.environ["VERDA_CLIENT_ID"],"client_secret":os.environ["VERDA_CLIENT_SECRET"]}).encode()
    r=urllib.request.Request(f"{API}/oauth2/token",data=b,headers={"Content-Type":"application/json"})
    return json.load(urllib.request.urlopen(r,timeout=30))["access_token"]
def call(p,body=None,m=None):
    d=json.dumps(body).encode() if body is not None else None
    req=urllib.request.Request(f"{API}/{p}",data=d,method=m,headers={"Authorization":f"Bearer {tok()}","Content-Type":"application/json"})
    try:
        with urllib.request.urlopen(req,timeout=40) as r:return r.read().decode()
    except urllib.error.HTTPError as e:return f"ERR {e.code} {e.read().decode()[:200]}"
def b64(p): return base64.b64encode(open(f"{SD}/skyrl/{p}","rb").read()).decode()
startup=r'''#!/bin/bash
exec > /var/log/evsys.log 2>&1
set -x
export DEBIAN_FRONTEND=noninteractive HOME=/root
TOPIC="__TOPIC__"; NT="https://ntfy.sh/$TOPIC"
say(){ curl -sS -H "Title: $1" -d "$2" "$NT" >/dev/null 2>&1 || true; }
sayfile(){ # gzip+base64 the file, stream in spaced chunks (ntfy rate-limit safe)
  local t="$1" f="$2"
  gzip -c "$f" | base64 -w0 > /tmp/enc
  local sz=$(stat -c%s /tmp/enc); local n=$(( (sz+2999)/3000 ))
  for i in $(seq 0 $((n-1))); do
    say "$t.$((i+1))of$n" "$(dd if=/tmp/enc bs=3000 skip=$i count=1 2>/dev/null)"
    sleep 2
  done
}
say boot "SKYRL __CARD__ up $(hostname); $(nvidia-smi -L | tr '\n' '|' | head -c 200)"
( sleep 12600; say deadman "3.5h cap"; /sbin/shutdown -h now ) &
echo __SETUP__  | base64 -d > /root/setup.sh;  chmod +x /root/setup.sh
echo __SERVER__ | base64 -d > /root/server_gapBs.sh; chmod +x /root/server_gapBs.sh
echo __LONGCTX__| base64 -d > /root/longctx.py
(
  export PATH="/root/.local/bin:$PATH" HF_HUB_ENABLE_HF_TRANSFER=1
  say setup "starting SkyRL clone + uv sync (the long pole)..."
  S=$(date +%s); bash /root/setup.sh > /root/setup.log 2>&1
  if ! grep -q SETUP_EXIT=0 /root/setup.log; then
    say "FAIL:setup" "$(( $(date +%s)-S ))s | $(tail -c 1500 /root/setup.log | tr '\n' ' ')"; /sbin/shutdown -h now; exit
  fi
  say setup "SkyRL synced OK in $(( $(date +%s)-S ))s"
  start_server(){ # $1=model $2=tag
    pkill -f skyrl.tinker.api; ray stop --force >/dev/null 2>&1; pkill -9 -f 'ray::'; pkill -9 -f raylet; sleep 10
    GAP_MODEL="$1" setsid nohup /root/server_gapBs.sh > /root/srv_$2.log 2>&1 < /dev/null &
    for i in $(seq 1 150); do
      curl -sf --max-time 3 http://127.0.0.1:8000/api/v1/healthz >/dev/null && return 0
      sleep 8
    done
    return 1
  }
  cd /root/skyrl 2>/dev/null || cd ~/skyrl
  UV="uv run --extra tinker --extra megatron python"
  # ---- 4B validation (known-good numbers on branch) ----
  say srv4b "starting tinker server for 4B..."
  if start_server Qwen/Qwen3-4B-Instruct-2507 s4b; then
    say srv4b "4B server healthy; running longctx cells"
    $UV /root/longctx.py --model Qwen/Qwen3-4B-Instruct-2507 --price-hr __PRICE__ \
        --rungs 512:128,2048:32,8192:8 --out /root/val4b.json
    say cells4b "rc=$?"; sayfile RES4B /root/val4b.json
  else
    say "FAIL:srv4b" "$(tail -c 1500 /root/srv_s4b.log | tr '\n' ' ')"
  fi
  # ---- 9B on the REAL stack: does GDN JIT on Hopper? ----
  say srv9b "starting tinker server for Qwen3.5-9B (GDN JIT test on Hopper)..."
  if start_server Qwen/Qwen3.5-9B s9b; then
    say srv9b "9B SERVER HEALTHY ON HOPPER; running cells"
    $UV /root/longctx.py --model Qwen/Qwen3.5-9B --price-hr __PRICE__ \
        --rungs 512:16,2048:8 --out /root/val9b.json
    say cells9b "rc=$?"; sayfile RES9B /root/val9b.json
  else
    say "FAIL:srv9b" "$(tail -c 2500 /root/srv_s9b.log | tr '\n' ' ')"
  fi
  say ALL_DONE "skyrl run complete"
  /sbin/shutdown -h now
) &
echo STARTUP_RETURNED
'''
for k,v in {"__TOPIC__":TOPIC,"__CARD__":card,"__PRICE__":price,
            "__SETUP__":b64("setup_remote.sh"),"__SERVER__":b64("server_gapBs.sh"),
            "__LONGCTX__":b64("longctx.py")}.items():
    startup=startup.replace(k,v)
sid=call("scripts",{"name":f"evskyrl-{card}","script":startup},"POST").strip().strip('"')
av=call(f"instance-availability/{sku}?is_spot=true&location_code={region}").strip()
print(f"{card} {sku}/{region} spot avail:",av)
if av!="true": print("NOT AVAILABLE"); sys.exit(2)
vol=f"evskyrl-{int(time.time())}"
iid=call("instances",{"instance_type":sku,"image":"aaaaaaaa-3dd9-4d09-9512-52d8032fff6e","ssh_key_ids":[KID],
    "startup_script_id":sid,"hostname":"evskyrl","description":"skyrl real-stack bench","location_code":region,
    "is_spot":True,"contract":"SPOT","os_volume":{"name":vol,"size":300}}).strip().strip('"')
print("LAUNCHED:",iid,vol)
json.dump([{"id":iid,"volume":vol,"card":card,"script":sid}],open(f"{SD}/live_instances.json","w"))
