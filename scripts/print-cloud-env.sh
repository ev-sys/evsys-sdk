#!/bin/bash
# Print the .env block to paste into a Claude Code cloud environment's
# "Environment variables" field, read from the credential files on THIS machine.
#
# Run it yourself and copy the output straight into the browser. It is a
# separate script precisely so the values are never echoed into an agent
# transcript, which is where credentials leak from in practice.
set -uo pipefail
python3 - <<'PY'
import json, pathlib
def read(p, *keys):
    f = pathlib.Path(p).expanduser()
    if not f.exists():
        return None
    try:
        d = json.loads(f.read_text())
    except Exception:
        return None
    return [d.get(k) for k in keys]

v = read("~/.verda/config.json", "client_id", "client_secret")
if v and all(v):
    print(f"VERDA_CLIENT_ID={v[0]}")
    print(f"VERDA_CLIENT_SECRET={v[1]}")
else:
    print("# ~/.verda/config.json not found or incomplete")

k = read("~/.vast/config.json", "api_key")
print(f"VAST_API_KEY={k[0]}" if k and k[0] else "# no ~/.vast/config.json (search works without a key)")

p = read("~/.prime/config.json", "api_key")
if p and p[0]:
    print(f"PRIME_API_KEY={p[0]}")
PY
