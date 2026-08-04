# Running this repo in a Claude Code cloud session

Most of the setup lives in the repo and needs nothing from you. One part
cannot: Claude Code keeps a cloud environment's **network policy, environment
variables and setup script in the environment config at claude.ai**, not in
git. So the repo can make a session build correctly, but only you can make it
reach a GPU vendor.

## What already works, with zero configuration

The full test suite. It is hermetic — every provider test stubs the HTTP layer
and keeps the parsing real — so it passes with no credentials and no vendor
network. Verified by running the compute tests with `HOME` pointed at a
directory that does not exist: 170 passed.

`.claude/settings.json` runs `scripts/claude-session-start.sh` at SessionStart.
It no-ops locally (`CLAUDE_CODE_REMOTE` is only `true` in the cloud) and in a
cloud session installs the `dev` and `skypilot` extras, then prints whether each
GPU vendor is reachable.

## What you have to configure, and why

### Network access must be Custom

The default **Trusted** level allows package registries, GitHub and the big
clouds. It does **not** include any of the GPU vendors this SDK talks to, so on
a default environment `availability.scan()` and every launch path fail with a
timeout that looks like the vendor being down.

Open the environment selector at claude.ai/code, set **Network access** to
**Custom**, tick *Also include default list of common package managers*, and add:

```
api.verda.com
console.vast.ai
api.primeintellect.ai
```

The session-start hook probes exactly these three and prints
`vendor UNREACHABLE` for any that are missing, so a misconfigured environment
announces itself in the first few lines rather than fifteen minutes later.

### Setup script (optional, but it makes sessions start faster)

Paste `scripts/cloud-setup.sh` into the environment's **Setup script** field.
It is kept in the repo for review, but Claude Code does not read it from there.

Worth doing because of how caching works: the setup script runs **once**, then
Anthropic snapshots the filesystem and later sessions restore it. The
SessionStart hook runs **every** session. Moving the dependency install into the
setup script means each session starts with `.venv` already on disk.

### Credentials

There is **no secrets store**. Environment variables in a cloud environment are
readable by anyone who can use that environment, and the dialog says so.

That is a real constraint here rather than a nuisance: this SDK provisions
machines that bill by the hour, and a leaked Verda key is someone else's GPU
fleet on your balance. So:

* **Prefer not to.** Almost every task on this repo — writing providers, fixing
  the queue, extending the availability layer — needs no credentials at all.
* **Vast.ai search needs none.** Pricing and availability work unauthenticated;
  only renting needs a key. Use Vast for live capacity questions in the cloud.
* If a session genuinely must rent, put `VAST_API_KEY` in the environment
  variables, scope it as narrowly as the vendor allows, and rotate it after.
  Verda uses OAuth client credentials from `~/.verda/config.json`, which a
  setup script would have to write from an environment variable — same
  visibility caveat, so do it only when the task requires it.

## Limits that matter for this repo

Cloud VMs get roughly **4 vCPU, 16 GB RAM, 30 GB disk**. That is fine for the
SDK and its tests. It is nowhere near enough to run SkyRL itself — the training
stack alone installs ~27 GB and wants an 80 GB GPU. Cloud sessions are for
working on the *control plane*; the GPUs stay rented from a vendor.
