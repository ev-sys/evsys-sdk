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
cloud session installs dependencies, then prints whether each GPU vendor is
reachable.

`skypilot` is installed on its own rather than as a sync extra, because
`uv sync --extra skypilot` is **unresolvable**: skypilot 0.13 pins
`uvicorn<0.36` while harbor requires `>=0.38`. Installing it after the sync
downgrades uvicorn and works, which is how a working dev venv is actually
built. Without it the provisioning code does not import at all.

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

The SDK reads vendor credentials from **files**, but a cloud environment only
offers **variables**. `scripts/cloud-setup.sh` bridges the two: it writes
`~/.verda/config.json`, `~/.vast/config.json` and `~/.prime/config.json` at
0600 from the variables below, so the SDK finds them where it expects.

Add these to the environment's **Environment variables** field:

```
VERDA_CLIENT_ID=...
VERDA_CLIENT_SECRET=...
VAST_API_KEY=...
PRIME_API_KEY=...
```

To generate that block from the credentials already on your machine, without
reading them out anywhere they could be captured:

```bash
./scripts/print-cloud-env.sh
```

Copy its output straight into the browser field.

Know what you are accepting. A cloud environment has **no secrets store** — the
dialog says as much — and its variables are readable by anyone who can use the
environment. These particular keys rent GPUs that bill by the hour, so a leak is
someone else's fleet on your balance. Scope each key as narrowly as the vendor
allows and rotate them when the work is done.

Two ways to need fewer of them:

* Most work on this repo — providers, the queue, the availability layer — needs
  no credentials at all, because the tests are hermetic.
* **Vast.ai search needs no key.** Pricing and availability work
  unauthenticated; only renting requires one. For live capacity questions from
  a cloud session, Vast answers without a credential in the environment.

## Limits that matter for this repo

Cloud VMs get roughly **4 vCPU, 16 GB RAM, 30 GB disk**. That is fine for the
SDK and its tests. It is nowhere near enough to run SkyRL itself — the training
stack alone installs ~27 GB and wants an 80 GB GPU. Cloud sessions are for
working on the *control plane*; the GPUs stay rented from a vendor.
