# Fork SkyPilot and put our spot abstractions inside it?

Assessed 2026-08-08 against skypilot @ 85fdf6b (2026-08-07), 239k lines of
Python in `sky/`. Verdict up front: **do not fork — bridge.** SkyPilot's
launcher breadth is real, but the two things we would fork it *for* are the
two places it is weakest, and both of our killer features live in our SDK
layer anyway, which composes with SkyPilot unmodified.

## What SkyPilot already has (the "router stuff")

* **~25 in-tree clouds** — including `verda.py` (383 LOC, spot supported),
  `nebius.py` (680 LOC), `vast.py`, `primeintellect.py`. Our two providers
  already exist upstream; forking to "add providers" solves a solved problem.
* **Managed jobs** (`sky/jobs/`): a 3.5k-line controller plus recovery
  strategies (`FAILOVER`, `EAGER_FAILOVER`, 1.6k LOC). Preemption detection
  by cluster-status polling; recovery relaunches — same region first, then
  failover across regions/clouds. This overlaps our router's
  reconcile-and-replace loop and is more battle-tested at breadth.
* Provisioning/ssh/storage plumbing, autostop, a service catalog with price
  data, an optimizer that places by price across clouds.

## What it does NOT have (verified in source, not guessed)

* **No checkpoint concept in the jobs layer.** `grep checkpoint sky/jobs/`
  returns nothing. Their contract: your job script checkpoints itself to a
  mounted bucket; recovery re-runs your command from scratch. No checkpoint
  map, no resume step, no delta encoding, no data-cursor fast-forward.
* **No live availability.** The catalog is fetched price CSVs; the optimizer
  ranks by price and discovers capacity by *failing over at launch time*.
  Nothing like Verda's per-SKU endpoint or Nebius's capacity advisor feeds
  placement.
* **No cross-relaunch data-volume reuse.** `sky volumes` exists but is
  k8s-PVC-centric; the Verda provision adapter creates only the OS volume.
  Zero-copy resume (reattach the surviving checkpoint volume to the
  replacement node) has no home in their model.

## The work, if we forked anyway

| Piece | Where it lands | Estimate |
|---|---|---|
| Live availability feeding placement | `sky/optimizer.py` + per-cloud probes | 1–2 weeks, invasive in their hottest file |
| Volume-reuse recovery | jobs controller + `sky/volumes` + per-cloud volume APIs (verda/nebius adapters lack data-volume attach) | 2–4 weeks |
| Delta snapshots, checkpoint map, ambient callback, resume | **nothing — these are SDK-side already** | 0 |
| Young/Daly cadence | env var into the job | hours |
| Glue + tests | — | 1–2 weeks |

Initial integration ≈ **4–8 weeks**, and then the real cost: a fork of a
repo that commits daily, diverging exactly in its two most-churned files
(optimizer, jobs controller) — a permanent rebase tax of days per month,
growing with divergence.

## The bridge instead (recommended)

Our architecture already split the right way: snapshotting, the checkpoint
map, and true resume are **SDK-side and launcher-agnostic**. Only
*placement* and *node lifecycle* are router-side, and provisioners are a
registry extension kind. So:

1. **`SkyPilotProvisioner`** (~300–500 LOC + tests, days not weeks):
   `provision()` = `sky launch` with our agent as the run command and a
   bucket/volume mount for `EVSYS_STORE_DIR`; `alive()` = `sky status`;
   `terminate()` = `sky down`. One provisioner class buys every SkyPilot
   cloud for fresh placements. Our router keeps live availability, spot
   defaults, the map, and Young/Daly.
2. **Keep native Verda/Nebius provisioners** for the clouds where we control
   volumes — that is where zero-copy resume lives, and SkyPilot cannot offer
   it anyway.
3. **Upstream small PRs** where SkyPilot is deficient and it helps us
   (e.g. data-volume attach in their verda adapter) rather than forking.

Revisit forking only if (a) we need SkyPilot's managed-jobs controller as
the *primary* orchestrator across many clouds at once, AND (b) upstream
rejects availability/volume hooks, AND (c) someone owns the rebase tax as a
standing job.
