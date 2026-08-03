# Working notes for agents

See `CLAUDE.md` for repo conventions (registry pattern, dev workflow, extension
points). This file covers how to *operate* — the things that waste an hour when
you get them wrong.

## Always run long commands in the background

Anything that provisions, trains, installs, or polls goes in the background
(`run_in_background: true`), never as a blocking foreground call. These
routinely outlast a foreground timeout, and a timeout kills the call while the
*remote* work carries on — so you lose the handle to something that is still
running and still billing.

Applies to:

  * **GPU provisioning and capacity retries** — spot capacity appears and
    disappears within seconds. Retry loops belong in the background.
  * **Remote installs** (`uv sync` with the megatron extra pulls torch,
    transformer-engine, vLLM, megatron-core).
  * **Model downloads and server warmup** — the first `create_model` builds the
    policy and pulls the base model.
  * **Benchmarks and training runs** of any length.
  * **`sky launch` / `sky down`**, and any poll-until-ready loop.

Start it detached on the remote host too (`setsid nohup … < /dev/null &`), so
the work survives the SSH connection dropping. Then poll the log.

Two traps worth knowing:

  * `pkill -f <pattern>` **matches its own command line**, including the rest of
    a compound SSH command. `pkill -f run_matrix` inside a command that later
    mentions `run_matrix.sh` kills its own shell before reaching the next
    statement. Use a bracket-glob (`pkill -f "run_[m]atrix"`) or check whether
    anything is running first.
  * Never leave a helper script in the directory a benchmark runs from —
    Python puts the script's own directory at `sys.path[0]`, so an
    `inspect.py` next to the runner shadows the stdlib and every import of a
    third-party package dies with something unrelated-looking.

## Renting GPUs

Credentials for every provider go through `evsys_sdk.compute.credentials`,
which writes each provider's native format at mode 0600. `credentials.report()`
shows who is authenticated and what they support.

Price and stock come from `evsys_sdk.compute.pricing` — the **provider's live
API**, not SkyPilot's catalog. The catalog is a pre-generated CSV and has been
wrong by 65% on price and wrong about stock existing at all.

**A GPU left running bills until something terminates it.** Some providers
(PrimeIntellect, Verda) implement no autostop at all, so `max_lifetime_s` on
the SkyPilot target is the only backstop. Tear down explicitly when finished
and verify with a follow-up API call that the instance list is actually empty.

## Track reliability, per provider AND per GPU

Where to rent turns on two facts no provider publishes: whether a launch
succeeds, and how long the machine survives. Both vary by provider, by GPU
type, and by region, and both are knowable only by keeping score. Record every
outcome — `evsys_sdk.compute.reliability`:

```python
from evsys_sdk.compute import reliability as rel

rel.record(rel.LAUNCH_OK,    provider="verda", gpu="H200", count=1,
           region="FIN-03", usd_hr=1.40, wait_s=110)
rel.record(rel.LAUNCH_FAIL,  provider="verda", gpu="H100", reason="no_capacity")
rel.record(rel.PREEMPTED,    provider="verda", gpu="A100", uptime_s=2400)
rel.record(rel.TORN_DOWN,    provider="verda", gpu="A100", uptime_s=9000)

print(rel.report())                                  # the table
rel.suggested_snapshot_interval_s("verda", "H200")   # cadence from real MTBF
```

Append-only JSONL at `~/.evsys/reliability.jsonl`. Never rewritten, so
concurrent writers are safe and a half-written line costs one observation
rather than the history.

**Why this is not busywork.** The snapshot cadence is `T* = sqrt(2*C*MTBF)`,
which takes mean-time-between-preemptions as a direct input. Without measured
MTBF that formula is a guess wearing a formula's clothes. Every preemption you
record makes the next run's cadence better.

Three distinctions the ledger keeps and a naive counter would lose:

  * **`no_capacity` vs `no_funds`.** SkyPilot and most providers report both as
    "resources unavailable". They are opposite signals: one says something
    about the provider, the other says nothing at all. Conflating them cost
    hours before this was written down.
  * **`preempted` vs `torn_down`.** Only preemptions inform MTBF. Counting a
    deliberate teardown as uptime inflates the estimate and slackens the
    cadence; ignoring it hides how censored the evidence is. Both are stored.
  * **GPU count.** `1xA100` and `2xA100` are different products at different
    prices with different availability. They get separate rows.

Absent data reads as `-`, never as zero — a zero MTBF would mean "preempted
instantly" rather than "never observed".

**Observed so far** (2026-07-31 to 2026-08-03, small samples — treat as
directional):

| provider | gpu | launches | MTBF | notes |
|---|---|---|---|---|
| primeintellect | A100-80GB x1 | 1/4 | 0.7h (n=1) | 3 refusals were an empty wallet, not capacity |
| primeintellect | H100 x1 | 0/1 | - | catalog routed to a region with no stock |
| verda | A100 x2 | 1/1 | censored | ran 2.5h, torn down deliberately |
| verda | H100 x1 | 0/1 | - | 503 ~30s after availability listed it |
| verda | H200 x1 | 1/1 | - | no preemption observed yet |

The single most useful number here is that one observed spot preemption at
**~40 minutes**. If that holds, snapshot roughly every 3-5 minutes; one sample
is far too few to trust, which is exactly why the ledger exists.
