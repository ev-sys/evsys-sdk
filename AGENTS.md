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
