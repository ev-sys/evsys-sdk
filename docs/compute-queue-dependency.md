# Compute queue dependency (MVP adapter)

The public SDK ships a **thin queue adapter** at
``evsys_sdk.compute.platform_queue``. It enqueues self-hosted SkyRL configs into
the local JSONL queue (``~/.evsys/queue.jsonl``) when:

```bash
export EVSYS_COMPUTE_QUEUE=1
# optional: block until placed
export EVSYS_COMPUTE_QUEUE_AUTO=1
```

## What works today

- ``submit_training_config()`` appends a job and optionally runs
  ``Scheduler(q).run()`` using the **built-in** availability + pricing modules.
- ``maybe_queue_experiment_backend()`` is the experiment hook for
  ``compute: {kind: skypilot}`` configs.

## What requires enterprise compute PRs (#22–#33)

These features live on **unmerged** branches and were intentionally **not**
squashed into this MVP:

- Multi-LoRA packing + live merge (#23)
- Zero-config durable checkpoints + resume (#22, #30)
- Util-aware pool admission (#31)
- Modal cloud backend (#32–#33)
- Vanish relaunch watchdog (#29)

Until those merge, spot placement uses the OSS ``evsys_sdk.compute.queue`` +
``skypilot`` provider only. Set ``EVSYS_COMPUTE_QUEUE=1`` to queue instead of
failing immediately when no GPU is free.

## Live smoke

```bash
EVSYS_COMPUTE_QUEUE=1 uv run pytest tests/test_platform_hosted.py::TestPlatformQueue -v
```

For a real spot launch you also need cloud credentials (``sky check``) and an
unmerged compute tip — do not expect full packing on public ``dev`` alone.
