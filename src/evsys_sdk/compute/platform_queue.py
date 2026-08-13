"""Thin adapter: submit GPU training jobs to the local compute queue.

Enterprise compute packing / spot launch lives on unmerged PRs (#22–#33). This
module wires the **hosted SDK path** only: when ``EVSYS_COMPUTE_QUEUE=1`` and a
training config needs self-hosted SkyRL, enqueue locally and optionally run the
scheduler in-process.

Full spot packing requires merging the compute stack — see ``docs/compute-queue-dependency.md``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from ..logger import get_logger
from .queue import DEFAULT_GPUS, Job, Queue, Scheduler

log = get_logger(__name__)

QUEUE_ENV = "EVSYS_COMPUTE_QUEUE"
AUTO_SCHEDULE_ENV = "EVSYS_COMPUTE_QUEUE_AUTO"


def queue_enabled() -> bool:
    return os.environ.get(QUEUE_ENV, "").lower() in ("1", "true", "yes")


def submit_training_config(
    config_path: str | Path,
    *,
    model: str,
    gpus: list[str] | None = None,
    spot: bool | None = True,
    queue: Queue | None = None,
    auto_schedule: bool | None = None,
) -> Job:
    """Enqueue a YAML training config for spot/on-demand placement.

    Returns the :class:`Job` record. When ``auto_schedule`` is true (default
    when ``EVSYS_COMPUTE_QUEUE_AUTO=1``), blocks in :class:`Scheduler.run`
    until placed or failed.
    """
    q = queue or Queue()
    job = q.submit(
        str(config_path),
        model=model,
        gpus=gpus if gpus is not None else list(DEFAULT_GPUS),
        spot=spot,
    )
    log.info("[platform-queue] enqueued %s", job.describe())
    if auto_schedule if auto_schedule is not None else os.environ.get(AUTO_SCHEDULE_ENV, "").lower() in ("1", "true", "yes"):
        Scheduler(q).run()
    return job


def maybe_queue_experiment_backend(experiment_cfg: Any) -> dict[str, Any] | None:
    """Hook for experiment launch — queue self-hosted SkyRL when enabled.

    Returns queue metadata when a job was enqueued, else ``None`` (Tinker /
    hosted backends proceed unchanged).
    """
    if not queue_enabled():
        return None
    compute = getattr(experiment_cfg, "compute", None)
    if compute is None:
        return None
    kind = getattr(compute, "kind", None) or (compute.get("kind") if isinstance(compute, dict) else None)
    if kind not in ("skypilot", "sky", "spot"):
        return None
    model = getattr(getattr(experiment_cfg, "model", None), "name", None) or getattr(experiment_cfg, "model", "unknown")
    config_path = getattr(experiment_cfg, "path", None) or getattr(experiment_cfg, "config_path", None)
    if not config_path:
        log.warning("[platform-queue] compute=%s but no config path — skip queue", kind)
        return None
    job = submit_training_config(config_path, model=str(model))
    return {"queued": True, "job_id": job.id, "state": job.state}


__all__ = [
    "AUTO_SCHEDULE_ENV",
    "QUEUE_ENV",
    "maybe_queue_experiment_backend",
    "queue_enabled",
    "submit_training_config",
]
