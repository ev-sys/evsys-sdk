"""Run several experiments on one GPU node, correctly.

The protocol already permits it: every ``create_lora_training_client()`` against
a SkyRL server is a new adapter sharing the resident base model, so N clients
pointed at one ``base_url`` are N concurrent experiments. What the SDK lacked
was the piece that makes that safe — bring the node up once, run the
experiments against it with the placement rules the benchmarks established, and
tear it down once. Without this, a researcher either rents one GPU per
experiment (the expensive way) or hand-wires threads against a shared URL and
rediscovers each rule below the hard way.

The rules, all measured rather than assumed:

  * **Same base model is mandatory.** Adapters share the resident weights, so a
    9B experiment packed onto a 4B server trains against the wrong model and
    looks fine until the numbers stop making sense. ``Pack`` takes ONE model
    and every experiment gets it.
  * **SFT packs; concurrency defaults to the measured peak.** Aggregate
    throughput peaks at n=4 on every card tested (+1-14%, then declines past
    n=8), and memory is flat in adapter count — so packing buys concurrency,
    roughly n-times wall-clock per experiment, at ~1x the rent.
  * **RL on one GPU cannot pack.** Colocated multi-adapter RL fails outright —
    n=2 and n=4 error where n=1 succeeds on the same server. One GPU, one RL
    experiment, enforced here rather than discovered as an opaque
    "Error retrieving result".
  * **RL on two or more GPUs packs well.** Disaggregated generation scales
    nearly linearly to n=4 (938 -> 1,820 -> 2,371 tok/s measured), because the
    inference engine gets its own card and batches across adapters.
  * **Never exceed the server's adapter slots.** ``max_cpu_loras`` is an LRU
    with no on-demand reload: an evicted adapter 404s on its next request.
    More experiments than slots run in waves instead.

Usage::

    from evsys_sdk.compute import build_compute
    from evsys_sdk.compute.pack import Pack

    compute = build_compute({"kind": "skypilot", "params": {
        "model": "Qwen/Qwen3-4B-Instruct-2507",
        "accelerators": "H100:1", "server_backend": "megatron"}})

    pack = Pack(compute, model="Qwen/Qwen3-4B-Instruct-2507")
    results = pack.run([exp_a, exp_b, exp_c, exp_d])   # one node, 4 adapters

Each experiment is a callable taking the server's ``base_url``; inside it,
construct clients however the run normally would — ``TINKER_BASE_URL`` is set,
so ``ServiceClient()`` with no arguments lands on the shared node.
"""

from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Sequence, TypeVar

from ..logger import get_logger

log = get_logger(__name__)

T = TypeVar("T")

#: Aggregate throughput peaked at n=4 on every card measured (H100, H200, A100)
#: and declined past n=8. More concurrency than this costs wall-clock per
#: experiment without adding throughput, so it is the default, not the maximum.
SFT_PEAK_CONCURRENCY = 4

#: Disaggregated RL generation scaled to n=4 in measurement; beyond that is
#: unmeasured, so the default stops where the data does.
RL_DISAGG_CONCURRENCY = 4


@dataclass
class PackResult:
    """Outcome of one packed experiment, in submission order."""

    name: str
    ok: bool
    value: Any = None
    error: BaseException | None = None


class Pack:
    """One node, several experiments, teardown guaranteed.

    ``compute`` is anything with ``up() -> url`` and ``down()`` — the same
    contract every compute target implements. It is brought up once before the
    first experiment and torn down once after the last, including when
    experiments raise: a failed run that leaks a GPU bills until someone
    notices, and nobody notices at 3am.
    """

    def __init__(self, compute: Any, *, model: str, rl: bool = False,
                 gpus: int | None = None, concurrency: int | None = None,
                 max_adapters: int | None = None):
        self.compute = compute
        self.model = model
        self.rl = rl
        self.gpus = gpus if gpus is not None else _infer_gpus(compute)
        self.max_adapters = (max_adapters if max_adapters is not None
                             else _infer_max_adapters(compute))
        self.concurrency = concurrency

    def _effective_concurrency(self, n_jobs: int) -> int:
        """How many experiments may share the node at once.

        Explicit ``concurrency`` wins, but the RL single-GPU ceiling is a
        correctness bound, not a tuning default — colocated multi-adapter RL
        fails outright — so it caps even an explicit request, loudly.
        """
        if self.rl and (self.gpus or 1) <= 1:
            if self.concurrency and self.concurrency > 1:
                log.warning("[pack] RL on a single GPU cannot run concurrent "
                            "experiments (n>=2 fails colocated); capping "
                            "concurrency from %d to 1", self.concurrency)
            return 1
        cap = self.concurrency or (RL_DISAGG_CONCURRENCY if self.rl
                                   else SFT_PEAK_CONCURRENCY)
        if self.max_adapters:
            if cap > self.max_adapters:
                log.warning("[pack] concurrency %d exceeds the server's %d "
                            "adapter slots; capping — an evicted adapter 404s "
                            "on its next request", cap, self.max_adapters)
            cap = min(cap, self.max_adapters)
        return max(1, min(cap, n_jobs))

    def run(self, experiments: Sequence[Callable[[str], T]],
            *, names: Sequence[str] | None = None,
            raise_on_error: bool = False) -> list[PackResult]:
        """Run every experiment against one shared node.

        Results come back in submission order. A failing experiment records its
        exception and does not stop the others — four experiments sharing a
        node are four independent runs, and one bad config should cost one
        result, not four. ``raise_on_error=True`` raises the first failure
        after everything has finished and the node is down.
        """
        names = list(names) if names is not None else [
            getattr(e, "__name__", f"exp{i}") for i, e in enumerate(experiments)]
        if len(names) != len(experiments):
            raise ValueError("names must match experiments")

        results = [PackResult(name=n, ok=False) for n in names]
        conc = self._effective_concurrency(len(experiments))
        log.info("[pack] %d experiment(s) on one node, %d at a time "
                 "(model=%s rl=%s gpus=%s)", len(experiments), conc,
                 self.model, self.rl, self.gpus or "?")

        url = self.compute.up().rstrip("/")
        # Every client in the SDK (and harbor's rollout engine) constructs
        # ServiceClient() with no args and reads this. One assignment redirects
        # every experiment to the shared node.
        os.environ["TINKER_BASE_URL"] = url
        try:
            lock = threading.Lock()

            def _one(idx: int) -> None:
                try:
                    value = experiments[idx](url)
                    with lock:
                        results[idx] = PackResult(names[idx], True, value)
                except BaseException as e:  # noqa: BLE001
                    log.error("[pack] experiment %r failed: %s: %s",
                              names[idx], type(e).__name__, e)
                    with lock:
                        results[idx] = PackResult(names[idx], False, None, e)

            # Waves fall out of the executor: max_workers IS the adapter
            # ceiling, so an 11th experiment waits for a slot instead of
            # evicting someone's adapter mid-run.
            with ThreadPoolExecutor(max_workers=conc) as ex:
                list(ex.map(_one, range(len(experiments))))
        finally:
            try:
                self.compute.down()
            except Exception as e:  # noqa: BLE001
                log.error("[pack] teardown failed: %s — the node may still be "
                          "billing", e)

        if raise_on_error:
            for r in results:
                if r.error is not None:
                    raise r.error
        return results


def _infer_gpus(compute: Any) -> int | None:
    """GPU count from the compute's own config, if it is discoverable."""
    cfg = getattr(compute, "cfg", None)
    acc = getattr(cfg, "accelerators", None)
    if not acc:
        return None
    first = acc[0] if isinstance(acc, list) else acc
    _, _, n = str(first).partition(":")
    try:
        return int(n) if n else 1
    except ValueError:
        return 1


def _infer_max_adapters(compute: Any) -> int | None:
    cfg = getattr(compute, "cfg", None)
    n = getattr(cfg, "max_adapters", None)
    return int(n) if n else None


__all__ = ["Pack", "PackResult", "RL_DISAGG_CONCURRENCY",
           "SFT_PEAK_CONCURRENCY"]
