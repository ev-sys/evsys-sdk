"""SkyPilot compute target — bring up a SkyRL training service on your infra.

`SkyPilot <https://docs.skypilot.co>`_ is a compute manager: you describe the
resources you want and it provisions them on whichever cloud, Kubernetes
cluster or reserved fleet you have credentials for, and tears them down when
idle. That is exactly the missing half of :mod:`evsys_sdk.backends.skyrl` —
SkyRL speaks the Tinker protocol, SkyPilot finds it a GPU.

Together they make the SDK runnable on infrastructure we do not own::

    backend:
      kind: skyrl
      params:
        compute:
          kind: skypilot
          params:
            infra: aws               # or k8s, gcp, azure, runpod, …
            accelerators: "L4:1"
            model: Qwen/Qwen3-0.6B
            idle_minutes_to_autostop: 15

``up()`` launches a cluster whose ``run`` command is the SkyRL Tinker server,
waits for the port to answer, and returns its URL. The backend then points
``TINKER_BASE_URL`` at it and the whole run — training, sampling, rollouts —
executes on that hardware.

**Cost safety.** SkyPilot clusters live until told otherwise, so
``idle_minutes_to_autostop`` defaults to a real value and ``down=True`` is the
default: an autostopped cluster is *terminated*, not left billing. A crash on
the host still leaves the cluster to autostop on its own.

**Credentials.** SkyPilot uploads local cloud credentials to every VM it
launches unless told not to. This provider defaults ``remote_identity`` to
``NO_UPLOAD`` because the workload here is model-authored code, and shipping a
user's cloud keys to it is not a defensible default.

``sky`` is imported lazily, so the SDK does not depend on it until a project
actually selects this target.
"""

from __future__ import annotations

import json
import shlex
import time
import urllib.error
import urllib.request
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field

from ..logger import get_logger
from ..registry import register_compute
from .base import BaseCompute, ComputeError
from .pricing import PricingUnavailable, cheapest_available, live_offers
from .snapshot import SnapshotPolicy

log = get_logger(__name__)

SERVER_PORT = 8000
"""Port the SkyRL Tinker server listens on, and the one SkyPilot exposes."""

SKYRL_REPO = "https://github.com/NovaSky-AI/SkyRL"

SETUP = """\
set -e
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
if [ ! -d ~/skyrl ]; then git clone --depth 1 {repo} ~/skyrl; fi
cd ~/skyrl && uv sync --extra tinker --extra {extra}
"""

RUN = """\
export PATH="$HOME/.local/bin:$PATH"
cd ~/skyrl
uv run --extra tinker --extra {extra} python -m skyrl.tinker.api \\
    --host 0.0.0.0 --port {port} \\
    --base-model {model} --backend {server_backend}{backend_config}{durable}
"""

#: SkyRL's uv extra per execution backend. `megatron` is the only backend that
#: supports multiple LoRA tenants on one server; `jax` runs on CPU.
BACKEND_EXTRA = {"megatron": "megatron", "fsdp": "fsdp", "jax": "jax"}


class SkyPilotComputeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str
    """Base model the SkyRL server loads. Must match the run's model."""
    infra: str | list[str] | None = None
    """Where to provision: ``aws``, ``gcp``, ``k8s``, ``runpod``, … None lets
    SkyPilot pick the cheapest option you have credentials for.

    **A list is a fallback chain.** Spot capacity is not a price, it is a
    queue — the SKU we benchmarked on simply ceased to exist mid-run. One
    provider is a single point of failure; several are a supply."""
    accelerators: str | list[str] | None = "L4:1"
    """GPU request, SkyPilot syntax (``"A100:1"``). None for a CPU cluster —
    which works, with the JAX backend, for small models and smoke tests.

    **A list is a fallback chain**, e.g. ``["H100:1", "H200:1", "A100-80GB:1"]``.
    Anything that fits the model will do when the cheap one is gone; refusing
    to substitute just means not running."""
    max_hourly_cost: float | None = Field(default=None, gt=0)
    """Hard ceiling on $/hr. SkyPilot rejects any candidate above it, so a
    fallback chain cannot quietly escalate to an expensive machine."""
    managed: bool = False
    """Launch as a **managed job**, which SkyPilot relaunches on preemption.

    A plain ``sky.launch`` cluster is never resurrected — that is why
    ``recover()`` exists. Managed jobs restart the command *from scratch*,
    so this only preserves progress if state is durable (see
    ``checkpoints_path`` and ``database_url``)."""
    job_recovery: str | None = None
    """Managed-job recovery strategy: ``EAGER_NEXT_REGION`` (default; move on
    immediately) or ``FAILOVER`` (keep retrying the same region first)."""
    cpus: str | None = None
    memory: str | None = None
    use_spot: bool = False
    """Spot instances are far cheaper and can be preempted mid-run."""
    multi_lora: bool = True
    """Serve several LoRA adapters from one resident base model.

    On by default because it is the reason to host a server at all: measured on
    an H200, a second adapter costs no additional GPU memory (the slots live in
    CPU memory) and aggregate throughput *rises* with tenant count, since more
    in-flight requests keep the pipeline fed. One adapter per GPU is the
    expensive way to run the same experiments.

    Sets the four knobs SkyRL requires together — ``colocate_all: false``,
    ``merge_lora: false``, ``max_loras`` and ``max_cpu_loras``. Explicit values
    in ``backend_config`` always win, so this is a default, not a policy.

    Requires ``server_backend='megatron'``: multi-tenant LoRA exists on no other
    backend, so this is ignored elsewhere rather than silently misconfiguring."""
    max_adapters: int = Field(default=8, ge=1)
    """Peak concurrent adapters to size the LoRA slots for.

    Size for the PEAK, not the average: ``max_cpu_loras`` is vLLM's LRU
    capacity and there is no on-demand reload, so an evicted adapter makes the
    next ``sample()`` 404. Costs CPU memory only (~hundreds of MB per slot for
    a 4B model at rank 32)."""
    server_backend: str = "jax"
    """SkyRL execution backend: ``jax`` (single process, CPU or GPU), ``fsdp``
    or ``megatron``. ``fsdp`` needs a second GPU for RL sampling.

    **Multi-tenant LoRA exists only on ``megatron``** — several adapters share
    one resident base model, which is the whole reason to host a server rather
    than train one adapter per GPU."""
    backend_config: dict[str, Any] | None = None
    """Passed through to the server as ``--backend-config <json>``.

    Multi-tenant LoRA needs, at minimum, ``merge_lora: false`` (so vLLM serves
    each adapter by name instead of a merged base), plus ``max_loras`` and
    ``max_cpu_loras`` sized to the PEAK number of concurrent tenants — there is
    no on-demand reload, and an evicted adapter makes the next ``sample()``
    404."""
    cluster_name: str = "evsys-skyrl"
    """Reused across runs: an existing cluster is detected and not relaunched."""
    idle_minutes_to_autostop: int = Field(default=30, ge=1)
    """Idle minutes before SkyPilot reclaims the cluster. Never disabled."""
    down: bool = True
    """Terminate on autostop rather than leaving a stopped cluster billing disk."""
    remote_identity: str = "NO_UPLOAD"
    """Whether SkyPilot uploads your cloud credentials to the VM. Do not relax
    this without knowing that agent-authored code runs there."""
    checkpoints_path: str | None = None
    """Where the server writes checkpoints. **Point this off the machine**
    (``s3://…``, ``gs://…``) for anything running on spot.

    The default is a local directory, which is fine until the instance goes
    away — and a spot instance goes away by definition. Preemption then costs
    the whole run, not the last few minutes."""
    database_url: str | None = None
    """Server metadata store. Defaults to SQLite *on the instance*.

    Durable checkpoints alone do not survive preemption: this database maps
    model ids to those checkpoints, so losing it turns them into unreadable
    orphans. Resume needs both off the box — e.g. ``postgresql://…``."""
    snapshot_max_loss_s: float = Field(default=300.0, gt=0)
    """Most work you are willing to redo after a preemption, in seconds.

    A requirement, not an optimisation: it clamps the snapshot interval no
    matter what the cost/MTBF optimum says. The default of five minutes costs
    only about a point of extra overhead over the true optimum at a 2 h MTBF,
    and buys a bounded worst case."""
    snapshot_cost_s: float = Field(default=20.0, gt=0)
    """Starting estimate for what one snapshot costs the training loop.

    Only a seed — :class:`~evsys_sdk.compute.snapshot.SnapshotScheduler`
    replaces it with measured values once the run is going. It matters
    because the first interval is derived from it, and a guess that is too
    low means the run's opening minutes are mostly checkpointing."""
    preemption_mtbf_s: float = Field(default=2 * 3600, gt=0)
    """Expected seconds between preemptions on the pool you are buying from.

    Nobody knows this precisely and it does not need to be precise: the
    optimum moves with its square root, and overhead near the optimum is
    flat. The default assumes spot capacity that turns over every couple of
    hours, which is what we saw."""
    restart_s: float = Field(default=900.0, gt=0)
    """Time from preemption to a replacement actually serving.

    Reported, not optimised — it does not change the cadence, but at a 2 h
    MTBF a 15-minute respawn wastes more than the cadence ever will. If this
    number is large, the fix is a pre-baked image, not more snapshots."""
    price_check: bool = True
    """Ask the provider for real prices and stock before provisioning.

    SkyPilot plans from a pre-generated catalog, which is stale on both price
    and existence. Off only if the provider has no probe and the log noise
    bothers you — it never blocks a launch on its own."""
    retry_until_up: bool = False
    """Keep retrying provisioning instead of failing when capacity is gone.

    Spot capacity comes and goes, so the first attempt failing says little.
    Bounded by ``startup_timeout_s`` rather than left to run forever."""
    wait_for_capacity_s: float | None = None
    """Poll the provider for free GPUs before asking SkyPilot to provision.

    SkyPilot has no notion of availability — its catalog holds prices, and it
    discovers capacity by attempting a launch and failing over. That is fine
    when something is free somewhere, and useless overnight when nothing is:
    it burns a provisioning round-trip per guess and reports the same
    ``ResourcesUnavailableError`` whether the vendor is out of stock or out of
    reach. :mod:`.availability` asks the vendors directly, so the launch is
    attempted when it can plausibly succeed.

    None (default) launches immediately, preserving SkyPilot's own behaviour.
    A number waits up to that many seconds. 0 checks once and reports."""
    startup_timeout_s: float = Field(default=2400.0, gt=0)
    """Provisioning + model download + engine warmup. Genuinely slow."""
    teardown: bool = True
    """``down()`` terminates the cluster. False leaves it up for reuse (it
    still autostops), which is what you want while iterating."""
    max_lifetime_s: float | None = Field(default=6 * 3600, gt=0)
    """Hard deadline after which the cluster is torn down no matter what.

    Not the same as ``idle_minutes_to_autostop``: several providers (notably
    PrimeIntellect) support **no** autostop, autodown or stop at all, so a
    cluster there bills until something explicitly terminates it. A crashed or
    hung host would otherwise leak a GPU indefinitely. Set None only on a
    provider whose own autostop you have verified."""


@register_compute("skypilot")
class SkyPilotCompute(BaseCompute):
    """A SkyRL training service, on whatever infrastructure SkyPilot can reach."""

    name: ClassVar[str] = "skypilot"
    Config: ClassVar[type] = SkyPilotComputeConfig

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        self._url: str | None = None
        self._timer: Any = None
        self._armed = False

    # -- provider plumbing --------------------------------------------------

    def _sky(self) -> Any:
        try:
            import sky
        except ImportError as e:  # pragma: no cover - exercised by the message
            raise ComputeError(
                "the skypilot package is required for compute.kind=skypilot — "
                "install it with `pip install 'evsys-sdk[skypilot]'`"
            ) from e
        return sky

    def _task(self, sky: Any) -> Any:
        # The extra follows the BACKEND, not the presence of a GPU: multi-tenant
        # LoRA only exists on megatron, and asking for `gpu` there installs the
        # wrong stack.
        extra = BACKEND_EXTRA.get(self.cfg.server_backend,
                                  "gpu" if self.cfg.accelerators else "jax")
        # `backend_config` was declared but never reached the command line, so
        # every multi-tenant knob (merge_lora, max_loras, max_cpu_loras) was
        # silently dropped and the server came up single-tenant.
        cfg_arg = ""
        merged = self._server_config()
        if merged:
            cfg_arg = " \\\n    --backend-config " + shlex.quote(json.dumps(merged))
        durable = ""
        if self.cfg.checkpoints_path:
            durable += " \\\n    --checkpoints-base " + shlex.quote(self.cfg.checkpoints_path)
        if self.cfg.database_url:
            durable += " \\\n    --database-url " + shlex.quote(self.cfg.database_url)
        task = sky.Task(
            name=self.cfg.cluster_name,
            setup=SETUP.format(repo=SKYRL_REPO, extra=extra),
            run=RUN.format(extra=extra, model=self.cfg.model,
                           server_backend=self.cfg.server_backend,
                           port=SERVER_PORT, backend_config=cfg_arg,
                           durable=durable),
        )
        # A LIST is `ordered` (try in sequence); a set would be `any_of` and
        # let the optimizer reorder by catalog price — which is the price we
        # already know to be wrong. We order it ourselves, from live data.
        extras: dict[str, Any] = {}
        if self.cfg.max_hourly_cost:
            extras["max_hourly_cost"] = self.cfg.max_hourly_cost
        if self.cfg.managed and self.cfg.job_recovery:
            extras["job_recovery"] = self.cfg.job_recovery
        task.set_resources([
            sky.Resources(
                infra=infra,
                accelerators=accel,
                cpus=self.cfg.cpus,
                memory=self.cfg.memory,
                use_spot=self.cfg.use_spot,
                ports=[SERVER_PORT],
                **extras,
            )
            for infra, accel in self._ordered_candidates()
        ])
        return task

    #: The four knobs multi-tenant LoRA needs, all of them load-bearing.
    #: `merge_lora: true` would have vLLM serve the merged base, so
    #: `sample(model=<adapter>)` silently returns the wrong tenant's weights.
    def _multi_lora_defaults(self) -> dict[str, Any]:
        n = self.cfg.max_adapters
        return {
            "strategy": "megatron",
            "trainer.placement.colocate_all": False,
            "trainer.policy.megatron_config.lora_config.merge_lora": False,
            "trainer.policy.model.lora.max_loras": n,
            "trainer.policy.model.lora.max_cpu_loras": n,
        }

    def _server_config(self) -> dict[str, Any]:
        """Backend config actually sent, defaults under explicit settings."""
        cfg: dict[str, Any] = {}
        if self.cfg.multi_lora and self.cfg.server_backend == "megatron":
            cfg.update(self._multi_lora_defaults())
        elif self.cfg.multi_lora:
            log.info("[skypilot] multi_lora ignored on backend=%s — multi-tenant "
                     "LoRA exists only on megatron", self.cfg.server_backend)
        # The caller's own keys win: a default that overrode an explicit
        # setting would be a trap, not a convenience.
        cfg.update(self.cfg.backend_config or {})
        return cfg

    #: Clouds whose credential upload `remote_identity` controls. SkyPilot
    #: reads it per-cloud, so the setting has to be written for each one.
    _IDENTITY_CLOUDS = ("aws", "gcp", "azure", "kubernetes")

    def _config_overrides(self, sky: Any) -> Any:
        """Apply ``remote_identity`` for the launch.

        It is NOT a ``Resources`` argument — it lives in SkyPilot's config, so
        setting it means overriding config around the launch. Getting this
        wrong is silent: the field would read as a security setting while
        SkyPilot happily uploaded the user's cloud credentials to a VM running
        agent-authored code.
        """
        from sky import skypilot_config

        if not self.cfg.remote_identity:
            from contextlib import nullcontext

            return nullcontext()
        overrides = {c: {"remote_identity": self.cfg.remote_identity}
                     for c in self._IDENTITY_CLOUDS}
        return skypilot_config.override_skypilot_config(overrides)

    def _autostop_supported(self) -> bool:
        """Does the target cloud implement autostop/autodown at all?

        Not every cloud does. PrimeIntellect reports AUTOSTOP, AUTODOWN *and*
        STOP unsupported, so passing ``idle_minutes_to_autostop`` there does
        not merely get ignored — the optimizer rejects every candidate and the
        launch dies with ``ResourcesUnavailableError``. Unknown or unset infra
        is assumed to support it; a wrong guess surfaces as that same loud
        error rather than a silent leak.
        """
        if not self.cfg.infra:
            return True
        try:
            from sky.clouds.cloud import CloudImplementationFeatures as feat
            from sky.utils import registry

            cloud = registry.CLOUD_REGISTRY.from_str(self.cfg.infra.split("/")[0])
            unsupported = getattr(type(cloud), "_CLOUD_UNSUPPORTED_FEATURES", {})
            return feat.AUTODOWN not in unsupported
        except Exception as e:
            log.debug("[skypilot] could not read autostop support for %s: %s",
                      self.cfg.infra, e)
            return True

    def _reclaim_kwargs(self) -> dict[str, Any]:
        """Launch kwargs for cluster reclamation, per the cloud's capabilities."""
        if self._autostop_supported():
            return {"idle_minutes_to_autostop": self.cfg.idle_minutes_to_autostop,
                    "down": self.cfg.down}
        log.warning(
            "[skypilot] %s has no autostop — this cluster bills until it is "
            "explicitly torn down. The %s lifetime cap is the only backstop.",
            self.cfg.infra,
            f"{self.cfg.max_lifetime_s / 3600:.1f} h" if self.cfg.max_lifetime_s
            else "MISSING (max_lifetime_s=None)")
        return {"idle_minutes_to_autostop": None, "down": False}

    def snapshot_policy(self) -> SnapshotPolicy:
        """The cadence a workload on this cluster should snapshot at.

        The compute target cannot *enforce* a cadence — SkyPilot snapshots
        nothing and a managed job restarts the command from scratch, so only
        the training loop can decide to call ``save_weights``. What the
        target can do is own the numbers, since it is the thing that knows
        what was bought and on what terms. The loop asks it, rather than
        every caller re-deriving the arithmetic from its own guesses.
        """
        return SnapshotPolicy(snapshot_cost_s=self.cfg.snapshot_cost_s,
                              mtbf_s=self.cfg.preemption_mtbf_s,
                              max_loss_s=self.cfg.snapshot_max_loss_s,
                              restart_s=self.cfg.restart_s)

    def _warn_if_preemption_would_lose_everything(self) -> None:
        """Spot without durable state is a run you will lose, not a discount.

        Worth saying out loud at launch: preemption gives no notice, and the
        loss is silent — the instance is simply gone, along with every
        checkpoint and the database that indexed them.
        """
        if not self.cfg.use_spot:
            return
        missing = [n for n, v in (("checkpoints_path", self.cfg.checkpoints_path),
                                  ("database_url", self.cfg.database_url)) if not v]
        if missing:
            log.warning(
                "[skypilot] spot instance with no durable %s — a preemption "
                "loses the entire run, with no warning and nothing to resume "
                "from. Point these at storage that outlives the machine "
                "(s3://…, postgresql://…) before running anything long.",
                " or ".join(missing))
            return
        # State is durable, so the interesting question is no longer "will we
        # lose everything" but "what does surviving cost" — which is the one
        # number that decides whether spot was worth it.
        policy = self.snapshot_policy()
        log.info("[skypilot] preemption budget: %s", policy.describe())

    def recover(self) -> str:
        """Bring the service back after the instance went away.

        Spot capacity is reclaimed without notice — that is the deal — so a
        long run needs an answer for "the box vanished". ``sky.launch`` builds
        an *unmanaged* cluster, which SkyPilot does not resurrect on its own,
        so recovery is ours to drive: forget the dead endpoint and launch
        again. Whether anything is actually resumed depends on the state
        having been durable; this restores the *service*, not your progress.
        """
        log.warning("[skypilot] recovering %s — relaunching after the instance "
                    "went away", self.cfg.cluster_name)
        self._url = None
        return self.up()

    def healthy(self) -> bool:
        """Is the service still answering? False after a preemption."""
        return bool(self._url) and self._answers(self._url, timeout=10.0)

    @staticmethod
    def _parse_accel(accel: str | None) -> tuple[str, int] | None:
        """``"A100-80GB:1"`` → ``("A100-80GB", 1)``. None for a CPU cluster."""
        if not accel:
            return None
        name, _, count = str(accel).partition(":")
        try:
            return name, int(count or 1)
        except ValueError:
            return name, 1

    @staticmethod
    def _as_list(v: Any) -> list:
        return list(v) if isinstance(v, list) else [v]

    def _gpu_request(self) -> tuple[str, int] | None:
        return self._parse_accel(self._as_list(self.cfg.accelerators)[0])

    #: Ranking of what the provider told us. Lower tries first.
    _IN_STOCK, _UNKNOWN, _OUT_OF_STOCK = 0, 1, 2

    def _live_price(self, infra: str | None,
                    accel: str | None) -> tuple[int, float | None]:
        """``(status, $/hr)`` for one combination, right now.

        Three outcomes, and conflating them is how today went wrong: a real
        price, "we could not ask", and "the provider says there is none". The
        last is the one SkyPilot reports identically to an unpayable wallet
        and an unsupported feature.
        """
        req = self._parse_accel(accel)
        if not (infra and req):
            return self._UNKNOWN, None
        try:
            best = cheapest_available(infra, req[0], req[1], spot=self.cfg.use_spot)
        except PricingUnavailable:
            return self._UNKNOWN, None
        return (self._IN_STOCK, best.usd_hr) if best else (self._OUT_OF_STOCK, None)

    def _ordered_candidates(self) -> list[tuple[str | None, str | None]]:
        """Every (infra, accelerator) pair, cheapest **live** price first.

        Ordering by the catalog would be ordering by fiction — it had our H100
        at $1.97 when it cost $3.25. Combinations the provider says are out of
        stock, or that we cannot price, keep their declared order and go last:
        unknown is not the same as unavailable, and we would rather try than
        refuse.
        """
        combos = [(i, a) for i in self._as_list(self.cfg.infra)
                  for a in self._as_list(self.cfg.accelerators)]
        if len(combos) == 1:
            return combos
        ranked = []
        for n, (infra, accel) in enumerate(combos):
            status, price = self._live_price(infra, accel)
            # Ties keep declared order; price only separates in-stock offers.
            ranked.append((status, price if price is not None else 0.0, n,
                           infra, accel))
        ranked.sort(key=lambda t: (t[0], t[1], t[2]))
        note = {self._IN_STOCK: "in stock", self._UNKNOWN: "no live price, trying anyway",
                self._OUT_OF_STOCK: "provider reports NONE in stock — trying last"}
        for status, price, _, infra, accel in ranked:
            log.info("[skypilot] candidate %s on %s — %s%s", accel, infra,
                     f"${price:.4f}/hr " if status == self._IN_STOCK else "",
                     note[status])
        return [(i, a) for _, _, _, i, a in ranked]

    def _await_capacity(self) -> None:
        """Block until some vendor has the accelerator free, if asked to.

        Deliberately advisory: it never cancels a launch. Capacity answers go
        stale in about half a minute — every SKU that probed free was refused
        seconds later at some point today — so treating a probe as a veto would
        block launches that would have succeeded. What it buys is knowing
        *when* to try, and being able to wait for hours without hammering the
        provisioning API.
        """
        wait = self.cfg.wait_for_capacity_s
        if wait is None or not self.cfg.accelerators:
            return
        accel = self._as_list(self.cfg.accelerators)[0]
        gpu, _, n = str(accel).partition(":")
        try:
            count = int(n) if n else 1
        except ValueError:
            count = 1
        try:
            from . import availability as av
            found = av.wait_for(gpu, count, timeout_s=wait, poll_s=60)
        except Exception as e:  # noqa: BLE001
            log.debug("[skypilot] capacity check skipped: %s", e)
            return
        ready = [c for c in found if c.ok]
        if ready:
            log.info("[skypilot] %d vendor(s) have %s x%d free; cheapest %s",
                     len(ready), gpu, count, ready[0].describe())
        else:
            # Launching anyway: a probe saying no is weaker evidence than the
            # launch call itself, and SkyPilot may reach clouds we cannot probe.
            log.warning("[skypilot] no vendor reports %s x%d free after %.0fs "
                        "— launching anyway", gpu, count, wait)

    def _refresh_catalog(self) -> None:
        """Update SkyPilot's own price catalog before it plans.

        SkyPilot decides everything — which candidate is cheapest, what order
        to fail over in — from a CSV it downloaded some time ago. Rewriting
        that file with live prices means its optimizer keeps working exactly as
        designed, on numbers that are true. Better than patching SkyPilot, and
        better than routing around it.

        Best-effort: a provider we cannot price leaves the catalog alone, since
        stale prices beat no catalog at all.
        """
        if not (self.cfg.price_check and self.cfg.infra):
            return
        try:
            from . import catalog
            cloud = self._as_list(self.cfg.infra)[0].split("/")[0].lower()
            got = catalog.refresh_if_stale(cloud)
            if got and got.get("updated"):
                log.info("[skypilot] refreshed %d catalog prices for %s before "
                         "planning", got["updated"], cloud)
        except Exception as e:  # noqa: BLE001
            log.debug("[skypilot] catalog refresh skipped: %s", e)

    def _price_check(self) -> None:
        """Log what this will really cost, and say so plainly if it cannot run.

        Advisory by design. A probe that cannot answer, or a provider with no
        probe, must not stop a launch — but when the provider does answer
        "nothing in stock", saying that beats letting SkyPilot report it as the
        same ResourcesUnavailableError it uses for an unpayable wallet.
        """
        req = self._gpu_request()
        infra = self._as_list(self.cfg.infra)[0]
        if not (self.cfg.price_check and infra and req):
            return
        gpu, count = req
        try:
            offers = live_offers(infra, gpu, count, spot=self.cfg.use_spot)
        except PricingUnavailable as e:
            log.info("[skypilot] live pricing unavailable (%s) — using the catalog", e)
            return
        if not offers:
            return
        region = infra.partition("/")[2]
        here = [o for o in offers if not region or o.region.startswith(region)]
        stocked = [o for o in (here or offers) if o.available]
        if not stocked:
            log.warning("[skypilot] the provider reports NO %s x%d in stock%s. "
                        "Nearest offers: %s", gpu, count,
                        f" in {region}" if region else "",
                        "; ".join(o.describe() for o in offers[:3]) or "none at all")
            return
        best = stocked[0]
        log.info("[skypilot] live price: %s (catalog prices are pre-generated "
                 "and routinely stale)", best.describe())

    def _endpoint(self, sky: Any) -> str | None:
        try:
            got = sky.get(sky.endpoints(self.cfg.cluster_name, SERVER_PORT))
        except Exception as e:
            log.debug("[skypilot] endpoint not ready: %s", e)
            return None
        url = got.get(SERVER_PORT) if isinstance(got, dict) else got
        if not url:
            return None
        return url if str(url).startswith("http") else f"http://{url}"

    @staticmethod
    def _answers(url: str, timeout: float = 10.0) -> bool:
        try:
            urllib.request.urlopen(f"{url}/api/v1/get_server_capabilities", timeout=timeout)
        except urllib.error.HTTPError:
            return True          # it replied; any status proves a server is up
        except Exception:
            return False
        return True

    # -- the contract -------------------------------------------------------

    def up(self) -> str:
        if self._url:
            return self._url
        sky = self._sky()

        # An already-running cluster is reused rather than relaunched — this is
        # what makes up() idempotent, and what stops a second run paying to
        # provision a second GPU.
        url = self._endpoint(sky)
        if url and self._answers(url):
            log.info("[skypilot] reusing %s at %s", self.cfg.cluster_name, url)
            self._url = url
            return url

        self._refresh_catalog()
        self._await_capacity()
        self._price_check()
        self._warn_if_preemption_would_lose_everything()
        log.info("[skypilot] launching %s (infra=%s accel=%s model=%s creds=%s)",
                 self.cfg.cluster_name, self.cfg.infra or "auto",
                 self.cfg.accelerators or "cpu", self.cfg.model,
                 self.cfg.remote_identity)
        with self._config_overrides(sky):
            if self.cfg.managed:
                # Managed jobs own their own cluster lifecycle: a controller
                # watches the job (polling every ~15s — SkyPilot consumes no
                # provider interruption notice) and relaunches elsewhere when
                # the instance disappears. The autostop/down knobs belong to
                # unmanaged clusters and are not accepted here.
                request_id = sky.jobs.launch(self._task(sky),
                                             name=self.cfg.cluster_name)
            else:
                request_id = sky.launch(
                    self._task(sky),
                    cluster_name=self.cfg.cluster_name,
                    fast=True,       # skip provisioning when it is already up
                    retry_until_up=self.cfg.retry_until_up,
                    **self._reclaim_kwargs(),
                )
        # The launch request resolves at job SUBMISSION, returning
        # (job_id, handle) — it does not wait for the run command, which is the
        # server and never exits. `get` rather than `stream_and_get` so we do
        # not sit tailing its logs; readiness is decided by polling the port.
        sky.get(request_id)

        deadline = time.monotonic() + self.cfg.startup_timeout_s
        while time.monotonic() < deadline:
            url = self._endpoint(sky)
            if url and self._answers(url):
                log.info("[skypilot] %s serving at %s", self.cfg.cluster_name, url)
                self._url = url
                self._arm_deadline()
                return url
            time.sleep(10)

        raise ComputeError(
            f"the SkyRL server on '{self.cfg.cluster_name}' did not answer within "
            f"{self.cfg.startup_timeout_s:.0f}s. Check `sky logs {self.cfg.cluster_name}`. "
            f"The cluster is left up and will autostop after "
            f"{self.cfg.idle_minutes_to_autostop} idle minutes."
        )

    def _arm_deadline(self) -> None:
        """Belt and braces against a leaked, billing GPU.

        ``down()`` in ``teardown()`` covers the normal path and the context
        manager covers exceptions, but neither survives the host being killed.
        An atexit hook catches interpreter shutdown; a daemon timer catches a
        hang. Both call the same idempotent ``down()``.
        """
        import atexit
        import threading

        if self._armed:
            return
        self._armed = True
        atexit.register(self._down_quietly)
        if self.cfg.max_lifetime_s:
            t = threading.Timer(self.cfg.max_lifetime_s, self._deadline_reached)
            t.daemon = True
            t.start()
            self._timer = t
            log.info("[skypilot] %s will be torn down after %.1f h at the latest",
                     self.cfg.cluster_name, self.cfg.max_lifetime_s / 3600)

    def _deadline_reached(self) -> None:
        log.warning("[skypilot] %s hit its %.1f h lifetime cap — tearing down",
                    self.cfg.cluster_name, (self.cfg.max_lifetime_s or 0) / 3600)
        # force: the deadline overrides teardown=False. `teardown=False` means
        # "leave it up for the next run", not "leave it up forever" — and on a
        # cloud with no autostop, forever is exactly what it would mean.
        self._down_quietly(force=True)

    def _down_quietly(self, force: bool = False) -> None:
        try:
            self.down(force=force)
        except Exception:  # atexit must never raise
            pass

    def down(self, force: bool = False) -> None:
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
        self._url = None
        if not self.cfg.teardown and not force:
            reclaim = ("it autostops in %d min" % self.cfg.idle_minutes_to_autostop
                       if self._autostop_supported()
                       else "THIS CLOUD HAS NO AUTOSTOP — it bills until torn down")
            log.info("[skypilot] leaving %s up (%s)", self.cfg.cluster_name, reclaim)
            return
        try:
            sky = self._sky()
            sky.stream_and_get(sky.down(self.cfg.cluster_name))
            log.info("[skypilot] %s terminated", self.cfg.cluster_name)
        except Exception as e:
            # Never raise from teardown, but be loud: this one costs money.
            log.error("[skypilot] COULD NOT TEAR DOWN %s: %s — it may still be "
                      "billing. Run `sky down %s` by hand; some providers have "
                      "no autostop to fall back on.",
                      self.cfg.cluster_name, e, self.cfg.cluster_name)


__all__ = ["SkyPilotCompute", "SkyPilotComputeConfig"]
