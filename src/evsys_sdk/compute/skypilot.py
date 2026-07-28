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

import time
import urllib.error
import urllib.request
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field

from ..logger import get_logger
from ..registry import register_compute
from .base import BaseCompute, ComputeError

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
    --base-model {model} --backend {server_backend} --port {port}
"""


class SkyPilotComputeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str
    """Base model the SkyRL server loads. Must match the run's model."""
    infra: str | None = None
    """Where to provision: ``aws``, ``gcp``, ``k8s``, ``runpod``, … None lets
    SkyPilot pick the cheapest option you have credentials for."""
    accelerators: str | None = "L4:1"
    """GPU request, SkyPilot syntax (``"A100:1"``). None for a CPU cluster —
    which works, with the JAX backend, for small models and smoke tests."""
    cpus: str | None = None
    memory: str | None = None
    use_spot: bool = False
    """Spot instances are far cheaper and can be preempted mid-run."""
    server_backend: str = "jax"
    """SkyRL execution backend: ``jax`` (single process, CPU or GPU), ``fsdp``
    or ``megatron``. ``fsdp`` needs a second GPU for RL sampling."""
    cluster_name: str = "evsys-skyrl"
    """Reused across runs: an existing cluster is detected and not relaunched."""
    idle_minutes_to_autostop: int = Field(default=30, ge=1)
    """Idle minutes before SkyPilot reclaims the cluster. Never disabled."""
    down: bool = True
    """Terminate on autostop rather than leaving a stopped cluster billing disk."""
    remote_identity: str = "NO_UPLOAD"
    """Whether SkyPilot uploads your cloud credentials to the VM. Do not relax
    this without knowing that agent-authored code runs there."""
    startup_timeout_s: float = Field(default=2400.0, gt=0)
    """Provisioning + model download + engine warmup. Genuinely slow."""
    teardown: bool = True
    """``down()`` terminates the cluster. False leaves it up for reuse (it
    still autostops), which is what you want while iterating."""


@register_compute("skypilot")
class SkyPilotCompute(BaseCompute):
    """A SkyRL training service, on whatever infrastructure SkyPilot can reach."""

    name: ClassVar[str] = "skypilot"
    Config: ClassVar[type] = SkyPilotComputeConfig

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        self._url: str | None = None

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
        extra = "gpu" if self.cfg.accelerators else "jax"
        task = sky.Task(
            name=self.cfg.cluster_name,
            setup=SETUP.format(repo=SKYRL_REPO, extra=extra),
            run=RUN.format(extra=extra, model=self.cfg.model,
                           server_backend=self.cfg.server_backend, port=SERVER_PORT),
        )
        task.set_resources(sky.Resources(
            infra=self.cfg.infra,
            accelerators=self.cfg.accelerators,
            cpus=self.cfg.cpus,
            memory=self.cfg.memory,
            use_spot=self.cfg.use_spot,
            ports=[SERVER_PORT],
        ))
        return task

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

        log.info("[skypilot] launching %s (infra=%s accel=%s model=%s creds=%s)",
                 self.cfg.cluster_name, self.cfg.infra or "auto",
                 self.cfg.accelerators or "cpu", self.cfg.model,
                 self.cfg.remote_identity)
        with self._config_overrides(sky):
            request_id = sky.launch(
                self._task(sky),
                cluster_name=self.cfg.cluster_name,
                idle_minutes_to_autostop=self.cfg.idle_minutes_to_autostop,
                down=self.cfg.down,
                fast=True,           # skip provisioning when it is already up
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
                return url
            time.sleep(10)

        raise ComputeError(
            f"the SkyRL server on '{self.cfg.cluster_name}' did not answer within "
            f"{self.cfg.startup_timeout_s:.0f}s. Check `sky logs {self.cfg.cluster_name}`. "
            f"The cluster is left up and will autostop after "
            f"{self.cfg.idle_minutes_to_autostop} idle minutes."
        )

    def down(self) -> None:
        self._url = None
        if not self.cfg.teardown:
            log.info("[skypilot] leaving %s up (autostops in %d min)",
                     self.cfg.cluster_name, self.cfg.idle_minutes_to_autostop)
            return
        try:
            sky = self._sky()
            sky.stream_and_get(sky.down(self.cfg.cluster_name))
            log.info("[skypilot] %s terminated", self.cfg.cluster_name)
        except Exception as e:
            # Never raise from teardown, but be loud: this one costs money.
            log.warning("[skypilot] could not tear down %s: %s — it will autostop "
                        "after %d idle minutes", self.cfg.cluster_name, e,
                        self.cfg.idle_minutes_to_autostop)


__all__ = ["SkyPilotCompute", "SkyPilotComputeConfig"]
