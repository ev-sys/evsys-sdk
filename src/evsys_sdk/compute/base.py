"""Compute targets — *where the training service runs*.

A backend says which protocol to speak. A **compute target** says whose
hardware speaks it. The two are orthogonal, and separating them is what lets
the identical ``config.yaml`` train against the hosted service, against a
SkyRL server you started by hand, or against one SkyPilot brings up on your
own cloud account or Kubernetes cluster::

    backend:
      kind: skyrl
      params:
        compute:                    # optional — omit and you point at a URL
          kind: skypilot
          params: {infra: aws, accelerators: "L4:1", model: Qwen/Qwen3-0.6B}

The contract is two methods, because that is all a backend needs:

  * ``up()``   → the base URL of a service that is ready to accept requests
  * ``down()`` → release it (idempotent; must never raise)

Everything else — how the machine is provisioned, whether it is a VM, a pod or
a laptop, how the server is installed — belongs to the provider. Providers
follow the repo-wide extension convention (``name`` + ``Config`` ClassVars and
a ``@register_compute`` decorator), so a lab with its own cluster manager
registers one in their project and selects it from YAML by name.
"""

from __future__ import annotations

from typing import Any, ClassVar


class ComputeError(RuntimeError):
    """A compute target could not be brought up."""


class BaseCompute:
    """Somewhere to run a training service.

    Construct via :func:`evsys_sdk.compute.build_compute`, which resolves the
    ``kind`` through the registry and validates ``params`` against ``Config``.
    """

    name: ClassVar[str] = ""
    Config: ClassVar[type | None] = None

    def __init__(self, **params: Any) -> None:
        self.cfg: Any = self.Config(**params) if self.Config is not None else None

    def up(self) -> str:
        """Provision if needed and return the service's base URL.

        Must be idempotent: calling it twice against a target that is already
        running returns the same URL rather than launching a second one.
        """
        raise NotImplementedError

    def down(self) -> None:
        """Release the compute. Best-effort — never raises."""

    # -- context manager, so a caller cannot leak a billed cluster ----------

    def __enter__(self) -> BaseCompute:
        self.up()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.down()


__all__ = ["BaseCompute", "ComputeError"]
