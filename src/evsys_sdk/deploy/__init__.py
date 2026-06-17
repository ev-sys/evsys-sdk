"""Post-training deployment — push a trained checkpoint to a serving provider.

A *deployer* is a registry extension (``{kind, params}`` + ``Config``, exactly
like inference/backend). It takes a trained checkpoint URI (e.g. a ``tinker://``
sampler path) and stands the model up on a provider, returning a
:class:`DeployResult` (the served model ref + an OpenAI-compatible endpoint).

Two entry points:

* **Inline** — an experiment-level ``deploy: {kind, params}`` spec; after
  :meth:`evsys_sdk.experiment.Experiment.run` selects ``best_arm`` by
  ``success_metric``, that arm's checkpoint is deployed.
* **Standalone** — :func:`deploy_checkpoint` (and the ``evsys deploy`` CLI):
  pass ``{kind, params}`` + a checkpoint URI directly, no experiment needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar, Protocol, runtime_checkable

from ..registry import get_deployer

# Import built-ins for their registration side effects.
from . import fireworks  # noqa: F401


@dataclass
class DeployResult:
    """Outcome of a deployment."""

    provider: str
    model_ref: str
    """Provider model reference, e.g. ``accounts/<acct>/models/<id>``."""
    endpoint: str = ""
    """OpenAI-compatible base URL for inference against the deployed model."""
    deployment_id: str | None = None
    """The running deployment's ref (None if only uploaded, not deployed)."""
    deployed: bool = False
    """True once a live deployment is up; False if we only uploaded weights."""
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Deployer(Protocol):
    """Push a checkpoint to a serving provider."""

    name: ClassVar[str]
    Config: ClassVar[type]

    def deploy(
        self, checkpoint_uri: str, *,
        base_model: str | None = None, model_id: str | None = None,
    ) -> DeployResult: ...


def build_deployer(spec: Any) -> Deployer:
    """Resolve a ``{kind, params}`` spec (a ``DeploySpec`` or a dict) to a
    constructed deployer, validating ``params`` against the class's ``Config``."""
    kind = spec.kind if hasattr(spec, "kind") else spec["kind"]
    raw = (spec.params if hasattr(spec, "params") else spec.get("params")) or {}
    cls = get_deployer(kind)
    cfg = getattr(cls, "Config", None)
    params = cfg(**raw).model_dump() if cfg is not None else dict(raw)
    return cls(**params)


def deploy_checkpoint(
    kind: str,
    params: dict[str, Any] | None,
    checkpoint_uri: str,
    *,
    base_model: str | None = None,
    model_id: str | None = None,
) -> DeployResult:
    """Standalone deploy: build a deployer from ``{kind, params}`` and push
    ``checkpoint_uri`` to it. Used by the ``evsys deploy`` CLI and directly,
    without running an experiment."""
    deployer = build_deployer({"kind": kind, "params": params or {}})
    return deployer.deploy(checkpoint_uri, base_model=base_model, model_id=model_id)


__all__ = ["DeployResult", "Deployer", "build_deployer", "deploy_checkpoint"]
