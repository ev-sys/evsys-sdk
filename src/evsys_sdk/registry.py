"""Registries — one per extension point.

Each registry is a tiny namespace with:
  * ``register_<thing>(name)`` decorator.
  * ``get_<thing>(name)`` lookup.
  * ``list_<thing>s()`` enumeration.

Adding a new algorithm/verifier/metric is one decorator + a Pydantic Config
class. No subclassing the library, no editing any list.

Third-party packages can also register via Python entry points; see
`_entry_points.py` for the loader.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

T = TypeVar("T")


class Registry:
    """Generic name->class registry."""

    def __init__(self, kind: str) -> None:
        self._kind = kind
        self._items: dict[str, type] = {}

    def register(self, name: str | None = None) -> Callable[[type[T]], type[T]]:
        def decorator(cls: type[T]) -> type[T]:
            key = name or getattr(cls, "name", None)
            if not key:
                raise ValueError(
                    f"{self._kind} class {cls.__name__} has no `name` and "
                    f"register_{self._kind} was called without a name."
                )
            if key in self._items and self._items[key] is not cls:
                raise ValueError(
                    f"{self._kind} '{key}' already registered "
                    f"(existing={self._items[key].__module__}.{self._items[key].__name__}, "
                    f"new={cls.__module__}.{cls.__name__})"
                )
            # Best-effort: ensure the class declares its name attribute
            try:
                setattr(cls, "name", key)
            except (TypeError, AttributeError):
                pass
            self._items[key] = cls
            return cls

        return decorator

    def get(self, name: str) -> type:
        if name not in self._items:
            available = ", ".join(sorted(self._items)) or "(none)"
            raise KeyError(
                f"No {self._kind} registered under '{name}'. Available: {available}"
            )
        return self._items[name]

    def list(self) -> list[str]:
        return sorted(self._items)

    def has(self, name: str) -> bool:
        return name in self._items

    def items(self) -> list[tuple[str, type]]:
        return sorted(self._items.items())

    def unregister(self, name: str) -> None:
        """Remove an entry. Mostly for tests."""
        self._items.pop(name, None)


# One registry per extension point.
_algorithms = Registry("algorithm")
_verifiers = Registry("verifier")
_metrics = Registry("metric")
_data_stores = Registry("data_store")
_backends = Registry("backend")
_inference = Registry("inference_client")
_transforms = Registry("transform")
_callbacks = Registry("callback")
_trace_sources = Registry("trace_source")
_triggers = Registry("trigger")
_sandboxes = Registry("sandbox")
_context_sources = Registry("context_source")
_computes = Registry("compute")

# Default inference factories per backend kind. Lets `Experiment` ask
# `get_default_inference_factory("tinker")` and get back a callable
# `(run_result, run_cfg) -> InferenceClient` without having to import the
# tinker module directly (keeping the tinker package optional for users
# that only run mock backends).
_DEFAULT_INFERENCE_FACTORIES: dict[str, Callable[..., Any]] = {}


# Public decorators
def register_algorithm(name: str | None = None):
    return _algorithms.register(name)


def register_verifier(name: str | None = None):
    return _verifiers.register(name)


def register_metric(name: str | None = None):
    return _metrics.register(name)


def register_data_store(name: str | None = None):
    return _data_stores.register(name)


def register_backend(name: str | None = None):
    return _backends.register(name)


def register_inference(name: str | None = None):
    return _inference.register(name)


def register_transform(name: str | None = None):
    return _transforms.register(name)


def register_callback(name: str | None = None):
    return _callbacks.register(name)


def register_trace_source(name: str | None = None):
    return _trace_sources.register(name)


def register_trigger(name: str | None = None):
    return _triggers.register(name)


def register_sandbox(name: str | None = None):
    return _sandboxes.register(name)
def register_context_source(name: str | None = None):
    return _context_sources.register(name)


def register_compute(name: str | None = None):
    """A place to run a training service — SkyPilot, or your own launcher."""
    return _computes.register(name)


def register_default_inference_factory(backend_kind: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Register the default ``(run_result, run_cfg) -> InferenceClient`` for
    a backend kind. Called by each inference module's import side-effect
    (e.g. ``inference/tinker.py`` registers ``"tinker"``).
    """
    def deco(fn: Callable[..., Any]) -> Callable[..., Any]:
        _DEFAULT_INFERENCE_FACTORIES[backend_kind] = fn
        return fn
    return deco


# Public lookups
def get_algorithm(name: str) -> type:
    return _algorithms.get(name)


def get_verifier(name: str) -> type:
    return _verifiers.get(name)


def get_metric(name: str) -> type:
    return _metrics.get(name)


def get_data_store(name: str) -> type:
    return _data_stores.get(name)


def get_backend(name: str) -> type:
    return _backends.get(name)


def get_inference(name: str) -> type:
    return _inference.get(name)


def get_transform(name: str) -> type:
    return _transforms.get(name)


def get_callback(name: str) -> type:
    return _callbacks.get(name)


def get_trace_source(name: str) -> type:
    return _trace_sources.get(name)


def get_trigger(name: str) -> type:
    return _triggers.get(name)


def get_sandbox(name: str) -> type:
    return _sandboxes.get(name)
def get_context_source(name: str) -> type:
    return _context_sources.get(name)
def get_compute(name: str) -> type:
    return _computes.get(name)


def get_default_inference_factory(backend_kind: str) -> Callable[..., Any] | None:
    """Return the registered default factory for ``backend_kind`` or ``None``."""
    return _DEFAULT_INFERENCE_FACTORIES.get(backend_kind)


# Public list functions
def list_algorithms() -> list[str]:
    return _algorithms.list()


def list_verifiers() -> list[str]:
    return _verifiers.list()


def list_metrics() -> list[str]:
    return _metrics.list()


def list_data_stores() -> list[str]:
    return _data_stores.list()


def list_backends() -> list[str]:
    return _backends.list()


def list_inferences() -> list[str]:
    return _inference.list()


def list_transforms() -> list[str]:
    return _transforms.list()


def list_callbacks() -> list[str]:
    return _callbacks.list()


def list_trace_sources() -> list[str]:
    return _trace_sources.list()


def list_triggers() -> list[str]:
    return _triggers.list()


def list_sandboxes() -> list[str]:
    return _sandboxes.list()
def list_context_sources() -> list[str]:
    return _context_sources.list()
def list_computes() -> list[str]:
    return _computes.list()


# Internal helpers used by yaml_loader / runner
def _all_registries() -> dict[str, Registry]:
    return {
        "algorithm": _algorithms,
        "verifier": _verifiers,
        "metric": _metrics,
        "data_store": _data_stores,
        "backend": _backends,
        "inference_client": _inference,
        "transform": _transforms,
        "callback": _callbacks,
        "trace_source": _trace_sources,
        "trigger": _triggers,
        "sandbox": _sandboxes,
        "context_source": _context_sources,
        "compute": _computes,
    }


def schema_for(kind: str, name: str) -> dict[str, Any]:
    """Return the JSON schema for a registered extension's Config.

    Used by evolutionary algorithms to know the legal field space.
    """
    reg = _all_registries().get(kind)
    if reg is None:
        raise KeyError(f"Unknown registry kind: {kind}")
    cls = reg.get(name)
    cfg_cls = getattr(cls, "Config", None)
    if cfg_cls is None or not hasattr(cfg_cls, "model_json_schema"):
        return {"type": "object", "properties": {}}
    return cfg_cls.model_json_schema()
