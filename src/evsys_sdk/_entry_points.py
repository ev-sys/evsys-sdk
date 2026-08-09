"""Load third-party extensions registered via Python entry points.

External packages declare extensions in their pyproject.toml:

    [project.entry-points."evsys_sdk.algorithms"]
    my_dpo = "my_pkg.algorithms:MyDPO"

When evsys_sdk imports, we walk those groups and import each
target module — its top-level @register_* decorators run, populating our
registries. No fork required.

Failures here are non-fatal: a third-party package with an import error
shouldn't break the library. We log a warning and continue.
"""

from __future__ import annotations

import logging
from importlib.metadata import entry_points

logger = logging.getLogger(__name__)

_GROUPS = (
    "evsys_sdk.algorithms",
    "evsys_sdk.verifiers",
    "evsys_sdk.metrics",
    "evsys_sdk.data_stores",
    "evsys_sdk.backends",
    "evsys_sdk.inference",
    "evsys_sdk.transforms",
    "evsys_sdk.trace_sources",
    "evsys_sdk.triggers",
)


def _load_group(group: str) -> None:
    try:
        eps = entry_points(group=group)
    except Exception as e:  # importlib.metadata raises on weird envs
        logger.debug("entry_points lookup failed for %s: %s", group, e)
        return
    for ep in eps:
        try:
            ep.load()
        except Exception as e:
            logger.warning("Failed to load %s entry point %s: %s", group, ep.name, e)


for _g in _GROUPS:
    _load_group(_g)
