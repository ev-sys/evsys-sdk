"""Load third-party extensions registered via Python entry points.

External packages declare extensions in their pyproject.toml:

    [project.entry-points."trajectory_labs.algorithms"]
    my_dpo = "my_pkg.algorithms:MyDPO"

When trajectory_labs imports, we walk those groups and import each
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
    "trajectory_labs.algorithms",
    "trajectory_labs.verifiers",
    "trajectory_labs.metrics",
    "trajectory_labs.data_stores",
    "trajectory_labs.log_stores",
    "trajectory_labs.backends",
    "trajectory_labs.inference",
    "trajectory_labs.transforms",
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
