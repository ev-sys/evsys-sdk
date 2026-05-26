"""TrajectoryStore — HTTP client for the project → goals/datasets/benchmarks →
experiments → groups → runs → checkpoints/evals/metrics hierarchy.

The SDK holds **no Supabase service key**. Every call routes through the
backend gateway (`POST {API}/api/dashboard/api/sdk/data/`) authenticated by the
user's Bearer API key; the backend validates the key, checks the user is a
member of the relevant project, runs the op with its service key, and returns
the data. Same method surface as before — only the transport changed.

Env: ``TRAJECTORY_API_URL`` (backend base URL), ``TRAJECTORY_API_KEY`` (Bearer),
``TRAJECTORY_PROJECT_ID`` (default project for project-scoped ops).
"""

from __future__ import annotations

import os
from typing import Any

import requests

from .constants import (
    API_PREFIX,
    DEFAULT_API_URL,
    DEFAULT_TIMEOUT_S,
    TRAJECTORY_API_KEY_ENV,
    TRAJECTORY_API_URL_ENV,
    TRAJECTORY_PROJECT_ID_ENV,
    bearer,
)

_GATEWAY = API_PREFIX + "/sdk/data/"


class TrajectoryStoreError(RuntimeError):
    def __init__(self, status: int, body: str) -> None:
        super().__init__(f"sdk gateway → HTTP {status}: {body[:300]}")
        self.status, self.body = status, body


class TrajectoryStore:
    def __init__(self, *, base_url: str | None = None, api_key: str | None = None,
                 project_id: str | None = None, timeout_s: float = DEFAULT_TIMEOUT_S) -> None:
        self.base_url = (base_url or os.environ.get(TRAJECTORY_API_URL_ENV)
                         or DEFAULT_API_URL).rstrip("/")
        self.api_key = api_key or os.environ.get(TRAJECTORY_API_KEY_ENV)
        self.project_id = project_id or os.environ.get(TRAJECTORY_PROJECT_ID_ENV)
        self.timeout_s = timeout_s
        if not self.api_key:
            raise TrajectoryStoreError(0, "missing TRAJECTORY_API_KEY")
        self._endpoint = f"{self.base_url}{_GATEWAY}"

    def _call(self, op: str, **args: Any) -> Any:
        clean = {k: v for k, v in args.items() if v is not None}
        r = requests.post(
            self._endpoint,
            headers={"Authorization": bearer(self.api_key), "Content-Type": "application/json"},
            json={"op": op, "args": clean},
            timeout=self.timeout_s,
        )
        if r.status_code >= 400:
            raise TrajectoryStoreError(r.status_code, r.text)
        return (r.json() or {}).get("result")

    def _project(self, project_id: str | None) -> str | None:
        return project_id or self.project_id

    # -- projects -------------------------------------------------------------

    def create_project(self, name: str, *, description: str | None = None,
                        organization_id: str | None = None) -> dict:
        return self._call("create_project", name=name, description=description,
                          organization_id=organization_id)

    def get_project(self, project_id: str | None = None) -> dict | None:
        return self._call("get_project", project_id=self._project(project_id))

    def delete_project(self, project_id: str | None = None) -> Any:
        return self._call("delete_project", project_id=self._project(project_id))

    # -- versioned goals ------------------------------------------------------

    def set_goal(self, goal: str, *, project_id: str | None = None) -> dict:
        return self._call("set_goal", project_id=self._project(project_id), goal=goal)

    def list_goals(self, project_id: str | None = None) -> list[dict]:
        return self._call("list_goals", project_id=self._project(project_id))

    def current_goal(self, project_id: str | None = None) -> dict | None:
        return self._call("current_goal", project_id=self._project(project_id))

    # -- experiments ----------------------------------------------------------

    def create_experiment(self, *, experiment_name: str, project_id: str | None = None,
                          hypothesis: str | None = None, project_goal_id: str | None = None,
                          tags: list[str] | None = None, **extra: Any) -> dict:
        # user_id is set by the backend from the authenticated API key.
        return self._call("create_experiment", project_id=self._project(project_id),
                          experiment_name=experiment_name, hypothesis=hypothesis,
                          project_goal_id=project_goal_id, tags=tags, **extra)

    def get_experiment(self, experiment_id: str) -> dict | None:
        return self._call("get_experiment", experiment_id=experiment_id)

    def update_experiment(self, experiment_id: str, **patch: Any) -> dict:
        return self._call("update_experiment", experiment_id=experiment_id, patch=patch)

    def set_conclusion(self, experiment_id: str, conclusion: str) -> dict:
        return self.update_experiment(experiment_id, conclusion=conclusion)

    def invalidate_experiment(self, experiment_id: str, *, reason: str | None = None) -> dict:
        patch: dict[str, Any] = {"is_valid": False}
        if reason:
            patch["error_message"] = reason
        return self.update_experiment(experiment_id, **patch)

    def list_experiments(self, project_id: str | None = None, *, valid_only: bool = False) -> list[dict]:
        return self._call("list_experiments", project_id=self._project(project_id),
                          valid_only=valid_only)

    def delete_experiment(self, experiment_id: str) -> Any:
        return self._call("delete_experiment", experiment_id=experiment_id)

    def experiment_summaries(self, project_id: str | None = None, *, valid_only: bool = False) -> list[dict]:
        return self._call("experiment_summaries", project_id=self._project(project_id),
                          valid_only=valid_only)

    def experiment_detail(self, experiment_id: str, *, include_metrics: bool = False) -> dict:
        return self._call("experiment_detail", experiment_id=experiment_id,
                          include_metrics=include_metrics)

    # -- groups ---------------------------------------------------------------

    def create_group(self, experiment_id: str, name: str, *, description: str | None = None) -> dict:
        return self._call("create_group", experiment_id=experiment_id, name=name,
                          description=description)

    def list_groups(self, experiment_id: str) -> list[dict]:
        return self._call("list_groups", experiment_id=experiment_id)

    # -- runs -----------------------------------------------------------------

    def create_run(self, *, experiment_id: str, group_id: str | None = None,
                   dataset_id: str | None = None, seed: int | None = None,
                   recipe_kind: str | None = None, run_config: dict | None = None,
                   status: str = "pending", **extra: Any) -> dict:
        return self._call("create_run", experiment_id=experiment_id, group_id=group_id,
                          dataset_id=dataset_id, seed=seed, recipe_kind=recipe_kind,
                          run_config=run_config, status=status, **extra)

    def get_run(self, run_id: str) -> dict | None:
        return self._call("get_run", run_id=run_id)

    def update_run(self, run_id: str, **patch: Any) -> dict:
        return self._call("update_run", run_id=run_id, patch=patch)

    def list_runs(self, *, experiment_id: str | None = None, group_id: str | None = None) -> list[dict]:
        return self._call("list_runs", experiment_id=experiment_id, group_id=group_id)

    # -- training datasets ----------------------------------------------------

    def create_dataset(self, *, name: str, format: str, project_id: str | None = None,
                       version: int = 1, source_kind: str | None = None,
                       transform: list | None = None, storage_uri: str | None = None,
                       metadata: dict | None = None) -> dict:
        return self._call("create_dataset", project_id=self._project(project_id), name=name,
                          format=format, version=version, source_kind=source_kind,
                          transform=transform, storage_uri=storage_uri, metadata=metadata)

    def add_dataset_rows(self, dataset_id: str, rows: list[dict], *, start_idx: int = 0) -> list[dict]:
        return self._call("add_dataset_rows", dataset_id=dataset_id, rows=rows, start_idx=start_idx)

    def get_dataset(self, dataset_id: str) -> dict | None:
        return self._call("get_dataset", dataset_id=dataset_id)

    def list_datasets(self, project_id: str | None = None) -> list[dict]:
        return self._call("list_datasets", project_id=self._project(project_id))

    def get_dataset_rows(self, dataset_id: str, *, limit: int = 100, offset: int = 0) -> list[dict]:
        return self._call("get_dataset_rows", dataset_id=dataset_id, limit=limit, offset=offset)

    # -- benchmarks -----------------------------------------------------------

    def create_benchmark(self, *, name: str, format: str, project_id: str | None = None,
                        version: int = 1, source_kind: str | None = None,
                        transform: list | None = None, storage_uri: str | None = None,
                        metadata: dict | None = None) -> dict:
        return self._call("create_benchmark", project_id=self._project(project_id), name=name,
                          format=format, version=version, source_kind=source_kind,
                          transform=transform, storage_uri=storage_uri, metadata=metadata)

    def add_benchmark_rows(self, benchmark_id: str, rows: list[dict], *, start_idx: int = 0) -> list[dict]:
        return self._call("add_benchmark_rows", benchmark_id=benchmark_id, rows=rows, start_idx=start_idx)

    def get_benchmark(self, benchmark_id: str) -> dict | None:
        return self._call("get_benchmark", benchmark_id=benchmark_id)

    def list_benchmarks(self, project_id: str | None = None) -> list[dict]:
        return self._call("list_benchmarks", project_id=self._project(project_id))

    def get_benchmark_rows(self, benchmark_id: str, *, limit: int = 100, offset: int = 0) -> list[dict]:
        return self._call("get_benchmark_rows", benchmark_id=benchmark_id, limit=limit, offset=offset)

    # -- checkpoints ----------------------------------------------------------

    def add_checkpoint(self, *, run_id: str, uri: str, label: str | None = None,
                      step: int | None = None, base_model: str | None = None,
                      is_final: bool = False) -> dict:
        return self._call("add_checkpoint", run_id=run_id, uri=uri, label=label, step=step,
                          base_model=base_model, is_final=is_final)

    def list_checkpoints(self, run_id: str) -> list[dict]:
        return self._call("list_checkpoints", run_id=run_id)

    # -- evals ----------------------------------------------------------------

    def create_eval(self, *, run_id: str, benchmark_id: str | None = None,
                   checkpoint_id: str | None = None, model_ref: str | None = None,
                   step: int | None = None, metrics: dict | None = None,
                   breakdowns: dict | None = None, sdk_version: str | None = None) -> dict:
        return self._call("create_eval", run_id=run_id, benchmark_id=benchmark_id,
                          checkpoint_id=checkpoint_id, model_ref=model_ref, step=step,
                          metrics=metrics or {}, breakdowns=breakdowns, sdk_version=sdk_version)

    def list_evals(self, run_id: str) -> list[dict]:
        return self._call("list_evals", run_id=run_id)

    # -- per-task predictions -------------------------------------------------

    def add_prediction(self, *, run_id: str, kind: str, eval_id: str | None = None,
                       task_id: str | None = None, instruction: str | None = None,
                       model_output: str | None = None, expected: Any = None,
                       reward: float | None = None, advantage: float | None = None,
                       step: int | None = None, sample_idx: int = 0,
                       metadata: dict | None = None) -> dict:
        return self._call("add_prediction", run_id=run_id, kind=kind, eval_id=eval_id,
                          task_id=task_id, instruction=instruction, model_output=model_output,
                          expected=expected, reward=reward, advantage=advantage, step=step,
                          sample_idx=sample_idx, metadata=metadata)

    def list_predictions(self, run_id: str, *, kind: str | None = None) -> list[dict]:
        return self._call("list_predictions", run_id=run_id, kind=kind)

    # -- training logs: long-format metrics -----------------------------------

    def log_metric(self, *, run_id: str, step: int, name: str, value: float,
                   split: str = "train") -> dict:
        return self._call("log_metric", run_id=run_id, step=step, name=name,
                          value=value, split=split)

    def log_metrics(self, *, run_id: str, step: int, metrics: dict[str, float],
                    split: str = "train") -> list[dict]:
        return self._call("log_metrics", run_id=run_id, step=step, metrics=metrics, split=split)

    def get_metrics(self, run_id: str, *, name: str | None = None,
                    split: str | None = None) -> list[dict]:
        return self._call("get_metrics", run_id=run_id, name=name, split=split)


__all__ = ["TrajectoryStore", "TrajectoryStoreError"]
