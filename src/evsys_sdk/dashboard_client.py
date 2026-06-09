"""DashboardClient — push SDK runs to the EvolvingSystems backend, with a local mirror.

Architecture: SDK → Django HTTP → Supabase. The user holds an API key (the
same key the dashboard issues at ``Settings → API keys``) plus a project id
(shared by everyone on the project, set in the SDK env). The SDK calls
``/api/dashboard/api/sdk/...`` routes; the backend authenticates the key, checks
the user belongs to the project, and writes Supabase with its service key.

Robustness (wandb-offline style):
  * Every write is **also** mirrored to a local folder (``EVSYS_LOG_DIR``,
    default ``./evsys_sdk``). There is no remote-only mode.
  * If the backend is unreachable (connection error / timeout / 5xx) the call
    logs a warning and keeps going with the local mirror — it does not crash.
  * Auth is required by default: without an API key + project id the client
    raises. Set ``EVSYS_OFFLINE=true`` (or ``offline=True``) to run with no
    auth, writing only to the local mirror.

Env vars (see constants.py): EVSYS_API_URL, EVSYS_API_KEY,
EVSYS_PROJECT_ID, EVSYS_LOG_DIR, EVSYS_OFFLINE,
EVSYS_LOGGING_LEVEL.

Quick usage::

    from evsys_sdk import DashboardClient, ExperimentRun

    client = DashboardClient()  # reads env vars

    with ExperimentRun(client, experiment_name="sft_run_v9",
                       recipe_kind="sft",
                       run_config={"lr": 1e-5, "batch_size": 16}) as run:
        for step in range(1, 1001):
            run.log_step(step, loss=...)
        run.log_eval(metrics={"pass_at_1": 0.83}, benchmark_id="...")
        run.set_best_score(0.83)
"""

from __future__ import annotations

import os
import uuid
from typing import Any

import requests

from .constants import (
    API_PREFIX,
    CONTENT_TYPE_JSON,
    DEFAULT_API_URL,
    DEFAULT_TIMEOUT_S,
    EP_ADD_CHECKPOINT,
    EP_CREATE_EXPERIMENT,
    EP_CREATE_RUN,
    EP_LOG_EVAL,
    EP_LOG_METRICS,
    EP_LOG_PREDICTIONS,
    EP_UPDATE_EXPERIMENT,
    EP_UPDATE_RUN,
    FIELD_BEST_SCORE,
    FIELD_CONCLUSION,
    FIELD_ERROR_MESSAGE,
    FIELD_PROJECT_ID,
    HEADER_AUTHORIZATION,
    HEADER_CONTENT_TYPE,
    KEY_EXPERIMENT,
    KEY_RAW,
    KEY_RUN,
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_PENDING,
    STATUS_RUNNING,
    EVSYS_API_KEY_ENV,
    EVSYS_API_URL_ENV,
    EVSYS_OFFLINE_ENV,
    EVSYS_PROJECT_ID_ENV,
    bearer,
    truthy_env,
)
from .local_store import LocalExperimentStore
from .logger import get_logger

log = get_logger(__name__)


class DashboardClientError(RuntimeError):
    """Raised on a 4xx response (auth / membership / bad request)."""

    def __init__(self, status: int, body: str, path: str) -> None:
        super().__init__(f"{path} → HTTP {status}: {body[:200]}")
        self.status = status
        self.body = body
        self.path = path


class EvsysAuthError(RuntimeError):
    """Raised when credentials are missing and offline mode is not enabled."""


def _new_id() -> str:
    return str(uuid.uuid4())


class DashboardClient:
    """HTTP client for the EvolvingSystems backend's SDK write routes, with a local
    mirror and graceful degradation when the backend is unreachable.
    """

    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        project_id: str | None = None,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        offline: bool | None = None,
        log_dir: str | None = None,
    ) -> None:
        self.base_url = (base_url or os.environ.get(EVSYS_API_URL_ENV) or DEFAULT_API_URL).rstrip("/")
        self.api_key = api_key or os.environ.get(EVSYS_API_KEY_ENV)
        self.project_id = project_id or os.environ.get(EVSYS_PROJECT_ID_ENV)
        self.timeout_s = timeout_s
        self.offline = offline if offline is not None else truthy_env(os.environ.get(EVSYS_OFFLINE_ENV))

        # Always-on local mirror.
        self.local = LocalExperimentStore(log_dir=log_dir)

        if self.offline:
            log.info(
                "evsys_sdk running in OFFLINE mode — writing only to %s",
                self.local.root,
            )
            self._session = None
            return

        # Online mode requires credentials.
        missing = []
        if not self.api_key:
            missing.append(EVSYS_API_KEY_ENV)
        if not self.project_id:
            missing.append(EVSYS_PROJECT_ID_ENV)
        if missing:
            raise EvsysAuthError(
                "DashboardClient is not authenticated: missing "
                + " and ".join(missing)
                + ". Set them (the API key from the dashboard at Settings → API keys, "
                "and the project id shared by your project), or run offline with "
                f"{EVSYS_OFFLINE_ENV}=true to log locally without auth."
            )

        self._session = requests.Session()
        self._session.headers.update({
            HEADER_AUTHORIZATION: bearer(self.api_key),
            HEADER_CONTENT_TYPE: CONTENT_TYPE_JSON,
        })
        # Flips to True after the first connection failure so we stop hammering
        # an unreachable backend within a single run.
        self._remote_down = False
        log.debug("DashboardClient online | base_url=%s project_id=%s", self.base_url, self.project_id)

    # -- low-level ---------------------------------------------------------

    def _post(self, path: str, body: dict | None = None) -> dict | None:
        """POST to the backend. Returns the JSON dict, or None when the write
        could not be sent remotely (offline, or backend unreachable). Raises
        DashboardClientError on a 4xx response.
        """
        if self.offline or self._session is None or self._remote_down:
            return None
        url = f"{self.base_url}{API_PREFIX}{path}"
        try:
            r = self._session.post(url, json=body or {}, timeout=self.timeout_s)
        except requests.exceptions.RequestException as e:
            self._remote_down = True
            log.warning(
                "backend unreachable (%s) — falling back to local mirror at %s for the rest of this run",
                e.__class__.__name__,
                self.local.root,
            )
            return None
        if 200 <= r.status_code < 300:
            try:
                return r.json()
            except Exception:
                return {KEY_RAW: r.text}
        if 400 <= r.status_code < 500:
            # Real client error (auth, not-a-member, bad payload): surface it.
            raise DashboardClientError(r.status_code, r.text, path)
        # 5xx — treat like an outage and degrade to local.
        log.warning("backend error HTTP %s on %s — falling back to local mirror", r.status_code, path)
        self._remote_down = True
        return None

    # -- experiments -------------------------------------------------------

    def create_experiment(
        self,
        *,
        experiment_name: str,
        hypothesis: str | None = None,
        hypothesis_reasoning: str | None = None,
        plan: str | None = None,
        tags: list[str] | None = None,
        project_goal_id: str | None = None,
        config: dict | None = None,
        declaration_path: str | None = None,
        is_valid: bool | None = None,
    ) -> dict:
        # NB: base_model is a per-run hyperparameter (runs.run_config), not an
        # experiment field. client / problem_statement_id / parent_experiment_id /
        # seed_config were dropped from the schema (D1/D3/D18).
        body: dict = {"experiment_name": experiment_name}
        if self.project_id:
            body[FIELD_PROJECT_ID] = self.project_id
        for k, v in (
            ("hypothesis", hypothesis),
            ("hypothesis_reasoning", hypothesis_reasoning), ("plan", plan),
            ("tags", tags), ("project_goal_id", project_goal_id),
            ("config", config), ("declaration_path", declaration_path),
            ("is_valid", is_valid),
        ):
            if v is not None:
                body[k] = v

        resp = self._post(EP_CREATE_EXPERIMENT, body)
        exp = (resp or {}).get(KEY_EXPERIMENT) or {}
        exp_id = exp.get("id") or _new_id()
        exp = {**body, **exp, "id": exp_id}
        self.local.create_experiment(exp_id, body)
        return exp

    def update_experiment(self, experiment_id: str, **patch: Any) -> dict:
        """PATCH experiment metadata. Whitelisted backend fields: status,
        best_score, best_generation_id, current_iteration, error_message,
        hypothesis, hypothesis_reasoning, plan, conclusion, tags,
        problem_statement_id, is_valid.

        e.g. ``update_experiment(exp_id, is_valid=False)`` to invalidate an
        experiment after a bug is found in its runs (D16).
        """
        resp = self._post(EP_UPDATE_EXPERIMENT.format(experiment_id=experiment_id), patch)
        self.local.update_experiment(experiment_id, patch)
        return resp or {"id": experiment_id, **patch}

    # -- generations -------------------------------------------------------

    def create_run(
        self,
        *,
        experiment_id: str,
        group_id: str | None = None,
        dataset_id: str | None = None,
        seed: int | None = None,
        recipe_kind: str | None = None,
        run_config: dict | None = None,
        status: str = STATUS_PENDING,
        wandb_run_url: str | None = None,
        tensorboard_path: str | None = None,
        tensorboard_archive_url: str | None = None,
    ) -> dict:
        body: dict = {"experiment_id": experiment_id, "status": status}
        for k, v in (
            ("group_id", group_id), ("dataset_id", dataset_id), ("seed", seed),
            ("recipe_kind", recipe_kind), ("run_config", run_config),
            ("wandb_run_url", wandb_run_url), ("tensorboard_path", tensorboard_path),
            ("tensorboard_archive_url", tensorboard_archive_url),
        ):
            if v is not None:
                body[k] = v

        resp = self._post(EP_CREATE_RUN, body)
        run = (resp or {}).get(KEY_RUN) or {}
        run_id = run.get("id") or _new_id()
        run = {**body, **run, "id": run_id}
        self.local.create_run(run_id, body)
        return run

    def update_run(self, run_id: str, **patch: Any) -> dict:
        resp = self._post(EP_UPDATE_RUN.format(run_id=run_id), patch)
        self.local.update_run(run_id, patch)
        return resp or {"id": run_id, **patch}

    # -- logging -----------------------------------------------------------

    def log_step_metric(
        self,
        run_id: str,
        *,
        step: int,
        split: str = "train",
        loss: float | None = None,
        accuracy: float | None = None,
        learning_rate: float | None = None,
        grad_norm: float | None = None,
        tokens_per_sec: float | None = None,
        **metrics: float,
    ) -> dict:
        """Log per-step training metrics as long-format run_metrics (D15).

        Accepts arbitrary named series via ``**metrics`` (e.g. ``val_loss=…``,
        ``kl_loss=…``) plus a ``split`` (``train``/``val``/``test``). The named
        kwargs (loss, accuracy, …) are folded into the same ``metrics`` map.
        None values are dropped.
        """
        named = {
            "loss": loss, "accuracy": accuracy, "learning_rate": learning_rate,
            "grad_norm": grad_norm, "tokens_per_sec": tokens_per_sec,
        }
        merged: dict[str, float] = {
            k: v for k, v in {**named, **metrics}.items() if v is not None
        }
        body: dict = {"step": int(step), "split": split, "metrics": merged}
        resp = self._post(EP_LOG_METRICS.format(run_id=run_id), body)
        self.local.log_step(run_id, body)
        return resp or {"ok": True}

    def create_eval(
        self,
        run_id: str,
        *,
        metrics: dict[str, float],
        benchmark_id: str | None = None,
        checkpoint_id: str | None = None,
        model_ref: str | None = None,
        step: int | None = None,
        breakdowns: dict | None = None,
        sdk_version: str | None = None,
    ) -> dict:
        """Record an eval (D13): a run scored on a benchmark → metrics{name:value}."""
        body: dict = {"metrics": dict(metrics)}
        for k, v in (
            ("benchmark_id", benchmark_id), ("checkpoint_id", checkpoint_id),
            ("model_ref", model_ref), ("step", step), ("breakdowns", breakdowns),
            ("sdk_version", sdk_version),
        ):
            if v is not None:
                body[k] = v
        resp = self._post(EP_LOG_EVAL.format(run_id=run_id), body)
        self.local.log_eval(run_id, body)
        return resp or {"ok": True}

    def add_checkpoint(
        self,
        run_id: str,
        *,
        uri: str,
        label: str | None = None,
        step: int | None = None,
        base_model: str | None = None,
        is_final: bool = False,
    ) -> dict:
        """Record a named checkpoint for a run (D7)."""
        body: dict = {"uri": uri, "is_final": is_final}
        for k, v in (("label", label), ("step", step), ("base_model", base_model)):
            if v is not None:
                body[k] = v
        resp = self._post(EP_ADD_CHECKPOINT.format(run_id=run_id), body)
        return resp or {"ok": True}

    def log_predictions(
        self,
        run_id: str,
        predictions: list[dict],
    ) -> dict:
        """Bulk-insert per-task predictions. Each is:
            { kind: 'eval' | 'rollout', eval_id?, task_id?, sample_idx?, step?,
              instruction, model_output, expected?, reward?, advantage?, metadata? }
        """
        if not predictions:
            return {"ok": True, "inserted": 0}
        resp = self._post(
            EP_LOG_PREDICTIONS.format(run_id=run_id),
            {"predictions": predictions},
        )
        self.local.log_predictions(run_id, predictions)
        return resp or {"ok": True, "inserted": len(predictions)}


# ---------------------------------------------------------------------------
# Ergonomic context-managed run
# ---------------------------------------------------------------------------


class ExperimentRun:
    """One generation in one experiment, lifecycle-managed.

    Use as a context manager. On clean exit, the generation + experiment are
    marked completed (with the best score / generation_id). On raised
    exception, both are marked failed with the exception message. All writes
    are mirrored locally and degrade gracefully if the backend is down.
    """

    def __init__(
        self,
        client: DashboardClient,
        *,
        experiment_name: str,
        hypothesis: str | None = None,
        hypothesis_reasoning: str | None = None,
        plan: str | None = None,
        tags: list[str] | None = None,
        project_goal_id: str | None = None,
        # run-level
        group_id: str | None = None,
        dataset_id: str | None = None,
        seed: int | None = None,
        recipe_kind: str | None = None,
        run_config: dict | None = None,
        wandb_run_url: str | None = None,
        tensorboard_path: str | None = None,
        # threading an existing experiment for multi-run campaigns
        experiment_id: str | None = None,
    ) -> None:
        self.client = client
        self._exp_kwargs = {
            "experiment_name": experiment_name,
            "hypothesis": hypothesis,
            "hypothesis_reasoning": hypothesis_reasoning,
            "plan": plan,
            "tags": tags,
            "project_goal_id": project_goal_id,
        }
        self._run_kwargs = {
            "group_id": group_id,
            "dataset_id": dataset_id,
            "seed": seed,
            "recipe_kind": recipe_kind,
            "run_config": run_config,
            "wandb_run_url": wandb_run_url,
            "tensorboard_path": tensorboard_path,
        }
        self._given_experiment_id = experiment_id
        self.experiment_id: str | None = None
        self.run_id: str | None = None
        self._best_score: float | None = None
        self._conclusion: str | None = None

    def __enter__(self) -> "ExperimentRun":
        if self._given_experiment_id:
            self.experiment_id = self._given_experiment_id
        else:
            exp = self.client.create_experiment(**{k: v for k, v in self._exp_kwargs.items() if v is not None})  # type: ignore[arg-type]
            self.experiment_id = exp["id"]
        assert self.experiment_id is not None
        run = self.client.create_run(
            experiment_id=self.experiment_id,
            status=STATUS_RUNNING,
            **{k: v for k, v in self._run_kwargs.items() if v is not None},  # type: ignore[arg-type]
        )
        self.run_id = run["id"]
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if exc_type is not None:
            msg = f"{exc_type.__name__}: {exc_val}" if exc_val else exc_type.__name__
            if self.run_id:
                try:
                    self.client.update_run(self.run_id, status=STATUS_FAILED, **{FIELD_ERROR_MESSAGE: msg})
                except Exception:
                    pass
            if self.experiment_id:
                try:
                    self.client.update_experiment(self.experiment_id, status=STATUS_FAILED, **{FIELD_ERROR_MESSAGE: msg})
                except Exception:
                    pass
            return None
        # Clean exit. best_score lives on the experiment; the run's headline
        # number (if any) goes in runs.summary.
        if self.run_id:
            run_patch: dict = {"status": STATUS_COMPLETED}
            if self._best_score is not None:
                run_patch["summary"] = {"best_score": self._best_score}
            self.client.update_run(self.run_id, **run_patch)
        if self.experiment_id:
            exp_patch: dict = {"status": STATUS_COMPLETED}
            if self._best_score is not None:
                exp_patch[FIELD_BEST_SCORE] = self._best_score
            if self._conclusion is not None:
                exp_patch[FIELD_CONCLUSION] = self._conclusion
            self.client.update_experiment(self.experiment_id, **exp_patch)

    # -- ergonomic passthroughs --

    def log_step(self, step: int, **metrics: Any) -> None:
        assert self.run_id
        self.client.log_step_metric(self.run_id, step=step, **metrics)

    def log_eval(
        self,
        *,
        metrics: dict[str, float],
        benchmark_id: str | None = None,
        checkpoint_id: str | None = None,
        model_ref: str | None = None,
        step: int | None = None,
    ) -> None:
        assert self.run_id
        self.client.create_eval(self.run_id, metrics=metrics, benchmark_id=benchmark_id,
                                checkpoint_id=checkpoint_id, model_ref=model_ref, step=step)

    def add_checkpoint(self, *, uri: str, label: str | None = None, step: int | None = None,
                       base_model: str | None = None, is_final: bool = False) -> None:
        assert self.run_id
        self.client.add_checkpoint(self.run_id, uri=uri, label=label, step=step,
                                   base_model=base_model, is_final=is_final)

    def log_predictions(self, predictions: list[dict]) -> None:
        assert self.run_id
        self.client.log_predictions(self.run_id, predictions)

    def set_best_score(self, score: float) -> None:
        self._best_score = float(score)

    def set_conclusion(self, text: str) -> None:
        """One- or two-line takeaway from the run. Flushed on clean __exit__."""
        self._conclusion = str(text)

    def update_run(self, **patch: Any) -> None:
        assert self.run_id
        self.client.update_run(self.run_id, **patch)

    def update_experiment(self, **patch: Any) -> None:
        assert self.experiment_id
        self.client.update_experiment(self.experiment_id, **patch)


__all__ = ["DashboardClient", "DashboardClientError", "EvsysAuthError", "ExperimentRun"]
