"""DashboardClient — push SDK runs to the Trajectory backend, with a local mirror.

Architecture: SDK → Django HTTP → Supabase. The user holds an API key (the
same key the dashboard issues at ``Settings → API keys``) plus a project id
(shared by everyone on the project, set in the SDK env). The SDK calls
``/api/dashboard/api/sdk/...`` routes; the backend authenticates the key, checks
the user belongs to the project, and writes Supabase with its service key.

Robustness (wandb-offline style):
  * Every write is **also** mirrored to a local folder (``TRAJECTORY_LOG_DIR``,
    default ``./trajectory_experiments``). There is no remote-only mode.
  * If the backend is unreachable (connection error / timeout / 5xx) the call
    logs a warning and keeps going with the local mirror — it does not crash.
  * Auth is required by default: without an API key + project id the client
    raises. Set ``TRAJECTORY_OFFLINE=true`` (or ``offline=True``) to run with no
    auth, writing only to the local mirror.

Env vars (see constants.py): TRAJECTORY_API_URL, TRAJECTORY_API_KEY,
TRAJECTORY_PROJECT_ID, TRAJECTORY_LOG_DIR, TRAJECTORY_OFFLINE,
TRAJECTORY_LOGGING_LEVEL.

Quick usage::

    from trajectory_experiments import DashboardClient, ExperimentRun

    client = DashboardClient()  # reads env vars

    with ExperimentRun(client, experiment_name="composio_sft_v9",
                       recipe_kind="sft",
                       run_config={"lr": 1e-5, "batch_size": 16}) as run:
        for step in range(1, 1001):
            run.log_step(step, loss=...)
        run.log_eval(eval_name="composio_eval_v3", metrics={"pass_at_1": 0.83})
        run.set_best_score(0.83)
"""

from __future__ import annotations

import os
import uuid
from contextlib import contextmanager
from typing import Any

import requests

from .constants import (
    API_PREFIX,
    CONTENT_TYPE_JSON,
    DEFAULT_API_URL,
    DEFAULT_TIMEOUT_S,
    EP_CREATE_EXPERIMENT,
    EP_CREATE_GENERATION,
    EP_LOG_EVAL,
    EP_LOG_PREDICTIONS,
    EP_LOG_STEP,
    EP_RECORD_BENCHMARK,
    EP_UPDATE_EXPERIMENT,
    EP_UPDATE_GENERATION,
    FIELD_BEST_GENERATION_ID,
    FIELD_BEST_SCORE,
    FIELD_COMBINED_SCORE,
    FIELD_CONCLUSION,
    FIELD_ERROR_MESSAGE,
    FIELD_PROJECT_ID,
    HEADER_AUTHORIZATION,
    HEADER_CONTENT_TYPE,
    KEY_EXPERIMENT,
    KEY_GENERATION,
    KEY_RAW,
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_PENDING,
    STATUS_RUNNING,
    TRAJECTORY_API_KEY_ENV,
    TRAJECTORY_API_URL_ENV,
    TRAJECTORY_OFFLINE_ENV,
    TRAJECTORY_PROJECT_ID_ENV,
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


class TrajectoryAuthError(RuntimeError):
    """Raised when credentials are missing and offline mode is not enabled."""


def _new_id() -> str:
    return str(uuid.uuid4())


class DashboardClient:
    """HTTP client for the Trajectory backend's SDK write routes, with a local
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
        self.base_url = (base_url or os.environ.get(TRAJECTORY_API_URL_ENV) or DEFAULT_API_URL).rstrip("/")
        self.api_key = api_key or os.environ.get(TRAJECTORY_API_KEY_ENV)
        self.project_id = project_id or os.environ.get(TRAJECTORY_PROJECT_ID_ENV)
        self.timeout_s = timeout_s
        self.offline = offline if offline is not None else truthy_env(os.environ.get(TRAJECTORY_OFFLINE_ENV))

        # Always-on local mirror.
        self.local = LocalExperimentStore(log_dir=log_dir)

        if self.offline:
            log.info(
                "trajectory_experiments running in OFFLINE mode — writing only to %s",
                self.local.root,
            )
            self._session = None
            return

        # Online mode requires credentials.
        missing = []
        if not self.api_key:
            missing.append(TRAJECTORY_API_KEY_ENV)
        if not self.project_id:
            missing.append(TRAJECTORY_PROJECT_ID_ENV)
        if missing:
            raise TrajectoryAuthError(
                "DashboardClient is not authenticated: missing "
                + " and ".join(missing)
                + ". Set them (the API key from the dashboard at Settings → API keys, "
                "and the project id shared by your project), or run offline with "
                f"{TRAJECTORY_OFFLINE_ENV}=true to log locally without auth."
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
        client: str | None = None,
        hypothesis: str | None = None,
        hypothesis_reasoning: str | None = None,
        plan: str | None = None,
        tags: list[str] | None = None,
        problem_statement_id: str | None = None,
        parent_experiment_id: str | None = None,
        base_model: str | None = None,
        seed_config: dict | None = None,
        config: dict | None = None,
        declaration_path: str | None = None,
    ) -> dict:
        body: dict = {"experiment_name": experiment_name}
        if self.project_id:
            body[FIELD_PROJECT_ID] = self.project_id
        for k, v in (
            ("client", client), ("hypothesis", hypothesis),
            ("hypothesis_reasoning", hypothesis_reasoning), ("plan", plan),
            ("tags", tags),
            ("problem_statement_id", problem_statement_id),
            ("parent_experiment_id", parent_experiment_id),
            ("base_model", base_model), ("seed_config", seed_config),
            ("config", config), ("declaration_path", declaration_path),
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
        problem_statement_id.
        """
        resp = self._post(EP_UPDATE_EXPERIMENT.format(experiment_id=experiment_id), patch)
        self.local.update_experiment(experiment_id, patch)
        return resp or {"id": experiment_id, **patch}

    # -- generations -------------------------------------------------------

    def create_generation(
        self,
        *,
        experiment_id: str,
        recipe_kind: str | None = None,
        run_config: dict | None = None,
        iteration: int | None = None,
        parent_id: str | None = None,
        status: str = STATUS_PENDING,
        wandb_run_url: str | None = None,
        tensorboard_path: str | None = None,
        resolved_data_path: str | None = None,
        checkpoint_url: str | None = None,
        tinker_run_id: str | None = None,
        variation_kind: str | None = None,
    ) -> dict:
        body: dict = {"experiment_id": experiment_id, "status": status}
        for k, v in (
            ("recipe_kind", recipe_kind), ("run_config", run_config),
            ("iteration", iteration), ("parent_id", parent_id),
            ("wandb_run_url", wandb_run_url), ("tensorboard_path", tensorboard_path),
            ("resolved_data_path", resolved_data_path),
            ("checkpoint_url", checkpoint_url), ("tinker_run_id", tinker_run_id),
            ("variation_kind", variation_kind),
        ):
            if v is not None:
                body[k] = v

        resp = self._post(EP_CREATE_GENERATION, body)
        gen = (resp or {}).get(KEY_GENERATION) or {}
        gen_id = gen.get("id") or _new_id()
        gen = {**body, **gen, "id": gen_id}
        self.local.create_generation(gen_id, body)
        return gen

    def update_generation(self, generation_id: str, **patch: Any) -> dict:
        resp = self._post(EP_UPDATE_GENERATION.format(generation_id=generation_id), patch)
        self.local.update_generation(generation_id, patch)
        return resp or {"id": generation_id, **patch}

    # -- logging -----------------------------------------------------------

    def log_step_metric(
        self,
        generation_id: str,
        *,
        step: int,
        loss: float | None = None,
        accuracy: float | None = None,
        learning_rate: float | None = None,
        grad_norm: float | None = None,
        tokens_per_sec: float | None = None,
    ) -> dict:
        body: dict = {"step": int(step)}
        for k, v in (
            ("loss", loss), ("accuracy", accuracy),
            ("learning_rate", learning_rate), ("grad_norm", grad_norm),
            ("tokens_per_sec", tokens_per_sec),
        ):
            if v is not None:
                body[k] = v
        resp = self._post(EP_LOG_STEP.format(generation_id=generation_id), body)
        self.local.log_step(generation_id, body)
        return resp or {"ok": True}

    def log_eval_run(
        self,
        generation_id: str,
        *,
        eval_name: str,
        metrics: dict[str, float],
        step: int | None = None,
    ) -> dict:
        body: dict = {"eval_name": eval_name, "metrics": dict(metrics)}
        if step is not None:
            body["step"] = int(step)
        resp = self._post(EP_LOG_EVAL.format(generation_id=generation_id), body)
        self.local.log_eval(generation_id, body)
        return resp or {"ok": True}

    def log_predictions(
        self,
        generation_id: str,
        predictions: list[dict],
    ) -> dict:
        """Bulk-insert. Each prediction is:
            { kind: 'eval' | 'rollout',
              task_id?: str, sample_idx?: int, step?: int, eval_name?: str,
              instruction: str, model_output: str, expected?: any,
              reward?: float, advantage?: float, metadata?: dict }
        """
        if not predictions:
            return {"ok": True, "inserted": 0}
        resp = self._post(
            EP_LOG_PREDICTIONS.format(generation_id=generation_id),
            {"predictions": predictions},
        )
        self.local.log_predictions(generation_id, predictions)
        return resp or {"ok": True, "inserted": len(predictions)}

    # -- benchmark (leaderboard) -------------------------------------------

    def record_benchmark(
        self,
        *,
        test_dataset_id: str,
        model_ref: str,
        score: float | None = None,
        n_passed: int | None = None,
        n_total: int | None = None,
        generation_id: str | None = None,
        step: int | None = None,
    ) -> dict:
        """Push an EXPLICIT scoring event to the leaderboard. Provide ``score``,
        or ``n_passed`` + ``n_total`` (backend derives score = n_passed/n_total).
        """
        body: dict = {"test_dataset_id": test_dataset_id, "model_ref": model_ref}
        for k, v in (
            ("score", score), ("n_passed", n_passed), ("n_total", n_total),
            ("generation_id", generation_id), ("step", step),
        ):
            if v is not None:
                body[k] = v
        resp = self._post(EP_RECORD_BENCHMARK, body)
        return resp or {"ok": True}


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
        client_name: str | None = None,
        hypothesis: str | None = None,
        hypothesis_reasoning: str | None = None,
        plan: str | None = None,
        tags: list[str] | None = None,
        problem_statement_id: str | None = None,
        base_model: str | None = None,
        # generation-level
        recipe_kind: str | None = None,
        run_config: dict | None = None,
        iteration: int | None = None,
        parent_generation_id: str | None = None,
        wandb_run_url: str | None = None,
        tensorboard_path: str | None = None,
        resolved_data_path: str | None = None,
        # threading an existing experiment for multi-gen sweeps
        experiment_id: str | None = None,
    ) -> None:
        self.client = client
        self._exp_kwargs = {
            "experiment_name": experiment_name,
            "client": client_name,
            "hypothesis": hypothesis,
            "hypothesis_reasoning": hypothesis_reasoning,
            "plan": plan,
            "tags": tags,
            "problem_statement_id": problem_statement_id,
            "base_model": base_model,
        }
        self._gen_kwargs = {
            "recipe_kind": recipe_kind,
            "run_config": run_config,
            "iteration": iteration,
            "parent_id": parent_generation_id,
            "wandb_run_url": wandb_run_url,
            "tensorboard_path": tensorboard_path,
            "resolved_data_path": resolved_data_path,
        }
        self._given_experiment_id = experiment_id
        self.experiment_id: str | None = None
        self.generation_id: str | None = None
        self._best_score: float | None = None
        self._conclusion: str | None = None

    def __enter__(self) -> "ExperimentRun":
        if self._given_experiment_id:
            self.experiment_id = self._given_experiment_id
        else:
            exp = self.client.create_experiment(**{k: v for k, v in self._exp_kwargs.items() if v is not None})  # type: ignore[arg-type]
            self.experiment_id = exp["id"]
        assert self.experiment_id is not None
        gen = self.client.create_generation(
            experiment_id=self.experiment_id,
            status=STATUS_RUNNING,
            **{k: v for k, v in self._gen_kwargs.items() if v is not None},  # type: ignore[arg-type]
        )
        self.generation_id = gen["id"]
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if exc_type is not None:
            msg = f"{exc_type.__name__}: {exc_val}" if exc_val else exc_type.__name__
            if self.generation_id:
                try:
                    self.client.update_generation(self.generation_id, status=STATUS_FAILED, **{FIELD_ERROR_MESSAGE: msg})
                except Exception:
                    pass
            if self.experiment_id:
                try:
                    self.client.update_experiment(self.experiment_id, status=STATUS_FAILED, **{FIELD_ERROR_MESSAGE: msg})
                except Exception:
                    pass
            return None
        # Clean exit. training_generations has no best_score column —
        # combined_score is the headline metric there. training_experiments
        # is where best_score lives.
        if self.generation_id:
            gen_patch: dict = {"status": STATUS_COMPLETED}
            if self._best_score is not None:
                gen_patch[FIELD_COMBINED_SCORE] = self._best_score
            self.client.update_generation(self.generation_id, **gen_patch)
        if self.experiment_id:
            exp_patch: dict = {"status": STATUS_COMPLETED}
            if self._best_score is not None:
                exp_patch[FIELD_BEST_SCORE] = self._best_score
            if self.generation_id:
                exp_patch[FIELD_BEST_GENERATION_ID] = self.generation_id
            if self._conclusion is not None:
                exp_patch[FIELD_CONCLUSION] = self._conclusion
            self.client.update_experiment(self.experiment_id, **exp_patch)

    # -- ergonomic passthroughs --

    def log_step(self, step: int, **metrics: Any) -> None:
        assert self.generation_id
        self.client.log_step_metric(self.generation_id, step=step, **metrics)

    def log_eval(
        self,
        *,
        eval_name: str,
        metrics: dict[str, float],
        step: int | None = None,
    ) -> None:
        assert self.generation_id
        self.client.log_eval_run(self.generation_id, eval_name=eval_name, metrics=metrics, step=step)

    def log_predictions(self, predictions: list[dict]) -> None:
        assert self.generation_id
        self.client.log_predictions(self.generation_id, predictions)

    def set_best_score(self, score: float) -> None:
        self._best_score = float(score)

    def set_conclusion(self, text: str) -> None:
        """One- or two-line takeaway from the run. Flushed on clean __exit__."""
        self._conclusion = str(text)

    def update_generation(self, **patch: Any) -> None:
        assert self.generation_id
        self.client.update_generation(self.generation_id, **patch)

    def update_experiment(self, **patch: Any) -> None:
        assert self.experiment_id
        self.client.update_experiment(self.experiment_id, **patch)

    @contextmanager
    def benchmark(self, *, test_dataset_id: str, model_ref: str):
        """Convenience for the end-of-run promote-to-leaderboard call:

            with run.benchmark(test_dataset_id=td, model_ref=model) as record:
                record(score=0.83)
        """
        def _record(**kwargs: Any) -> dict:
            assert self.generation_id
            return self.client.record_benchmark(
                test_dataset_id=test_dataset_id, model_ref=model_ref,
                generation_id=self.generation_id, **kwargs,
            )
        yield _record


__all__ = ["DashboardClient", "DashboardClientError", "TrajectoryAuthError", "ExperimentRun"]
