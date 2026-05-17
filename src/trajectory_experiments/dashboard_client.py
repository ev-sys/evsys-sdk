"""DashboardClient — push SDK runs to the Trajectory backend → Supabase.

Architecture: SDK → Django HTTP → Supabase. The user holds an API key (the
same key the dashboard issues at `Settings → API keys`), the SDK calls
``/api/dashboard/api/sdk/...`` routes, and the backend authenticates the key,
maps it to a user_id, and writes to Supabase using its service key. All RLS
is handled centrally on the backend; the SDK never needs Supabase creds.

Quick usage::

    from trajectory_experiments import DashboardClient

    client = DashboardClient(base_url="https://api.trajectory.ai", api_key="sk_...")

    exp = client.create_experiment(
        experiment_name="composio_sft_v9",
        client="composio",
        hypothesis="LoRA r=8 → r=32 raises pass@1",
        hypothesis_reasoning=(
            "Prior runs plateaued at 0.71 with r=8 and train-loss was still "
            "dropping; higher rank should give the adapter more capacity."
        ),
        tags=["axis:lora"],
    )
    gen = client.create_generation(
        experiment_id=exp["id"],
        recipe_kind="sft",
        run_config={"lr": 1e-5, "batch_size": 16, "epochs": 2},
    )
    client.log_step_metric(gen["id"], step=100, loss=0.42)
    client.log_eval_run(gen["id"], step=500, eval_name="composio_eval_v3",
                        metrics={"pass_at_1": 0.83})
    client.log_predictions(gen["id"], [
        {"task_id": "t1", "instruction": "...", "model_output": "...",
         "expected": "...", "reward": 1.0, "kind": "eval"},
    ])
    client.update_generation(gen["id"], status="completed", best_score=0.83)
    client.update_experiment(
        exp["id"],
        status="completed", best_score=0.83, best_generation_id=gen["id"],
        conclusion="r=32 raised pass@1 to 0.83 with no instability; promote.",
    )

Or use the context manager for the common case::

    with ExperimentRun(client,
                       experiment_name="composio_sft_v9",
                       client_name="composio",
                       hypothesis="LoRA r=8 → r=32 raises pass@1",
                       recipe_kind="sft",
                       run_config={"lr": 1e-5, "batch_size": 16, "epochs": 2},
    ) as run:
        for step in range(1, 1001):
            run.log_step(step, loss=...)
        run.log_eval(step=1000, eval_name="composio_eval_v3",
                     metrics={"pass_at_1": 0.83, "pass_at_3": 0.94})
        run.log_predictions([...])
        run.set_best_score(0.83)
    # auto-marks generation + experiment as completed on clean exit;
    # marks failed (with the exception message) on raise.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any

import requests

DEFAULT_TIMEOUT_S = 30.0


class DashboardClientError(RuntimeError):
    """Raised when a dashboard write fails. Carries status + response body."""

    def __init__(self, status: int, body: str, path: str) -> None:
        super().__init__(f"{path} → HTTP {status}: {body[:200]}")
        self.status = status
        self.body = body
        self.path = path


class DashboardClient:
    """Thin HTTP client for the Trajectory backend's SDK write routes.

    Authentication: Bearer token (the API key from the dashboard settings).
    Errors: raises DashboardClientError on any non-2xx response.
    """

    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout_s: float = DEFAULT_TIMEOUT_S,
    ) -> None:
        self.base_url = (base_url or os.environ.get("TRAJECTORY_API_URL") or "http://localhost:8000").rstrip("/")
        key = api_key or os.environ.get("TRAJECTORY_API_KEY")
        if not key:
            raise RuntimeError(
                "DashboardClient needs an api_key (or TRAJECTORY_API_KEY env var). "
                "Get one from the dashboard at Settings → API keys."
            )
        self.api_key = key
        self.timeout_s = timeout_s
        self._session = requests.Session()
        self._session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type":  "application/json",
        })

    # -- low-level ---------------------------------------------------------

    def _post(self, path: str, body: dict | None = None) -> dict:
        url = f"{self.base_url}/api/dashboard/api{path}"
        r = self._session.post(url, json=body or {}, timeout=self.timeout_s)
        if not 200 <= r.status_code < 300:
            raise DashboardClientError(r.status_code, r.text, path)
        try:
            return r.json()
        except Exception:
            return {"_raw": r.text}

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
        return self._post("/sdk/experiments/", body)["experiment"]

    def update_experiment(self, experiment_id: str, **patch: Any) -> dict:
        """PATCH experiment metadata. Whitelisted fields on the backend:
        status, best_score, best_generation_id, current_iteration,
        error_message, hypothesis, hypothesis_reasoning, plan, conclusion,
        tags, problem_statement_id.
        """
        return self._post(f"/sdk/experiments/{experiment_id}/", patch)

    # -- generations -------------------------------------------------------

    def create_generation(
        self,
        *,
        experiment_id: str,
        recipe_kind: str | None = None,
        run_config: dict | None = None,
        iteration: int | None = None,
        parent_id: str | None = None,
        status: str = "pending",
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
        return self._post("/sdk/generations/", body)["generation"]

    def update_generation(self, generation_id: str, **patch: Any) -> dict:
        return self._post(f"/sdk/generations/{generation_id}/", patch)

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
            if v is not None: body[k] = v
        return self._post(f"/sdk/generations/{generation_id}/step/", body)

    def log_eval_run(
        self,
        generation_id: str,
        *,
        eval_name: str,
        metrics: dict[str, float],
        step: int | None = None,
    ) -> dict:
        body: dict = {"eval_name": eval_name, "metrics": dict(metrics)}
        if step is not None: body["step"] = int(step)
        return self._post(f"/sdk/generations/{generation_id}/eval/", body)

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
        return self._post(
            f"/sdk/generations/{generation_id}/predictions/",
            {"predictions": predictions},
        )

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
        """Push an EXPLICIT scoring event to the leaderboard.

        Either provide ``score``, or ``n_passed`` + ``n_total`` (backend
        derives score = n_passed / n_total).
        """
        body: dict = {"test_dataset_id": test_dataset_id, "model_ref": model_ref}
        for k, v in (
            ("score", score), ("n_passed", n_passed), ("n_total", n_total),
            ("generation_id", generation_id), ("step", step),
        ):
            if v is not None: body[k] = v
        return self._post("/benchmark-runs/", body)


# ---------------------------------------------------------------------------
# Ergonomic context-managed run
# ---------------------------------------------------------------------------


class ExperimentRun:
    """One generation in one experiment, lifecycle-managed.

    Use as a context manager. On clean exit, the generation + experiment are
    marked completed (with the best score / generation_id). On raised
    exception, both are marked failed with the exception message.

    All log_* methods are passthroughs to the underlying DashboardClient.
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
            status="running",
            **{k: v for k, v in self._gen_kwargs.items() if v is not None},  # type: ignore[arg-type]
        )
        self.generation_id = gen["id"]
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if exc_type is not None:
            msg = f"{exc_type.__name__}: {exc_val}" if exc_val else exc_type.__name__
            if self.generation_id:
                try: self.client.update_generation(self.generation_id, status="failed", error_message=msg)
                except Exception: pass
            if self.experiment_id:
                try: self.client.update_experiment(self.experiment_id, status="failed", error_message=msg)
                except Exception: pass
            return None
        # Clean exit. training_generations has no best_score column —
        # combined_score is the headline metric there. training_experiments
        # is where best_score lives.
        if self.generation_id:
            gen_patch: dict = {"status": "completed"}
            if self._best_score is not None: gen_patch["combined_score"] = self._best_score
            self.client.update_generation(self.generation_id, **gen_patch)
        if self.experiment_id:
            exp_patch: dict = {"status": "completed"}
            if self._best_score is not None: exp_patch["best_score"] = self._best_score
            if self.generation_id: exp_patch["best_generation_id"] = self.generation_id
            if self._conclusion is not None: exp_patch["conclusion"] = self._conclusion
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
        """One- or two-line takeaway from the run. Flushed on clean __exit__.

        Example: "LoRA r=32 raised pass@1 from 0.71 → 0.83 with no train-loss
        instability — hypothesis confirmed; promote to leaderboard."
        """
        self._conclusion = str(text)

    def update_generation(self, **patch: Any) -> None:
        assert self.generation_id
        self.client.update_generation(self.generation_id, **patch)

    def update_experiment(self, **patch: Any) -> None:
        assert self.experiment_id
        self.client.update_experiment(self.experiment_id, **patch)

    @contextmanager
    def benchmark(self, *, test_dataset_id: str, model_ref: str):
        """Convenience for the common end-of-run promote-to-leaderboard call:

            with run.benchmark(test_dataset_id=td, model_ref=model) as record:
                record(score=0.83)            # → benchmark_runs row
        """
        results: list[dict] = []
        def _record(**kwargs: Any) -> dict:
            assert self.generation_id
            r = self.client.record_benchmark(
                test_dataset_id=test_dataset_id, model_ref=model_ref,
                generation_id=self.generation_id, **kwargs,
            )
            results.append(r)
            return r
        yield _record


__all__ = ["DashboardClient", "DashboardClientError", "ExperimentRun"]
