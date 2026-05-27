"""ModalMiles — dispatch a training run to a deployed Modal app (miles / unsloth).

Port of the backend repo's ``modal_runner.py`` adapted to the SDK protocol.
The training loop itself runs **remotely** inside the deployed Modal app
(``modal_app.py`` → ``train_remote_miles`` / ``train_remote_unsloth`` on the
``radixark/miles`` image); this algorithm only orchestrates:

  1. stage the run's rows to a Modal volume as JSONL,
  2. pick the engine (miles vs unsloth) + recipe bash/py script,
  3. look up the deployed remote function by name and ``spawn`` it,
  4. poll for completion, ingest the returned checkpoint into the RunResult.

Because the heavy lifting is remote, this needs a deployed Modal app + Modal
credentials to actually run — it is NOT exercised by the SDK's local/mock
tests end-to-end. ``modal`` is imported lazily so the SDK imports cleanly
without it; if ``modal`` is missing the algorithm returns a failed RunResult
with a clear message rather than raising at import.
"""

from __future__ import annotations

import json
import logging
import subprocess
import time
from pathlib import Path
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field

from ..protocols import RunContext, RunResult
from ..registry import register_algorithm

logger = logging.getLogger(__name__)

# Model substrings that are too big for unsloth's single-GPU path → force miles.
_TOO_BIG_FOR_UNSLOTH = (
    "70b", "72b", "32b", "30b", "27b", "35b", "40b",
    "235b", "397b", "moe", "a3b", "a22b", "a32b",
)


class ModalMilesConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    recipe_kind: Literal["sft", "rl", "sdft"] = "sft"
    engine: Literal["auto", "miles", "unsloth"] = "auto"
    """Override the backend's engine. 'auto' defers to backend.engine, then a
    size/recipe heuristic."""
    script_name: str | None = None
    """Explicit recipe script in the image's scripts/modal/. If None, derived
    from (recipe_kind, engine)."""
    dataset_filename: str = "rendered.jsonl"
    # Recipe knobs forwarded to the remote script as env vars.
    learning_rate: float = 5e-5
    batch_size: int = 8
    num_epochs: int = 1
    num_steps: int = 100
    max_steps: int = 0
    max_seq_len: int = 2048
    lora_rank: int = 16
    group_size: int = 8
    max_output_tokens: int = 512
    temperature: float = 1.0
    loss: str = "grpo"
    kl_coef: float = 0.0
    kl_penalty_coef: float = 0.0
    num_loss_tokens_to_skip: int = 0
    ref_model_sync_steps: int = 0
    teacher_template: str = ""
    extra_env: dict[str, str] = Field(default_factory=dict)
    """Arbitrary extra env vars merged into the remote call (highest priority)."""


@register_algorithm("modal_miles")
class ModalMiles:
    name: ClassVar[str] = "modal_miles"
    Config: ClassVar[type] = ModalMilesConfig

    def __init__(self, **kwargs: Any) -> None:
        self.cfg = ModalMilesConfig.model_validate(kwargs)

    # -- engine + script selection (ports modal_runner heuristics) -----------

    def _resolve_engine(self, handles: dict[str, Any], model_name: str) -> str:
        engine = self.cfg.engine
        if engine == "auto":
            engine = handles.get("engine", "auto")
        if engine in ("miles", "unsloth"):
            return engine
        name = model_name.lower()
        if any(s in name for s in _TOO_BIG_FOR_UNSLOTH):
            return "miles"
        if self.cfg.recipe_kind == "rl":
            return "miles"
        return "unsloth"

    def _build_call(self, engine: str, model_name: str, dataset_path: str) -> tuple[str, dict[str, str]]:
        env = self._recipe_env(model_name, dataset_path)
        if self.cfg.script_name:
            return self.cfg.script_name, env
        rk = self.cfg.recipe_kind
        if engine == "unsloth":
            script = {"sft": "run_unsloth_sft.py", "sdft": "run_unsloth_sdft.py"}.get(rk)
            if script is None:
                raise ValueError(f"unsloth engine has no script for recipe_kind={rk!r}")
            return script, env
        return {"sft": "run_miles_sft.sh", "rl": "run_miles_rl.sh", "sdft": "run_miles_sdft.sh"}[rk], env

    def _recipe_env(self, model_name: str, dataset_path: str) -> dict[str, str]:
        c = self.cfg
        common = {
            "BASE_MODEL": model_name,
            "DATASET_PATH": dataset_path,
            "LEARNING_RATE": str(c.learning_rate),
            "BATCH_SIZE": str(c.batch_size),
            "MAX_SEQ_LEN": str(c.max_seq_len),
            "LORA_RANK": str(c.lora_rank),
        }
        if c.recipe_kind == "sft":
            common |= {"NUM_EPOCHS": str(c.num_epochs), "MAX_STEPS": str(c.max_steps)}
        elif c.recipe_kind == "rl":
            common |= {
                "GROUP_SIZE": str(c.group_size), "NUM_STEPS": str(c.num_steps),
                "MAX_OUTPUT_TOKENS": str(c.max_output_tokens), "TEMPERATURE": str(c.temperature),
                "LOSS": c.loss, "KL_COEF": str(c.kl_coef),
            }
        elif c.recipe_kind == "sdft":
            common |= {
                "NUM_STEPS": str(c.num_steps), "MAX_OUTPUT_TOKENS": str(c.max_output_tokens),
                "TEMPERATURE": str(c.temperature), "KL_PENALTY_COEF": str(c.kl_penalty_coef),
                "NUM_LOSS_TOKENS_TO_SKIP": str(c.num_loss_tokens_to_skip),
                "REF_MODEL_SYNC_STEPS": str(c.ref_model_sync_steps),
                "TEACHER_TEMPLATE": c.teacher_template,
            }
        common |= dict(c.extra_env)
        return common

    # -- dataset staging -----------------------------------------------------

    def _stage_dataset(self, ctx: RunContext, rows: list[dict[str, Any]], data_volume: str, run_key: str) -> str:
        out = Path(ctx.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        local_path = out / self.cfg.dataset_filename
        with local_path.open("w") as f:
            for r in rows:
                f.write(json.dumps(r, default=str) + "\n")
        volume_path = f"/{run_key}/{self.cfg.dataset_filename}"
        container_path = f"/data{volume_path}"
        _volume_put(local_path=str(local_path), volume=data_volume, remote_path=volume_path)
        logger.info("modal_miles: staged %d rows → %s on %s", len(rows), volume_path, data_volume)
        return container_path

    # -- main ----------------------------------------------------------------

    def train(self, ctx: RunContext) -> RunResult:
        if ctx.backend.name != "modal":
            raise RuntimeError(f"ModalMiles requires backend=modal (got '{ctx.backend.name}')")
        try:
            import modal
        except ImportError as e:
            return RunResult(
                run_id=ctx.run_id, status="failed",
                error=f"modal not installed: {e}. `pip install modal` and deploy modal_app.py.",
            )

        rows = ctx.extras.get("train_rows")
        if not rows:
            return RunResult(run_id=ctx.run_id, status="failed", error="ctx.extras['train_rows'] missing/empty")
        handles = ctx.extras.get("backend_handles", {})
        model_name = handles.get("model_name") or ctx.extras.get("model_name")
        if not model_name:
            return RunResult(run_id=ctx.run_id, status="failed", error="model_name not set in backend handles")

        app_name = handles.get("app_name", "trajectory-training")
        data_volume = handles.get("data_volume", "trajectory-data")
        timeout_s = float(handles.get("timeout_hours", 12.0)) * 3600.0
        run_key = ctx.run_id.replace("/", "_")

        engine = self._resolve_engine(handles, model_name)
        container_path = self._stage_dataset(ctx, list(rows), data_volume, run_key)
        script_name, env_overrides = self._build_call(engine, model_name, container_path)
        env_overrides["EXPERIMENT_ID"] = run_key
        env_overrides["GENERATION_ID"] = "0"

        fn_name = "train_remote_unsloth" if engine == "unsloth" else "train_remote_miles"
        ctx.log_store.log_hyperparams({
            "algorithm": self.name, "engine": engine, "fn": fn_name,
            "app_name": app_name, "script_name": script_name, **self.cfg.model_dump(),
        })

        try:
            train_fn = modal.Function.from_name(app_name, fn_name)
        except Exception as e:
            return RunResult(
                run_id=ctx.run_id, status="failed",
                error=f"Modal function lookup failed (app={app_name} fn={fn_name}): {e}. "
                      f"Did you `modal deploy modal_app.py`?",
            )

        started = time.time()
        try:
            call = train_fn.spawn(
                script_name=script_name, experiment_id=run_key,
                generation_id="0", env_overrides=env_overrides,
            )
            result = call.get(timeout=timeout_s)
        except Exception as e:
            return RunResult(run_id=ctx.run_id, status="failed", error=f"Modal dispatch/wait failed: {e}")
        duration = time.time() - started

        exit_code = result.get("exit_code", -1)
        final_ckpt = result.get("final_checkpoint_path")
        self._write_tails(ctx, result)

        if exit_code != 0:
            return RunResult(
                run_id=ctx.run_id, status="failed",
                error=f"Modal script exit={exit_code}. stderr_tail: {(result.get('stderr_tail') or '')[-1500:]}",
                metrics={"duration_seconds": duration},
            )

        artifacts: dict[str, str] = {}
        if final_ckpt:
            artifacts["final_checkpoint"] = final_ckpt
            ctx.log_store.log_artifact("final_checkpoint", final_ckpt, kind="checkpoint")
        return RunResult(
            run_id=ctx.run_id, status="completed",
            metrics={"duration_seconds": duration},
            artifacts=artifacts,
        )

    @staticmethod
    def _write_tails(ctx: RunContext, result: dict[str, Any]) -> None:
        try:
            out = Path(ctx.output_dir)
            (out / "modal_stdout_tail.log").write_text(result.get("stdout_tail") or "")
            (out / "modal_stderr_tail.log").write_text(result.get("stderr_tail") or "")
        except Exception as e:
            logger.warning("modal_miles: failed to write log tails: %s", e)


def _volume_put(*, local_path: str, volume: str, remote_path: str) -> None:
    """Upload a local file to a Modal volume via the modal CLI.

    Factored out so tests can stub it without a real Modal volume.
    """
    subprocess.run(
        ["modal", "volume", "put", "--force", volume, local_path, remote_path],
        check=True, capture_output=True, text=True,
    )
