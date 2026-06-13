"""TinkerRL — GRPO-style RL via tinker_cookbook.

Builds an RLDataset where each prompt is a single-step environment whose reward
comes from a registered Verifier (see the verifiers registry).
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field

from ..protocols import RunContext, RunResult
from ..registry import get_verifier, register_algorithm

import chz  # noqa: E402
import tinker  # noqa: E402
from tinker_cookbook.completers import StopCondition  # noqa: E402
from tinker_cookbook.rl import train as rl_train  # noqa: E402
from tinker_cookbook.rl.types import (  # noqa: E402
    Env,
    EnvGroupBuilder,
    RLDataset,
    RLDatasetBuilder,
    StepResult,
)
from tinker_cookbook.tokenizer_utils import get_tokenizer  # noqa: E402

# Module-level cache of {cache_key: {rows, verifier, tokenizer, ...}} so the
# chz-frozen builder can fetch its mutable state.
_RL_CACHE: dict[str, dict[str, Any]] = {}

logger = logging.getLogger(__name__)


class TinkerRLConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    learning_rate: float = 5e-5
    num_steps: int = 100
    """Training iterations (groups per iteration = batch via dataset builder)."""
    groups_per_batch: int = 32
    """How many prompts (groups) per training iteration."""
    num_generations: int = 4
    """Group size for GRPO."""
    max_tokens: int = 256
    kl_penalty_coef: float = 0.04
    save_every: int = 50
    eval_every: int = 0
    lora_rank: int = 8
    verifier_kind: str = "format_only"
    verifier_params: dict[str, Any] = Field(default_factory=dict)
    stop_strings: list[str] = Field(default_factory=lambda: ["</answer>"])
    wandb_project: str | None = None
    wandb_name: str | None = None
    renderer_name: str | None = None
    kl_reference_checkpoint_path: str | None = None
    """Sampler-weights checkpoint URI for the KL reference model. If None
    and kl_penalty_coef>0, falls back to the policy's load_checkpoint_path,
    converting `weights/...` to `sampler_weights/...` on the fly. If that
    can't be derived either, the base model is used (which can collapse
    SFT progress — see RESULTS.md / 'PG vs KL' analysis)."""


# ---------- Single-step Env over a chat-rendered prompt ----------


class _SingleTurnEnv(Env):
    """One-shot environment: agent emits a completion, gets a verifier reward."""

    def __init__(self, *, prompt_input: tinker.ModelInput, target: dict[str, Any], verifier, tokenizer, stop_strings: list[str]) -> None:
        self._prompt = prompt_input
        self._target = target
        self._verifier = verifier
        self._tokenizer = tokenizer
        self._stop = stop_strings

    async def initial_observation(self) -> tuple[tinker.ModelInput, list[str]]:
        return self._prompt, list(self._stop)

    async def step(self, action, *, extra=None) -> StepResult:
        # action is list[int] (token ids); decode and verify.
        completion_text = self._tokenizer.decode(action, skip_special_tokens=True)
        result = self._verifier.verify(
            prompt="",
            completion=completion_text,
            target=self._target,
        )
        return StepResult(
            reward=float(result.reward),
            episode_done=True,
            next_observation=self._prompt,  # unused — episode_done=True
            next_stop_condition=list(self._stop),
            metrics={"reward": float(result.reward)},
            logs={"completion_preview": completion_text[:200]},
        )


class _GroupBuilder(EnvGroupBuilder):
    """Reuse the same env spec for each member of a group."""

    def __init__(self, *, prompt_input, target, verifier, tokenizer, stop_strings, group_size) -> None:
        self._prompt = prompt_input
        self._target = target
        self._verifier = verifier
        self._tokenizer = tokenizer
        self._stop = stop_strings
        self._group_size = group_size

    async def make_envs(self):
        return [
            _SingleTurnEnv(
                prompt_input=self._prompt,
                target=self._target,
                verifier=self._verifier,
                tokenizer=self._tokenizer,
                stop_strings=self._stop,
            )
            for _ in range(self._group_size)
        ]


class _RowsRLDataset(RLDataset):
    def __init__(
        self,
        cache_key: str,
        group_size: int,
        groups_per_batch: int,
        stop_strings: list[str],
        num_iterations: int,
    ) -> None:
        self._cache_key = cache_key
        self._group_size = group_size
        self._groups_per_batch = groups_per_batch
        self._stop_strings = stop_strings
        self._num_iterations = num_iterations

    def _state(self) -> dict[str, Any]:
        return _RL_CACHE[self._cache_key]

    def num_iterations(self) -> int:
        return self._num_iterations

    def __len__(self) -> int:
        return self._num_iterations

    def get_batch(self, iteration: int) -> list[EnvGroupBuilder]:
        state = self._state()
        rows = state["rows"]
        tokenizer = state["tokenizer"]
        start = (iteration * self._groups_per_batch) % len(rows)
        builders = []
        for i in range(self._groups_per_batch):
            row = rows[(start + i) % len(rows)]
            prompt_text = row.get("prompt") or row["messages"][-1]["content"]
            ids = tokenizer.encode(prompt_text)
            prompt_input = tinker.ModelInput.from_ints(ids)
            target = {"tool_slug": row.get("tool_slug", ""), "toolkit": row.get("toolkit", "")}
            builders.append(
                _GroupBuilder(
                    prompt_input=prompt_input,
                    target=target,
                    verifier=state["verifier"],
                    tokenizer=tokenizer,
                    stop_strings=self._stop_strings,
                    group_size=self._group_size,
                )
            )
        return builders


@chz.chz
class _RowsRLDatasetBuilder(RLDatasetBuilder):
    """chz-friendly RL dataset builder. Mutable state in _RL_CACHE."""

    cache_key: str
    group_size: int
    groups_per_batch: int
    stop_strings: list[str]
    num_iterations: int

    async def __call__(self):
        ds = _RowsRLDataset(
            cache_key=self.cache_key,
            group_size=self.group_size,
            groups_per_batch=self.groups_per_batch,
            stop_strings=self.stop_strings,
            num_iterations=self.num_iterations,
        )
        return ds, None


_DEPRECATION_MSG = (
    "TinkerRL (algorithm.kind: tinker_rl) delegates to tinker_cookbook "
    "and is deprecated in favor of `native_rl`, which runs the loop "
    "natively in the SDK with the new EnvGroupBuilder Protocol "
    "(single-turn out of the box; multi-turn extends via the same "
    "Protocol). Flip algorithm.kind to 'native_rl' when ready."
)


@register_algorithm("tinker_rl")
class TinkerRL:
    name: ClassVar[str] = "tinker_rl"
    Config: ClassVar[type] = TinkerRLConfig

    def __init__(self, **kwargs) -> None:
        import warnings
        warnings.warn(_DEPRECATION_MSG, DeprecationWarning, stacklevel=2)
        self.cfg = TinkerRLConfig.model_validate(kwargs)

    def train(self, ctx: RunContext) -> RunResult:
        if ctx.backend.name != "tinker":
            raise RuntimeError(f"TinkerRL requires backend=tinker (got '{ctx.backend.name}')")
        rows = ctx.extras.get("train_rows")
        if not rows:
            raise RuntimeError("TinkerRL.train: ctx.extras['train_rows'] missing/empty")

        handles = ctx.extras.get("backend_handles", {})
        model_name = handles.get("model_name") or ctx.extras.get("model_name")
        if not model_name:
            raise RuntimeError("model_name not set in backend handles")

        tokenizer = get_tokenizer(model_name)
        verifier_cls = get_verifier(self.cfg.verifier_kind)
        verifier = verifier_cls(**self.cfg.verifier_params)

        out = Path(ctx.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        log_path = str(out)

        ctx.log_store.log_hyperparams(
            {
                "algorithm": self.name,
                **self.cfg.model_dump(),
                "model_name": model_name,
                "n_train_rows": len(rows),
            }
        )

        renderer = self.cfg.renderer_name or handles.get("renderer_name")

        import uuid as _uuid
        cache_key = f"rl_{ctx.run_id}_{_uuid.uuid4().hex}"
        _RL_CACHE[cache_key] = {
            "rows": list(rows),
            "tokenizer": tokenizer,
            "verifier": verifier,
        }
        builder = _RowsRLDatasetBuilder(
            cache_key=cache_key,
            group_size=self.cfg.num_generations,
            groups_per_batch=self.cfg.groups_per_batch,
            stop_strings=list(self.cfg.stop_strings),
            num_iterations=self.cfg.num_steps,
        )

        kl_ref = None
        if self.cfg.kl_penalty_coef > 0:
            ref_ckpt = self.cfg.kl_reference_checkpoint_path
            if ref_ckpt is None:
                # Try to derive sampler_weights path from the policy's load path.
                policy_ckpt = handles.get("load_checkpoint_path")
                if policy_ckpt and "/weights/" in policy_ckpt:
                    ref_ckpt = policy_ckpt.replace("/weights/", "/sampler_weights/")
                else:
                    ref_ckpt = policy_ckpt
            kl_ref = rl_train.KLReferenceConfig(
                base_model=model_name,
                load_checkpoint_path=ref_ckpt,
            )

        # In-loop validation (harbor set scored with metrics.py every N steps);
        # overrides the plain cfg.eval_every passthrough when configured.
        from .validation_evaluator import build_validation_evaluator_builders
        evaluator_builders, val_eval_every = build_validation_evaluator_builders(ctx, tokenizer)
        eval_every = val_eval_every if val_eval_every is not None else self.cfg.eval_every

        config = rl_train.Config(
            learning_rate=self.cfg.learning_rate,
            dataset_builder=builder,
            model_name=model_name,
            max_tokens=self.cfg.max_tokens,
            log_path=log_path,
            eval_every=eval_every,
            evaluator_builders=evaluator_builders,
            save_every=self.cfg.save_every,
            load_checkpoint_path=handles.get("load_checkpoint_path"),
            renderer_name=renderer,
            wandb_project=self.cfg.wandb_project,
            wandb_name=self.cfg.wandb_name,
            kl_penalty_coef=self.cfg.kl_penalty_coef,
            kl_reference_config=kl_ref,
            lora_rank=self.cfg.lora_rank,
        )
        try:
            asyncio.run(rl_train.main(config))
        except Exception as e:
            logger.exception("TinkerRL.train failed")
            return RunResult(run_id=ctx.run_id, status="failed", error=str(e))

        artifacts: dict[str, str] = {"run_dir": str(out)}
        ckpt_manifest = out / "checkpoints.jsonl"
        if ckpt_manifest.exists():
            import json as _json
            for line in ckpt_manifest.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = _json.loads(line)
                except Exception:
                    continue
                step_label = str(entry.get("name", "?"))
                state_path = entry.get("state_path") or entry.get("sampler_path")
                if not state_path:
                    continue
                artifacts[f"checkpoint-{step_label}"] = state_path
        for k, v in artifacts.items():
            ctx.log_store.log_artifact(k, v, kind="checkpoint")
        return RunResult(
            run_id=ctx.run_id,
            status="completed",
            metrics={},
            artifacts=artifacts,
        )
