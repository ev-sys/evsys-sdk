"""TinkerBackend — concrete :class:`~evsys_sdk.training.backend.Backend`
implementation against ``tinker.ServiceClient``.

Constructed via ``await TinkerBackend.create(model_name=..., lora_rank=...)``
— the underlying ``create_lora_training_client_async`` is async, so the
factory is async. Algorithm composers run the loop inside one
``asyncio.run(...)`` call so the constructor and the loop body share the
same event loop:

    async def _train(self, ctx):
        backend = await TinkerBackend.create(...)
        loop = TrainingLoop(backend=backend, ...)
        return await loop.run(num_steps=cfg.max_steps)

    def train(self, ctx):
        return asyncio.run(self._train(ctx))

The wrapper is thin — every method is a one-to-one passthrough to the
underlying tinker client, plus the one ergonomic affordance of returning
a wrapped :class:`TinkerSamplingClient` from
``snapshot_sampling_client`` (so callers don't have to know about the
sampler URI machinery).
"""

from __future__ import annotations

import logging
import os
from typing import Any

import tinker
from tinker_cookbook.tokenizer_utils import get_tokenizer  # type: ignore[import-untyped]

from .backend import LossCallable

logger = logging.getLogger(__name__)


class _CoroFuture:
    """Bridge tinker's coroutine-returning ``*_async`` methods to the
    :class:`~evsys_sdk.training.loop.TrainingLoop` contract.

    The loop fires forward_backward / optim WITHOUT awaiting, then awaits each
    one's ``result_async()`` (the shape :class:`MockBackend` implements). But
    tinker's ``forward_backward_async`` / ``optim_step_async`` are *coroutine
    functions* — calling them returns an un-awaited coroutine, not a future.
    This wraps that coroutine so ``await wrapper.result_async()`` awaits the
    coroutine to obtain the tinker future, then awaits the future's result.
    """

    __slots__ = ("_coro",)

    def __init__(self, coro: Any) -> None:
        self._coro = coro

    async def result_async(self) -> Any:
        future = await self._coro
        return await future.result_async()


class TinkerSamplingClient:
    """Wrap a ``tinker.SamplingClient`` to satisfy the
    :class:`~evsys_sdk.training.backend.SamplingClient` Protocol shape.

    Same name + same async method signatures as the Protocol, plus a
    ``raw`` attribute for callers that want the underlying tinker client.
    """

    def __init__(self, raw: Any, *, name: str = "tinker") -> None:
        self.raw = raw
        self.name = name
        self.model_path: str | None = None

    async def sample_async(
        self,
        *,
        prompt: tinker.ModelInput,
        params: tinker.SamplingParams,
        num_samples: int = 1,
        include_prompt_logprobs: bool = False,
        topk_prompt_logprobs: int = 0,
    ) -> Any:
        kwargs: dict[str, Any] = {
            "prompt": prompt,
            "sampling_params": params,
            "num_samples": num_samples,
        }
        # Tinker's sample_async forwards these only when set; the cookbook does
        # the same conditional inclusion (see distillation/sdft.py:361-369).
        if include_prompt_logprobs:
            kwargs["include_prompt_logprobs"] = True
        if topk_prompt_logprobs > 0:
            kwargs["topk_prompt_logprobs"] = topk_prompt_logprobs
        return await self.raw.sample_async(**kwargs)

    async def compute_logprobs_async(
        self, prompt: tinker.ModelInput
    ) -> list[float | None]:
        return await self.raw.compute_logprobs_async(prompt)


class TinkerBackend:
    """Implements :class:`~evsys_sdk.training.backend.Backend` over tinker.

    Construct via the :meth:`create` async factory. Sync method calls
    (``forward_backward_async``, ``optim_step_async``,
    ``forward_backward_custom_async``) return the underlying tinker
    ``APIFuture`` directly so the loop's existing ``.result_async()``
    awaits work unchanged.
    """

    def __init__(
        self,
        *,
        service_client: Any,
        training_client: Any,
        model_name: str,
        tokenizer: Any,
    ) -> None:
        self._service = service_client
        self._training = training_client
        self._model_name = model_name
        self._tokenizer = tokenizer
        self._save_counter = 0

    @classmethod
    async def create(
        cls,
        *,
        model_name: str,
        lora_rank: int = 32,
        renderer_name: str | None = None,
        resume_state_path: str | None = None,
        api_key_env: str = "TINKER_API_KEY",
        user_metadata: dict[str, str] | None = None,
    ) -> "TinkerBackend":
        """Async factory.

        ``resume_state_path``: when provided, the training client is created
        from that prior ``state_path`` (preserves optimizer state). When
        ``None``, a fresh LoRA training client is allocated at
        ``model_name`` with ``lora_rank``.

        ``renderer_name`` is recorded onto the underlying training client's
        user metadata so the inference path can read it back from a
        checkpoint manifest (same convention tinker_cookbook used).
        """
        if not os.environ.get(api_key_env):
            raise RuntimeError(f"{api_key_env} not set in environment")
        service = tinker.ServiceClient()
        meta = dict(user_metadata or {})
        if renderer_name:
            meta["renderer_name"] = renderer_name

        if resume_state_path:
            training = await service.create_training_client_from_state_with_optimizer_async(
                resume_state_path, user_metadata=meta or None,
            )
            logger.info("TinkerBackend: resumed from %s", resume_state_path)
        else:
            training = await service.create_lora_training_client_async(
                model_name, rank=lora_rank, user_metadata=meta or None,
            )
            logger.info("TinkerBackend: created LoRA client model=%s rank=%d",
                        model_name, lora_rank)

        tokenizer = get_tokenizer(model_name)
        return cls(
            service_client=service,
            training_client=training,
            model_name=model_name,
            tokenizer=tokenizer,
        )

    # --- training-side ------------------------------------------------------

    def forward_backward_async(
        self,
        data: list[tinker.Datum],
        *,
        loss_fn: tinker.types.LossFnType,
        loss_fn_config: dict[str, Any] | None = None,
    ) -> Any:
        kwargs: dict[str, Any] = {"data": data, "loss_fn": loss_fn}
        if loss_fn_config:
            kwargs["loss_fn_config"] = loss_fn_config
        return _CoroFuture(self._training.forward_backward_async(**kwargs))

    def forward_backward_custom_async(
        self,
        data: list[tinker.Datum],
        loss_fn: LossCallable,
    ) -> Any:
        return _CoroFuture(self._training.forward_backward_custom_async(
            data=data, loss_fn=loss_fn,
        ))

    def optim_step_async(self, adam: tinker.AdamParams) -> Any:
        return _CoroFuture(self._training.optim_step_async(adam))

    # --- save / snapshot ----------------------------------------------------

    async def save_for_sampler(self, name: str) -> str:
        future = await self._training.save_weights_for_sampler_async(name)
        result = await future.result_async()
        return _result_path(result)

    async def save_full_state(self, name: str) -> str:
        future = await self._training.save_state_async(name)
        result = await future.result_async()
        return _result_path(result)

    async def snapshot_sampling_client(
        self, name: str | None = None
    ) -> TinkerSamplingClient:
        label = name or self._next_snapshot_label()
        sampler_path = await self.save_for_sampler(label)
        raw = self._service.create_sampling_client(
            base_model=self._model_name, model_path=sampler_path,
        )
        client = TinkerSamplingClient(raw, name=label)
        client.model_path = sampler_path  # harbor-backed evaluators re-sample from this
        return client

    def get_tokenizer(self) -> Any:
        return self._tokenizer

    # --- internals ----------------------------------------------------------

    def _next_snapshot_label(self) -> str:
        self._save_counter += 1
        return f"snap_{self._save_counter}"


def _result_path(result: Any) -> str:
    """Tinker's save_*_async result envelopes carry a ``.path`` attribute
    (verified against cookbook usage in distillation/sdft.py:739, where the
    same dotted access ships)."""
    path = getattr(result, "path", None)
    if not path:
        raise RuntimeError(
            f"tinker save result missing .path attribute (got {result!r})"
        )
    return str(path)


__all__ = ["TinkerBackend", "TinkerSamplingClient"]
