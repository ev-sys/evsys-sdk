"""Backend — the only surface that talks to tinker.

The `Backend` Protocol is the seam the rest of the training stack is built
against. A concrete backend (:class:`~evsys_sdk.training.backend.MockBackend`
for tests; the future ``TinkerBackend`` for real runs) implements the
six-method contract and the rest of the package (loop, step builders,
checkpoint manager) stays backend-agnostic. That's how we drop the
tinker_cookbook dependency without locking ourselves to tinker either.

Method shape mirrors tinker's own client so a TinkerBackend implementation
is one-to-one plumbing — see the docstrings on each method for the
contract.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable

import tinker

LossCallable = Callable[[Any, dict[str, Any]], Any]
"""Custom client-side loss. Receives ``(model_output, batch_metadata)`` and
returns a scalar (typically a ``torch.Tensor``) the backend uses for the
backward pass. The exact shape of ``model_output`` is backend-defined —
TinkerBackend passes ``tinker.ForwardOutput`` here."""


# ---------------------------------------------------------------------------
# Result envelopes — kept thin and decoupled from tinker types so MockBackend
# tests don't need to construct full pydantic ForwardBackwardOutput objects.
# TinkerBackend converts tinker's responses into these on the way out.
# ---------------------------------------------------------------------------


class ForwardBackwardResult(Protocol):
    """What ``forward_backward_*_async`` resolves to.

    Tinker's own ``ForwardBackwardOutput`` is opaque (pydantic with no public
    attrs in the installed version), so consumers go through the loss_fn_outputs
    field on the actual response. The Protocol here lists what the loop reads
    so the contract is explicit; MockBackend constructs duck-typed objects.
    """

    loss_fn_outputs: list[dict[str, Any]]
    """One entry per Datum. Each carries algorithm-specific keys —
    ``"logprobs"`` for cross_entropy, ``"loss"`` / ``"advantage_sum"`` for IS, etc."""


class OptimStepResult(Protocol):
    metrics: dict[str, float]
    """Optimizer-side metrics emitted by tinker (``optim/lr``, gradient norms,
    etc.). The loop merges these into the per-step record."""


# ---------------------------------------------------------------------------
# SamplingClient — what an Evaluator or a StepBuilder gets handed
# ---------------------------------------------------------------------------


@runtime_checkable
class SamplingClient(Protocol):
    """Inference-only client backed by a saved sampler checkpoint.

    Created by :meth:`Backend.snapshot_sampling_client`. Both the loop's
    in-loop eval slot and any rollout-based StepBuilder (SDFT, RL) consume
    this surface — never the underlying tinker client directly.
    """

    async def sample_async(
        self,
        *,
        prompt: tinker.ModelInput,
        params: tinker.SamplingParams,
        num_samples: int = 1,
        include_prompt_logprobs: bool = False,
        topk_prompt_logprobs: int = 0,
    ) -> Any:
        """Generate tokens for ``prompt``. Return shape is backend-defined.

        TinkerBackend returns tinker's native ``SamplingResponse`` with
        ``.sequences`` (one per sample) and optional ``.topk_prompt_logprobs``.
        MockBackend returns canned strings for tests.
        """
        ...

    async def compute_logprobs_async(self, prompt: tinker.ModelInput) -> list[float | None]:
        """Per-position logprobs of the prompt under the current weights."""
        ...


# ---------------------------------------------------------------------------
# Backend — what an Algorithm composer constructs once per train() call
# ---------------------------------------------------------------------------


@runtime_checkable
class Backend(Protocol):
    """Wraps a training client + factory for sampling clients.

    A `Backend` is constructed once per run (typically by the Algorithm
    composer, see :mod:`evsys_sdk.algorithms.sft`). It owns the
    live training client; everything else in the package operates against
    this Protocol.
    """

    def forward_backward_async(
        self,
        data: list[tinker.Datum],
        *,
        loss_fn: tinker.types.LossFnType,
        loss_fn_config: dict[str, Any] | None = None,
    ) -> Any:
        """Submit a forward + backward pass with a named server-side loss.

        Returns an awaitable that resolves to a :class:`ForwardBackwardResult`.
        Tinker's loss names are ``"cross_entropy"`` (with per-position weights
        on Datum.loss_fn_inputs["weights"]) and ``"importance_sampling"``
        (with advantages on Datum.loss_fn_inputs["advantages"]).
        """
        ...

    def forward_backward_custom_async(
        self,
        data: list[tinker.Datum],
        loss_fn: LossCallable,
    ) -> Any:
        """Same as :meth:`forward_backward_async`, but the loss is a
        client-side Python callable.

        Tinker streams logits/logprobs back, the callable runs locally on
        each Datum, and the returned scalar gets sent back for the backward
        pass. This is the same hook SDFT uses for analytical reverse-KL,
        exposed here as a first-class extension point.
        """
        ...

    def optim_step_async(self, adam: tinker.AdamParams) -> Any:
        """Optimizer step. Returns an awaitable resolving to :class:`OptimStepResult`."""
        ...

    async def save_for_sampler(self, name: str) -> str:
        """Snapshot the current weights for inference. Returns the sampler URI."""
        ...

    async def save_full_state(self, name: str) -> str:
        """Snapshot weights + optimizer state for resume. Returns the URI."""
        ...

    async def snapshot_sampling_client(self, name: str | None = None) -> SamplingClient:
        """Save_weights_for_sampler + construct an inference client at that URI.

        Convenience for the in-loop eval slot, which needs both: save the
        latest weights AND get a client that can sample from them. The
        manager records the URI; the eval slot uses the client.
        """
        ...

    def get_tokenizer(self) -> Any:
        """HF tokenizer the backend is using. Used by StepBuilders that need
        to tokenize text (SFT) or apply chat templates (SDFT)."""
        ...


# ---------------------------------------------------------------------------
# MockBackend — tests use this to exercise the loop without spending real
# compute. No tinker server needed.
# ---------------------------------------------------------------------------


class _ResolvedFuture:
    """Tinker-style future with ``.result_async()`` resolving to a value."""

    def __init__(self, value: Any) -> None:
        self._value = value

    async def result_async(self) -> Any:
        return self._value

    def result(self) -> Any:
        return self._value


class _MockResult:
    """Duck-typed ForwardBackwardResult / OptimStepResult."""

    def __init__(self, **fields: Any) -> None:
        for k, v in fields.items():
            setattr(self, k, v)


class MockSamplingClient:
    """In-memory SamplingClient for tests.

    Returns ``canned[i % len(canned)]`` per call to ``sample_async``. Tests
    that need fancy behavior can subclass; the default is enough to exercise
    the loop's eval slot.
    """

    def __init__(self, *, canned: list[str] | None = None, name: str = "mock") -> None:
        self.canned = canned or [""]
        self.name = name
        self.model_path: str | None = None
        self.sample_calls: list[dict] = []
        self.logprob_calls: list[tinker.ModelInput] = []
        self._idx = 0

    async def sample_async(
        self,
        *,
        prompt: tinker.ModelInput,
        params: tinker.SamplingParams,
        num_samples: int = 1,
        include_prompt_logprobs: bool = False,
        topk_prompt_logprobs: int = 0,
    ) -> Any:
        self.sample_calls.append({
            "prompt_len": prompt.length, "params": params, "num_samples": num_samples,
            "topk_prompt_logprobs": topk_prompt_logprobs,
        })
        text = self.canned[self._idx % len(self.canned)]
        self._idx += 1
        # Mimic tinker's SamplingResponse: a .sequences list, each with .tokens.
        seq = _MockResult(tokens=list(range(len(text))) or [0])
        return _MockResult(sequences=[seq], topk_prompt_logprobs=None)

    async def compute_logprobs_async(self, prompt: tinker.ModelInput) -> list[float | None]:
        self.logprob_calls.append(prompt)
        return [0.0] * prompt.length


class MockBackend:
    """In-memory Backend for tests.

    Records every call to ``forward_backward_async`` /
    ``forward_backward_custom_async`` / ``optim_step_async`` /
    ``save_for_sampler`` / ``save_full_state`` /
    ``snapshot_sampling_client`` for assertion. Returns predictable result
    objects so the loop's metric / save / eval paths exercise end-to-end.
    """

    def __init__(
        self,
        *,
        tokenizer: Any = None,
        sampler_factory: Callable[[str], MockSamplingClient] | None = None,
    ) -> None:
        self._tokenizer = tokenizer
        self._sampler_factory = sampler_factory or (lambda name: MockSamplingClient(name=name))
        # call recorders
        self.fb_calls: list[dict] = []
        self.fb_custom_calls: list[dict] = []
        self.optim_calls: list[tinker.AdamParams] = []
        self.save_sampler_calls: list[str] = []
        self.save_state_calls: list[str] = []
        # default per-Datum loss_fn_outputs scalar (used for train_mean_nll math)
        self.fb_logprob: float = -0.5
        # default optim metrics
        self.optim_metrics: dict[str, float] = {"optim/lr": 1e-4}

    # --- training-side ------------------------------------------------------

    def forward_backward_async(
        self,
        data: list[tinker.Datum],
        *,
        loss_fn: tinker.types.LossFnType,
        loss_fn_config: dict[str, Any] | None = None,
    ) -> Any:
        self.fb_calls.append({"n_data": len(data), "loss_fn": loss_fn,
                              "loss_fn_config": loss_fn_config})
        outputs = [{"logprobs": [self.fb_logprob] * 4} for _ in data]
        return _ResolvedFuture(_MockResult(loss_fn_outputs=outputs))

    def forward_backward_custom_async(
        self,
        data: list[tinker.Datum],
        loss_fn: LossCallable,
    ) -> Any:
        self.fb_custom_calls.append({"n_data": len(data), "loss_fn": loss_fn})
        outputs = [{"logprobs": [self.fb_logprob] * 4} for _ in data]
        return _ResolvedFuture(_MockResult(loss_fn_outputs=outputs))

    def optim_step_async(self, adam: tinker.AdamParams) -> Any:
        self.optim_calls.append(adam)
        metrics = dict(self.optim_metrics)
        # Echo the LR actually applied this step (schedules change it).
        metrics["optim/lr"] = float(adam.learning_rate)
        return _ResolvedFuture(_MockResult(metrics=metrics))

    # --- save / snapshot ----------------------------------------------------

    async def save_for_sampler(self, name: str) -> str:
        self.save_sampler_calls.append(name)
        return f"mock://sampler/{name}"

    async def save_full_state(self, name: str) -> str:
        self.save_state_calls.append(name)
        return f"mock://state/{name}"

    async def snapshot_sampling_client(self, name: str | None = None) -> SamplingClient:
        label = name or f"snapshot_{len(self.save_sampler_calls)}"
        path = await self.save_for_sampler(label)
        client = self._sampler_factory(label)
        try:
            client.model_path = path  # harbor-backed evaluators re-sample from this
        except Exception:
            pass
        return client

    def get_tokenizer(self) -> Any:
        return self._tokenizer


__all__ = [
    "Backend",
    "ForwardBackwardResult",
    "LossCallable",
    "MockBackend",
    "MockSamplingClient",
    "OptimStepResult",
    "SamplingClient",
]
