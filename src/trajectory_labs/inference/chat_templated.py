"""ChatTemplatedInference — wrap an InferenceClient so eval-time inputs
match the chat-templated distribution the model was trained on.

When a model is SFT'd on chat-templated sequences (HF
``tokenizer.apply_chat_template`` over a system + user + assistant message
list), the assistant-side output format only emerges reliably when the
prompt at inference time is shaped the same way. ``Benchmark.score()`` hands
the client a raw ``task.instruction`` string with no role markers; this
wrapper rebuilds the (system + user) opener around it before forwarding to
the underlying client.

Project-specific values: the ``system_prompt`` (verbatim training-time
string) and ``user_template`` (e.g. ``"Query: {prompt}"`` if user-turn
content has a domain-specific prefix). With ``user_template`` left at the
default ``"{prompt}"`` the raw instruction passes through unchanged.

Tokenizer-agnostic: any base client exposing a HF-style ``_tokenizer``
attribute works (e.g. ``TinkerInference``).
"""

from __future__ import annotations

from typing import Any, ClassVar


class ChatTemplatedInference:
    """Wraps an InferenceClient to apply a chat template before each ``generate`` call."""

    name: ClassVar[str] = "chat_templated"

    def __init__(
        self,
        base: Any,
        *,
        system_prompt: str,
        user_template: str = "{prompt}",
        enable_thinking: bool | None = None,
    ) -> None:
        if not hasattr(base, "_tokenizer"):
            raise TypeError(
                "base inference client must expose a `_tokenizer` attribute "
                "(HF tokenizer with apply_chat_template); "
                f"got {type(base).__name__}"
            )
        self._base = base
        self.system_prompt = system_prompt
        self.user_template = user_template
        # Forwarded to ``apply_chat_template`` only when set, so non-Qwen
        # tokenizers (which don't accept the kwarg) keep working unchanged.
        self.enable_thinking = enable_thinking

    def generate(
        self,
        *,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.0,
        stop: list[str] | None = None,
    ) -> str:
        user_content = self.user_template.format(prompt=prompt)
        template_kwargs: dict[str, Any] = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        if self.enable_thinking is not None:
            template_kwargs["enable_thinking"] = self.enable_thinking
        templated = self._base._tokenizer.apply_chat_template(
            [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_content},
            ],
            **template_kwargs,
        )
        return self._base.generate(
            prompt=templated,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop,
        )


__all__ = ["ChatTemplatedInference"]
