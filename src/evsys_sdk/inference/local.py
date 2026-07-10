"""LocalInference — generate via a local HuggingFace model."""

from __future__ import annotations

from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict

from ..registry import register_inference

import torch  # noqa: E402


class LocalInferenceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model_name: str
    adapter_path: str | None = None
    dtype: str = "bfloat16"
    device: str = "auto"


_DTYPE_MAP = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


@register_inference("local")
class LocalInference:
    name: ClassVar[str] = "local"
    Config: ClassVar[type] = LocalInferenceConfig

    def __init__(
        self,
        *,
        model_name: str,
        adapter_path: str | None = None,
        dtype: str = "bfloat16",
        device: str = "auto",
    ) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # ChatTemplatedInference (inference/chat_templated.py) requires a
        # `_tokenizer` attribute to auto-wrap eval-time prompts with the
        # training-time chat template; alias it here so LocalInference is a
        # valid `base` for that wrapper.
        self._tokenizer = self.tokenizer
        torch_dtype = _DTYPE_MAP.get(dtype, torch.float32)
        if device == "mps":
            # MPS doesn't support device_map="auto"; load to CPU then move
            # (mirrors backends/local.py::LocalBackend.prepare).
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, dtype=torch_dtype, trust_remote_code=True,
            ).to("mps")
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                dtype=torch_dtype,
                trust_remote_code=True,
                device_map=device if device != "auto" else "auto",
            )
        if adapter_path:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter_path)
            if device == "mps":
                self.model = self.model.to("mps")
        self.model.eval()

    def generate(
        self,
        *,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.0,
        stop: list[str] | None = None,
    ) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        do_sample = temperature > 0.0
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature if do_sample else 1.0,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        decoded = self.tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        if stop:
            for s in stop:
                idx = decoded.find(s)
                if idx >= 0:
                    decoded = decoded[:idx]
        return decoded
