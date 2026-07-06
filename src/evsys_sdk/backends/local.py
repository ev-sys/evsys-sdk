"""LocalBackend — TRL-based local training (transformers + peft + trl).

prepare() loads tokenizer + model + (optional) LoRA. Heavy lifting (the
TRL trainer loop) is in algorithms/local_*.py.
"""

from __future__ import annotations

from typing import Any, ClassVar

# raise ImportError if torch isn't available
import torch
from pydantic import BaseModel, ConfigDict

from ..registry import register_backend


class LocalBackendConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dtype: str = "float32"
    """One of: 'bfloat16', 'float16', 'float32'. Defaults to float32 for CPU/MPS compatibility."""
    device: str = "auto"
    """'auto' (detect CUDA→MPS→CPU), 'cpu', 'cuda', or 'mps'."""
    trust_remote_code: bool = True


_DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


@register_backend("local")
class LocalBackend:
    name: ClassVar[str] = "local"
    Config: ClassVar[type] = LocalBackendConfig

    def __init__(
        self,
        *,
        dtype: str = "float32",
        device: str = "auto",
        trust_remote_code: bool = True,
    ) -> None:
        self.dtype = dtype
        self.device = device
        self.trust_remote_code = trust_remote_code

    def _resolve_device(self) -> str:
        if self.device != "auto":
            return self.device
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def prepare(self, *, model: dict[str, Any], run_dir: str) -> dict[str, Any]:
        # Lazy imports to keep cold-start fast.
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch_dtype = _DTYPE_MAP.get(self.dtype, torch.float32)
        device = self._resolve_device()

        tokenizer = AutoTokenizer.from_pretrained(
            model["name"], trust_remote_code=self.trust_remote_code
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # MPS does not support device_map="auto"; load to CPU then move.
        if device == "mps":
            hf_model = AutoModelForCausalLM.from_pretrained(
                model["name"],
                torch_dtype=torch_dtype,
                trust_remote_code=self.trust_remote_code,
            ).to("mps")
        else:
            hf_model = AutoModelForCausalLM.from_pretrained(
                model["name"],
                torch_dtype=torch_dtype,
                trust_remote_code=self.trust_remote_code,
                device_map=device,
            )

        return {
            "backend": "local",
            "model": hf_model,
            "tokenizer": tokenizer,
            "model_name": model["name"],
            "device": device,
            "run_dir": run_dir,
        }

    def teardown(self, handles: dict[str, Any]) -> None:
        handles.pop("model", None)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()
