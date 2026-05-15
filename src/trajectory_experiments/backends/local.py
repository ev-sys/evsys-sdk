"""LocalBackend — TRL-based local training (transformers + peft + trl).

prepare() loads tokenizer + model + (optional) LoRA. Heavy lifting (the
TRL trainer loop) is in algorithms/local_*.py.
"""

from __future__ import annotations

from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict

from ..registry import register_backend

# raise ImportError if torch isn't available
import torch  # noqa: E402


class LocalBackendConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dtype: str = "bfloat16"
    """One of: 'bfloat16', 'float16', 'float32'."""
    device: str = "auto"
    """'auto', 'cpu', 'cuda', or specific device id."""
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
        dtype: str = "bfloat16",
        device: str = "auto",
        trust_remote_code: bool = True,
    ) -> None:
        self.dtype = dtype
        self.device = device
        self.trust_remote_code = trust_remote_code

    def prepare(self, *, model: dict[str, Any], run_dir: str) -> dict[str, Any]:
        # Lazy imports to keep cold-start fast.
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch_dtype = _DTYPE_MAP.get(self.dtype, torch.float32)

        tokenizer = AutoTokenizer.from_pretrained(
            model["name"], trust_remote_code=self.trust_remote_code
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        device_map = self.device if self.device != "auto" else "auto"
        hf_model = AutoModelForCausalLM.from_pretrained(
            model["name"],
            dtype=torch_dtype,
            trust_remote_code=self.trust_remote_code,
            device_map=device_map,
        )

        return {
            "backend": "local",
            "model": hf_model,
            "tokenizer": tokenizer,
            "model_name": model["name"],
            "run_dir": run_dir,
        }

    def teardown(self, handles: dict[str, Any]) -> None:
        # Drop refs so GC can free GPU memory.
        handles.pop("model", None)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
