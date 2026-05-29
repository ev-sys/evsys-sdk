#!/usr/bin/env bash
# Patch SkyRL's tinker server to plumb `topk_prompt_logprobs` end-to-end so
# distillation recipes (tinker_cookbook.distillation.sdft) work against it.
#
# vLLM's /v1/completions already supports prompt_logprobs=K (vLLM extension —
# returns top-K logprobs at each prompt position). The gap is purely in
# SkyRL's request->engine->forwarder plumbing, which currently:
#   - accepts topk_prompt_logprobs in the request schema (api.py:585)
#   - drops it when constructing SampleInput (api.py:~1089)
#   - never sends prompt_logprobs=K to vLLM
#   - has no field on SampleOutput to carry the result back
#
# This patch adds 4 lines (1 each in types.py SampleInput + SampleOutput, 1
# in api.py, ~15 in skyrl_train_inference_forwarding.py for the vLLM
# payload + response parse).
#
# Usage:
#   ./examples/modal/patch_skyrl_topk.sh /root/SkyRL
#
# Idempotent: re-running on already-patched source is a no-op (grep guards).

set -euo pipefail

ROOT="${1:?usage: patch_skyrl_topk.sh <SkyRL repo root>}"

TYPES="$ROOT/skyrl/tinker/types.py"
API="$ROOT/skyrl/tinker/api.py"
FWD="$ROOT/skyrl/tinker/extra/skyrl_train_inference_forwarding.py"

for f in "$TYPES" "$API" "$FWD"; do
    test -f "$f" || { echo "ERR: $f not found"; exit 1; }
done

# ----------------------------------------------------------------------------
# 1) types.py — add topk_prompt_logprobs to SampleInput.
# ----------------------------------------------------------------------------
if ! grep -q "topk_prompt_logprobs: int = 0" "$TYPES"; then
    python3 - "$TYPES" <<'PY'
import re, sys
p = sys.argv[1]
src = open(p).read()
# Insert after `prompt_logprobs: bool` inside class SampleInput.
old = 'class SampleInput(BaseModel):\n    base_model: str | None = None\n    prompt: ModelInput\n    sampling_params: SamplingParams\n    num_samples: int\n    checkpoint_id: str\n    prompt_logprobs: bool\n'
new = old + '    topk_prompt_logprobs: int = 0\n'
assert old in src, "SampleInput preamble not found verbatim — refusing to patch."
src = src.replace(old, new, 1)
open(p, "w").write(src)
print("patched SampleInput")
PY
fi

# ----------------------------------------------------------------------------
# 2) types.py — add topk_prompt_logprobs field to SampleOutput.
# ----------------------------------------------------------------------------
if ! grep -q "topk_prompt_logprobs: list\[list\[tuple\[int, float\]\] | None\] | None = None" "$TYPES"; then
    python3 - "$TYPES" <<'PY'
import sys
p = sys.argv[1]
src = open(p).read()
old = 'class SampleOutput(BaseModel):\n    sequences: list[GeneratedSequence]\n    prompt_logprobs: list[float] | None = None\n'
new = old + '    topk_prompt_logprobs: list[list[tuple[int, float]] | None] | None = None\n'
assert old in src, "SampleOutput preamble not found verbatim — refusing to patch."
src = src.replace(old, new, 1)
open(p, "w").write(src)
print("patched SampleOutput")
PY
fi

# ----------------------------------------------------------------------------
# 3) api.py — forward topk_prompt_logprobs from the request into SampleInput.
# ----------------------------------------------------------------------------
if ! grep -q "topk_prompt_logprobs=request.topk_prompt_logprobs" "$API"; then
    python3 - "$API" <<'PY'
import sys
p = sys.argv[1]
src = open(p).read()
old = '            prompt_logprobs=request.prompt_logprobs if request.prompt_logprobs is not None else False,\n        ),\n    )\n'
new = '            prompt_logprobs=request.prompt_logprobs if request.prompt_logprobs is not None else False,\n            topk_prompt_logprobs=request.topk_prompt_logprobs,\n        ),\n    )\n'
assert old in src, "api.py SampleInput call site not found verbatim — refusing to patch."
src = src.replace(old, new, 1)
open(p, "w").write(src)
print("patched api.py SampleInput call")
PY
fi

# ----------------------------------------------------------------------------
# 4) skyrl_train_inference_forwarding.py — request top-K from vLLM and pack
#    the prompt_logprobs response back into SampleOutput.topk_prompt_logprobs.
#
#    vLLM /v1/completions accepts `prompt_logprobs: int` (vLLM extension to
#    the OpenAI API) and returns `choices[0].prompt_logprobs` as a list per
#    prompt position of {token_id_str: {logprob, rank, decoded_token, ...}}.
# ----------------------------------------------------------------------------
if ! grep -q "topk_prompt_logprobs_k = getattr" "$FWD"; then
    python3 - "$FWD" <<'PY'
import sys
p = sys.argv[1]
src = open(p).read()

# 4a) Inject the topk_k extraction + payload field, right before the line
#     'sp = sample_req.sampling_params'.
old_pre = '        sp = sample_req.sampling_params\n        payload = {\n            "model": model_name,\n            "prompt": prompt_tokens,\n            "n": sample_req.num_samples,\n            "seed": sp.seed,\n            "max_tokens": sp.max_tokens,\n            "temperature": sp.temperature,\n            "top_p": sp.top_p,\n            "top_k": sp.top_k,\n            # vllm-router rejects boolean; 1 = return the chosen token\'s logprob.\n            "logprobs": 1,\n            "stream": False,\n            "return_token_ids": True,\n        }\n'
new_pre = '        topk_prompt_logprobs_k = getattr(sample_req, "topk_prompt_logprobs", 0) or 0\n        sp = sample_req.sampling_params\n        payload = {\n            "model": model_name,\n            "prompt": prompt_tokens,\n            "n": sample_req.num_samples,\n            "seed": sp.seed,\n            "max_tokens": sp.max_tokens,\n            "temperature": sp.temperature,\n            "top_p": sp.top_p,\n            "top_k": sp.top_k,\n            # vllm-router rejects boolean; 1 = return the chosen token\'s logprob.\n            "logprobs": 1,\n            "stream": False,\n            "return_token_ids": True,\n        }\n        if topk_prompt_logprobs_k > 0:\n            # vLLM extension to the OpenAI completions API: returns top-K\n            # logprobs at each prompt position in choices[0].prompt_logprobs.\n            payload["prompt_logprobs"] = topk_prompt_logprobs_k\n'
assert old_pre in src, "_forward payload block not found verbatim — refusing to patch."
src = src.replace(old_pre, new_pre, 1)

# 4b) Replace the final return to include topk_prompt_logprobs parsed from
#     the first choice's prompt_logprobs.
old_ret = '        return types.SampleOutput(sequences=sequences, prompt_logprobs=None)\n'
new_ret = (
    '        # Parse vLLM\'s prompt_logprobs (when requested). Format per\n'
    '        # position: dict[token_id_str -> {"logprob": float, "rank": int,\n'
    '        # "decoded_token": str}]. None at positions where vLLM didn\'t\n'
    '        # compute logprobs (e.g. the very first prompt token).\n'
    '        topk_pl = None\n'
    '        if topk_prompt_logprobs_k > 0 and result.get("choices"):\n'
    '            raw_pl = result["choices"][0].get("prompt_logprobs")\n'
    '            if isinstance(raw_pl, list):\n'
    '                topk_pl = []\n'
    '                for pos in raw_pl:\n'
    '                    if not isinstance(pos, dict):\n'
    '                        topk_pl.append(None)\n'
    '                        continue\n'
    '                    items = []\n'
    '                    for k, v in pos.items():\n'
    '                        try:\n'
    '                            tok = int(k)\n'
    '                            lp = float(v["logprob"] if isinstance(v, dict) else v)\n'
    '                        except (KeyError, TypeError, ValueError):\n'
    '                            continue\n'
    '                        items.append((tok, lp))\n'
    '                    items.sort(key=lambda x: x[1], reverse=True)\n'
    '                    topk_pl.append(items[:topk_prompt_logprobs_k] if items else None)\n'
    '        return types.SampleOutput(\n'
    '            sequences=sequences, prompt_logprobs=None,\n'
    '            topk_prompt_logprobs=topk_pl,\n'
    '        )\n'
)
assert old_ret in src, "_forward return statement not found verbatim — refusing to patch."
src = src.replace(old_ret, new_ret, 1)

open(p, "w").write(src)
print("patched skyrl_train_inference_forwarding.py")
PY
fi

echo "=== verifying patched files parse ==="
python3 -c "import ast; ast.parse(open('$TYPES').read()); print('types.py: OK')"
python3 -c "import ast; ast.parse(open('$API').read()); print('api.py: OK')"
python3 -c "import ast; ast.parse(open('$FWD').read()); print('forwarding.py: OK')"
echo "=== topk patch applied ==="
