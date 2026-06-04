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
# Soft-target CE patch files (multi-target log-prob gather).
ENGINE="$ROOT/skyrl/tinker/engine.py"
BACKEND="$ROOT/skyrl/backends/skyrl_train_backend.py"
REPLAY="$ROOT/skyrl/train/dataset/replay_buffer.py"
WORKUTIL="$ROOT/skyrl/backends/skyrl_train/workers/worker_utils.py"
MWORKER="$ROOT/skyrl/backends/skyrl_train/workers/megatron/megatron_worker.py"
MMODEL="$ROOT/skyrl/backends/skyrl_train/workers/megatron/megatron_model_wrapper.py"

for f in "$TYPES" "$API" "$FWD" "$ENGINE" "$BACKEND" "$REPLAY" "$WORKUTIL" "$MWORKER" "$MMODEL"; do
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

# ============================================================================
# SOFT-TARGET CE PATCH — top-K forward-KL distillation on SkyRL.
#
# Why: The cookbook's `build_topk_distillation_datums` produces datums with
# `target_tokens` and `weights` of shape (N, K). Stock SkyRL's
# cross_entropy_loss only supports single-target (N,)-shape data; the (N, K)
# packed form blows up in `(loss * loss_mask)` (~ppo_utils.reduce_loss). The
# correct math is `-sum_k weight_k * log p_student(token_k)` per position.
#
# What: 7 surgical edits across 6 files thread `target_tokens_NK` and
# `weights_NK` from `loss_fn_inputs.target_tokens.shape == (N, K)` all the
# way through to `megatron_model_wrapper.loss_func`, which:
#   - keeps a single model forward (logits computed once)
#   - re-gathers logprobs K times from the same logits at K different
#     target positions (cheap index ops, dominated by the forward)
#   - computes `-(weights_NK * log_p_BNK).sum()` as the loss
#
# Standard single-target path is untouched (target_tokens_NK is None).
# ============================================================================

# ----------------------------------------------------------------------------
# S1) types.py — add `all_target_shapes` to PreparedModelPassBatch so the
#     shape info on TensorData survives the data → batch transition.
# ----------------------------------------------------------------------------
if ! grep -q "all_target_shapes: list" "$TYPES"; then
    python3 - "$TYPES" <<'PY'
import sys
p = sys.argv[1]
src = open(p).read()
old = '    # Mapping from examples back to requests: (request_id, model_id, start_idx, end_idx)\n    request_batch_slices: list[tuple[str, str, int, int]]\n'
new = old + '\n    # Original shape of target_tokens for each datum (None or 1D = standard\n    # single-target; 2D [N, K] = soft top-K distillation, threaded through\n    # _to_training_batch into target_tokens_NK + weights_NK tensors).\n    all_target_shapes: list[list[int] | None] = []\n'
assert old in src, "types.py PreparedModelPassBatch tail not found verbatim."
src = src.replace(old, new, 1)
open(p, "w").write(src)
print("patched types.py: all_target_shapes")
PY
fi

# ----------------------------------------------------------------------------
# S2) engine.py — populate all_target_shapes from target_tokens.shape.
# ----------------------------------------------------------------------------
if ! grep -q "all_target_shapes.append" "$ENGINE"; then
    python3 - "$ENGINE" <<'PY'
import sys
p = sys.argv[1]
src = open(p).read()
# Insert init for all_target_shapes alongside the other all_* lists.
old1 = '    all_targets = []\n    all_token_weights = []\n'
new1 = '    all_targets = []\n    all_target_shapes = []\n    all_token_weights = []\n'
assert old1 in src, "engine.py all_* init block not found verbatim."
src = src.replace(old1, new1, 1)

# Insert append alongside the target_tokens append.
old2 = '            all_targets.append(loss_fn_inputs.target_tokens.data)\n            all_token_weights.append(loss_fn_inputs.weights.data)\n'
new2 = '            all_targets.append(loss_fn_inputs.target_tokens.data)\n            all_target_shapes.append(loss_fn_inputs.target_tokens.shape)\n            all_token_weights.append(loss_fn_inputs.weights.data)\n'
assert old2 in src, "engine.py all_targets.append not found verbatim."
src = src.replace(old2, new2, 1)

# Add to PreparedModelPassBatch constructor call.
old3 = '        all_loss_fn_configs=all_loss_fn_configs,\n        request_batch_slices=request_batch_slices,\n    )\n'
new3 = '        all_loss_fn_configs=all_loss_fn_configs,\n        request_batch_slices=request_batch_slices,\n        all_target_shapes=all_target_shapes,\n    )\n'
assert old3 in src, "engine.py PreparedModelPassBatch ctor tail not found verbatim."
src = src.replace(old3, new3, 1)

open(p, "w").write(src)
print("patched engine.py: thread all_target_shapes")
PY
fi

# ----------------------------------------------------------------------------
# S3) skyrl_train_backend.py — in _to_training_batch, detect 2D target
#     shapes and build (B, max_response_len, K) tensors. Add them to
#     batch_dict as `target_tokens_NK` and `weights_NK`. Also override
#     loss_mask to be a per-position binary mask derived from weights_NK.
# ----------------------------------------------------------------------------
if ! grep -q "target_tokens_NK" "$BACKEND"; then
    python3 - "$BACKEND" <<'PY'
import sys
p = sys.argv[1]
src = open(p).read()
# Insert the multi-target detection + tensor build right BEFORE the
# `sequences_tensor = torch.tensor(...)` line, AFTER the for-loop that
# builds flat lists. We use the existing flat all_targets / all_token_weights
# to recover (N, K) by reshaping with all_target_shapes.
old = '        sequences_tensor = torch.tensor(sequences, dtype=torch.long)\n        attention_mask_tensor = torch.tensor(attention_masks, dtype=torch.long)\n        loss_mask_tensor = torch.tensor(loss_masks, dtype=torch.float32)\n        response_mask_tensor = torch.tensor(response_masks, dtype=torch.long)\n'
new = (
    '        # ----- Soft-target detection -----\n'
    '        # When any datum has 2D target_tokens shape (N, K), build per-position\n'
    '        # (B, max_response_len, K) tensors and override the flat loss_mask\n'
    '        # built above with a per-position binary mask. Single-target batches\n'
    '        # skip this block entirely.\n'
    '        target_shapes = list(getattr(prepared_batch, "all_target_shapes", []) or [])\n'
    '        is_multitarget = any(\n'
    '            s is not None and len(s) == 2 and s[1] > 1 for s in target_shapes\n'
    '        )\n'
    '        target_tokens_NK_tensor = None\n'
    '        weights_NK_tensor = None\n'
    '        if is_multitarget:\n'
    '            K_vals = [s[1] for s in target_shapes if s is not None and len(s) == 2]\n'
    '            if len(set(K_vals)) > 1:\n'
    '                raise ValueError(f"Mixed K values within one batch: {sorted(set(K_vals))}")\n'
    '            K = K_vals[0]\n'
    '            # Recompute max_response_len in N space (not N*K). Each datum\'s\n'
    '            # weights flat list has length N*K, so divide by K.\n'
    '            per_datum_N = [\n'
    '                (len(w) // K) if s is not None and len(s) == 2 else len(w)\n'
    '                for w, s in zip(prepared_batch.all_token_weights, target_shapes)\n'
    '            ]\n'
    '            mt_max_response_len = max(per_datum_N) if per_datum_N else 0\n'
    '            target_NKs = []\n'
    '            weight_NKs = []\n'
    '            binary_loss_masks = []\n'
    '            for idx, (targets, weights) in enumerate(zip(\n'
    '                prepared_batch.all_targets, prepared_batch.all_token_weights\n'
    '            )):\n'
    '                shape = target_shapes[idx]\n'
    '                if shape is not None and len(shape) == 2:\n'
    '                    N_i, K_i = int(shape[0]), int(shape[1])\n'
    '                    t_NK = torch.tensor(list(targets), dtype=torch.long).view(N_i, K_i)\n'
    '                    w_NK = torch.tensor(list(weights), dtype=torch.float32).view(N_i, K_i)\n'
    '                else:\n'
    '                    # Single-target datum mixed into multi-target batch: lift\n'
    '                    # to (N, K) by putting the target at slot 0, weight at slot 0.\n'
    '                    N_i = len(targets)\n'
    '                    t_NK = torch.zeros(N_i, K, dtype=torch.long)\n'
    '                    w_NK = torch.zeros(N_i, K, dtype=torch.float32)\n'
    '                    if N_i:\n'
    '                        t_NK[:, 0] = torch.tensor(list(targets), dtype=torch.long)\n'
    '                        w_NK[:, 0] = torch.tensor(list(weights), dtype=torch.float32)\n'
    '                pad = mt_max_response_len - N_i\n'
    '                if pad > 0:\n'
    '                    t_NK = torch.cat([torch.zeros(pad, K, dtype=torch.long), t_NK], dim=0)\n'
    '                    w_NK = torch.cat([torch.zeros(pad, K, dtype=torch.float32), w_NK], dim=0)\n'
    '                target_NKs.append(t_NK)\n'
    '                weight_NKs.append(w_NK)\n'
    '                # Per-position binary mask: 1 where any slot has weight > 0.\n'
    '                pos_mask = (w_NK.sum(dim=-1) > 0).to(torch.float32)\n'
    '                binary_loss_masks.append(pos_mask)\n'
    '            target_tokens_NK_tensor = torch.stack(target_NKs, dim=0)\n'
    '            weights_NK_tensor = torch.stack(weight_NKs, dim=0)\n'
    '            # Replace the flat loss_masks (built above with wrong length) with\n'
    '            # the per-position binary mask aligned to mt_max_response_len.\n'
    '            loss_masks = [m.tolist() for m in binary_loss_masks]\n'
    '\n'
    '        sequences_tensor = torch.tensor(sequences, dtype=torch.long)\n'
    '        attention_mask_tensor = torch.tensor(attention_masks, dtype=torch.long)\n'
    '        loss_mask_tensor = torch.tensor(loss_masks, dtype=torch.float32)\n'
    '        response_mask_tensor = torch.tensor(response_masks, dtype=torch.long)\n'
)
assert old in src, "_to_training_batch tensor build line not found verbatim."
src = src.replace(old, new, 1)

# Add target_tokens_NK / weights_NK to batch_dict.
old2 = '        batch_dict = {\n            "sequences": sequences_tensor,\n            "attention_mask": attention_mask_tensor,\n            "loss_mask": loss_mask_tensor,\n            "response_mask": response_mask_tensor,\n        }\n'
new2 = '        batch_dict = {\n            "sequences": sequences_tensor,\n            "attention_mask": attention_mask_tensor,\n            "loss_mask": loss_mask_tensor,\n            "response_mask": response_mask_tensor,\n        }\n        if target_tokens_NK_tensor is not None:\n            batch_dict["target_tokens_NK"] = target_tokens_NK_tensor\n            batch_dict["weights_NK"] = weights_NK_tensor\n'
assert old2 in src, "_to_training_batch batch_dict not found verbatim."
src = src.replace(old2, new2, 1)

open(p, "w").write(src)
print("patched skyrl_train_backend.py: _to_training_batch multi-target")
PY
fi

# ----------------------------------------------------------------------------
# S4) replay_buffer.py — Experience dataclass adds target_tokens_NK +
#     weights_NK with .to_device() handling.
# ----------------------------------------------------------------------------
if ! grep -q "target_tokens_NK" "$REPLAY"; then
    python3 - "$REPLAY" <<'PY'
import sys
p = sys.argv[1]
src = open(p).read()
# Add dataclass fields. They land before `@torch.no_grad()` decorator on
# to_device.
old = '    pixel_values: Optional[TensorList] = None\n    image_grid_thw: Optional[TensorList] = None\n\n    @torch.no_grad()\n    def to_device(self, device: torch.device) -> None:\n'
new = '    pixel_values: Optional[TensorList] = None\n    image_grid_thw: Optional[TensorList] = None\n    # Soft top-K distillation: (B, response_len, K) per-slot teacher targets +\n    # renormalized probabilities. None for single-target path (untouched).\n    target_tokens_NK: Optional[torch.Tensor] = None\n    weights_NK: Optional[torch.Tensor] = None\n\n    @torch.no_grad()\n    def to_device(self, device: torch.device) -> None:\n'
assert old in src, "replay_buffer.py Experience tail not found verbatim."
src = src.replace(old, new, 1)

# Extend to_device to move new tensors.
old2 = '        if self.image_grid_thw is not None:\n            self.image_grid_thw = self.image_grid_thw.to(device)\n'
new2 = '        if self.image_grid_thw is not None:\n            self.image_grid_thw = self.image_grid_thw.to(device)\n        if self.target_tokens_NK is not None:\n            self.target_tokens_NK = to(self.target_tokens_NK, device)\n        if self.weights_NK is not None:\n            self.weights_NK = to(self.weights_NK, device)\n'
assert old2 in src, "replay_buffer.py to_device tail not found verbatim."
src = src.replace(old2, new2, 1)

open(p, "w").write(src)
print("patched replay_buffer.py: Experience target_tokens_NK + weights_NK")
PY
fi

# ----------------------------------------------------------------------------
# S5) worker_utils.py — batch_to_experience pulls new fields from batch dict.
# ----------------------------------------------------------------------------
if ! grep -q "target_tokens_NK=batch.get" "$WORKUTIL"; then
    python3 - "$WORKUTIL" <<'PY'
import sys
p = sys.argv[1]
src = open(p).read()
old = '            rollout_logprobs=batch.get("rollout_logprobs"),\n            rollout_expert_indices=batch.get("rollout_expert_indices"),\n'
new = '            rollout_logprobs=batch.get("rollout_logprobs"),\n            rollout_expert_indices=batch.get("rollout_expert_indices"),\n            target_tokens_NK=batch.get("target_tokens_NK"),\n            weights_NK=batch.get("weights_NK"),\n'
assert old in src, "worker_utils.py batch_to_experience not found verbatim."
src = src.replace(old, new, 1)
open(p, "w").write(src)
print("patched worker_utils.py: batch_to_experience NK kwargs")
PY
fi

# ----------------------------------------------------------------------------
# S6) megatron_worker.py — include NK tensors in micro_buffer dict so
#     loss_func can see them via `data["target_tokens_NK"]`.
# ----------------------------------------------------------------------------
if ! grep -q '"target_tokens_NK": experience.target_tokens_NK' "$MWORKER"; then
    python3 - "$MWORKER" <<'PY'
import sys
p = sys.argv[1]
src = open(p).read()
old = '                    "action_mask": experience.action_mask,\n                    "rollout_expert_indices": rollout_expert_indices if self.enable_router_replay else None,\n                }\n'
new = '                    "action_mask": experience.action_mask,\n                    "rollout_expert_indices": rollout_expert_indices if self.enable_router_replay else None,\n                    "target_tokens_NK": experience.target_tokens_NK,\n                    "weights_NK": experience.weights_NK,\n                }\n'
assert old in src, "megatron_worker.py micro_buffer.append not found verbatim."
src = src.replace(old, new, 1)
open(p, "w").write(src)
print("patched megatron_worker.py: micro_buffer NK keys")
PY
fi

# ----------------------------------------------------------------------------
# S7) megatron_model_wrapper.py — loss_func multi-target branch. Detect
#     target_tokens_NK in data, do K-fold gather from same logits, compute
#     forward-KL weighted loss, override elementwise_loss for the SFT output
#     path. The model forward is shared (1×); each re-gather is a single
#     parallel-reduce + index op (cheap).
# ----------------------------------------------------------------------------
if ! grep -q "target_tokens_NK = data.get" "$MMODEL"; then
    python3 - "$MMODEL" <<'PY'
import sys
p = sys.argv[1]
src = open(p).read()

# (a) Replace the current_loss_fn call with a conditional multi-target branch.
old_a = (
    '            action_log_probs = token_logprobs[:, -num_actions:]\n'
    '\n'
    '            # policy loss should be calculated based on the selected token logprobs\n'
    '            policy_loss, loss_metrics = current_loss_fn(\n'
    '                action_log_probs,\n'
    '                old_action_log_probs,\n'
    '                advantages,\n'
    '                config=loss_config,\n'
    '                loss_mask=loss_mask,\n'
    '                rollout_logprobs=rollout_action_logprobs,\n'
    '            )\n'
)
new_a = (
    '            action_log_probs = token_logprobs[:, -num_actions:]\n'
    '\n'
    '            # ----- Soft-target top-K distillation branch -----\n'
    '            # If target_tokens_NK and weights_NK are present, do forward-KL\n'
    '            # distillation: gather student logprobs at K targets per position\n'
    '            # (from the same logits — model forward is shared), weight by\n'
    '            # renormalized teacher probabilities, sum. Otherwise fall through\n'
    '            # to standard current_loss_fn path.\n'
    '            target_tokens_NK = data.get("target_tokens_NK")\n'
    '            weights_NK = data.get("weights_NK")\n'
    '            _mt_elementwise_loss = None\n'
    '            if (\n'
    '                target_tokens_NK is not None\n'
    '                and weights_NK is not None\n'
    '                and resolved_loss_name == "cross_entropy"\n'
    '            ):\n'
    '                K = int(target_tokens_NK.shape[-1])\n'
    '                per_k_log_probs = []\n'
    '                for k in range(K):\n'
    '                    seq_k = sequences.clone()\n'
    '                    seq_k[:, -num_actions:] = target_tokens_NK[:, :, k]\n'
    '                    token_lp_k = from_parallel_logits_to_logprobs(\n'
    '                        logits,\n'
    '                        seq_k,\n'
    '                        vocab_start_index=tp_rank * logits.shape[-1],\n'
    '                        vocab_end_index=(tp_rank + 1) * logits.shape[-1],\n'
    '                        tp_group=tp_grp,\n'
    '                        inference_only=False,\n'
    '                        cp_group=None,\n'
    '                        chunk_size=self.cfg.logprobs_chunk_size,\n'
    '                    )\n'
    '                    per_k_log_probs.append(token_lp_k[:, -num_actions:])\n'
    '                log_p_BNK = torch.stack(per_k_log_probs, dim=-1)\n'
    '                elementwise_BNK = -(weights_NK * log_p_BNK)\n'
    '                if loss_mask is not None:\n'
    '                    # Gate at the row level so padded micro-batch rows do not\n'
    '                    # contribute (padding gives row-0 copies of weights_NK).\n'
    '                    elementwise_BNK = elementwise_BNK * loss_mask.unsqueeze(-1)\n'
    '                policy_loss = elementwise_BNK.sum()\n'
    '                loss_metrics = {"clip_ratio": 0.0}\n'
    '                _mt_elementwise_loss = elementwise_BNK.sum(dim=-1).detach()\n'
    '                # Use argmax-K slot (slot 0 in cookbook layout) as the\n'
    '                # representative action_log_probs for output builders below.\n'
    '                action_log_probs = per_k_log_probs[0]\n'
    '            else:\n'
    '                # policy loss should be calculated based on the selected token logprobs\n'
    '                policy_loss, loss_metrics = current_loss_fn(\n'
    '                    action_log_probs,\n'
    '                    old_action_log_probs,\n'
    '                    advantages,\n'
    '                    config=loss_config,\n'
    '                    loss_mask=loss_mask,\n'
    '                    rollout_logprobs=rollout_action_logprobs,\n'
    '                )\n'
)
assert old_a in src, "megatron_model_wrapper.py loss_func current_loss_fn block not found verbatim."
src = src.replace(old_a, new_a, 1)

# (b) Override elementwise_loss with the multi-target version inside the SFT
#     output build branch.
old_b = (
    '                # Compute elementwise loss for Tinker API (per-token NLL)\n'
    '                with torch.no_grad():\n'
    '                    elementwise_loss = -action_log_probs\n'
    '                    if loss_mask is not None:\n'
    '                        elementwise_loss = elementwise_loss * loss_mask\n'
)
new_b = (
    '                # Compute elementwise loss for Tinker API (per-token NLL)\n'
    '                with torch.no_grad():\n'
    '                    if _mt_elementwise_loss is not None:\n'
    '                        # Per-position sum-over-K from the soft-target branch.\n'
    '                        elementwise_loss = _mt_elementwise_loss\n'
    '                    else:\n'
    '                        elementwise_loss = -action_log_probs\n'
    '                        if loss_mask is not None:\n'
    '                            elementwise_loss = elementwise_loss * loss_mask\n'
)
assert old_b in src, "megatron_model_wrapper.py elementwise_loss block not found verbatim."
src = src.replace(old_b, new_b, 1)

open(p, "w").write(src)
print("patched megatron_model_wrapper.py: loss_func multi-target branch")
PY
fi

echo "=== verifying patched files parse ==="
python3 -c "import ast; ast.parse(open('$TYPES').read()); print('types.py: OK')"
python3 -c "import ast; ast.parse(open('$API').read()); print('api.py: OK')"
python3 -c "import ast; ast.parse(open('$FWD').read()); print('forwarding.py: OK')"
python3 -c "import ast; ast.parse(open('$ENGINE').read()); print('engine.py: OK')"
python3 -c "import ast; ast.parse(open('$BACKEND').read()); print('skyrl_train_backend.py: OK')"
python3 -c "import ast; ast.parse(open('$REPLAY').read()); print('replay_buffer.py: OK')"
python3 -c "import ast; ast.parse(open('$WORKUTIL').read()); print('worker_utils.py: OK')"
python3 -c "import ast; ast.parse(open('$MWORKER').read()); print('megatron_worker.py: OK')"
python3 -c "import ast; ast.parse(open('$MMODEL').read()); print('megatron_model_wrapper.py: OK')"
echo "=== topk + soft-target patches applied ==="
