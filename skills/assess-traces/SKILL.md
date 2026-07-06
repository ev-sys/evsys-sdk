---
name: assess-traces
description: >
  How the trigger agent judges an escalated batch of production traces — is this a
  real, recurring, learnable failure worth autoresearch, or noise? Use when
  deciding a trigger escalation's verdict. Reads the canonical evsys Trace form
  (OpenAI messages + per-turn feedback + metadata).
---

# Assessing an escalated batch of traces

The deterministic gate is cheap and blunt: it fires on an aggregate signal
(failure rate, feedback drop, drift, volume). Your job is the judgement it can't
make — **is this batch a real, learnable failure mode?** A YES commits a full
autoresearch experiment, so be a skeptic.

## What you're reading

Each trace (a line in `traces/<source>/traces.jsonl`, or via
`evsys_sdk.iter_traces_jsonl(path)`) is:

- `messages` — the whole conversation in OpenAI format: `system` / `user` /
  `assistant` (may carry `tool_calls`) / `tool` results. `trace.input` = first
  user message; `trace.output` = last assistant message.
- `feedback` — `{key, score, comment, source, turn}`; `turn` indexes into
  `messages` (`None` = whole-trace). This is the ground truth of "did it go well".
- `metadata` — `source`, `model`, `status` (`success`/`error`), `tags`, `timestamp`.

Load just the `trace_ids` named in the escalation event — not the whole store.

## The judgement (in order)

1. **Real vs artifact.** Are the failures genuine, or an artifact of *how feedback
   was collected* (a broken grader, a single angry user, mislabeled `status`)? Read
   the actual `messages` for a few implicated traces. If the "failures" look fine on
   inspection, it's noise → **NO** + tighten the gate.
2. **Recurring vs one-off.** Does the same failure shape repeat across traces (same
   tool misused, same step skipped, same class of input mishandled)? One weird
   trace is not a pattern → **NO**. A pattern across ≥ several traces → candidate.
3. **Learnable vs environmental.** Could a prompt/harness change plausibly fix it,
   or is it an outage / rate-limit / bad-tool-response the model can't control?
   Only *learnable* failures are worth autoresearch.
4. **Stateable hypothesis.** If YES, you must be able to say in one line *what to
   change and why* (e.g. "agent calls `search` before `plan` on multi-step tasks →
   try a harness that forces a plan step"). No crisp hypothesis → **NO**.

## The verdict

Write the JSON verdict your spawn prompt names:

```json
{
  "escalation": "escalation-00000010.json",
  "worth_autoresearch": true,
  "reasoning": "6/6 implicated traces skip the plan step on multi-tool tasks; feedback 'resolved'=0 on all.",
  "hypothesis": "Force an explicit plan turn before tool calls on multi-step inputs.",
  "trace_ids": ["prod-006", "prod-008", "prod-009"]
}
```

Bias to **NO** when the batch is thin, noisy, or you can't name a hypothesis — an
escalation is cheap to drop; a wrong YES burns a whole experiment. See
[tune-trigger](../tune-trigger/SKILL.md) when the gate itself needs adjusting.
