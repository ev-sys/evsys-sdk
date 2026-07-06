---
name: tune-trigger
description: >
  How the trigger agent retunes the cheap deterministic gate — editing
  .evsys/triggers/policy.json (fn kind, params, every_n, window, signals) so the
  gate escalates on real failures and stays quiet on noise. Use after judging an
  escalation when the gate itself fired too eagerly or too rarely.
---

# Tuning the deterministic gate (the self-improving loop)

The gate is one editable artifact: **`.evsys/triggers/policy.json`**. The runtime
re-reads it live every cycle, so your edits take effect on the next evaluation —
no restart, and they survive one (the persisted policy is authoritative, not the
YAML that seeded it). This is how the cheap stage gets smarter from your
judgement: you just assessed a batch; if the gate mis-fired, fix the gate.

## The knobs

```json
{
  "kind": "recent_failures",
  "params": {"threshold": 0.4, "reward_below": 0.5, "min_traces": 5},
  "every_n": 20,
  "window": 100,
  "signals": ["reward", "status", "input_sig", "n_tool_calls", "timestamp"]
}
```

- **`kind`** — which registered `@register_trigger` fn runs. Point it at a
  better one (or a new one you register) if the current gate is the wrong shape.
- **`params`** — the fn's thresholds (validated against its `Config`). The most
  common tune: raise a `threshold` if it over-fires, lower it if it misses.
- **`every_n`** — evaluate every N ingested traces. Raise it if escalations are
  too frequent / noisy; lower it to react faster to a fresh regression.
- **`window`** — how many recent traces the state keeps. A *smaller* window makes
  the gate sensitive to recent regressions (old successes stop diluting a fresh
  failure spike); a larger one smooths noise.
- **`signals`** — which summary fields the state tracks. Widen this if a smarter
  fn needs a field it isn't recording yet.

## When to tune

- **Over-fired (this escalation was noise):** raise `threshold`, raise `every_n`,
  or shrink `window` so a couple of bad traces don't trip it. Record in the verdict
  that you tightened it and why.
- **Under-firing (you can see real failures the gate missed):** lower `threshold`,
  lower `every_n`, or shrink `window` for recency. Or switch `kind` to a fn that
  captures the failure shape (e.g. a per-tool failure gate rather than a global rate).
- **Wrong signal entirely:** if the failure mode isn't visible in the tracked
  `signals`, add the field and, if needed, register a fn that reads it.

## How to edit

Read `policy.json`, change only the fields you intend, and write it back as valid
JSON (keep the other keys). Make the *smallest* change that fixes the observed
mis-fire — don't rewrite the whole policy on one escalation. Prefer a `params`
tweak over swapping `kind`. Always note the change (old → new + why) in your
verdict so the tuning is auditable.

**Don't** disable the gate (`every_n` to absurd values) to silence noise — that
blinds the system. Tune it to be *right*, not quiet. See
[assess-traces](../assess-traces/SKILL.md) for the judgement that precedes a tune.
