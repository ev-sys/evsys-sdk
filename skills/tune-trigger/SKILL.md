---
name: tune-trigger
description: >
  How the trigger agent evolves the cheap deterministic gate — editing
  .evsys/triggers/policy.json (fn kind, import_path, params, every_n, window),
  and authoring or rewriting the fn's own code, so the gate escalates on real
  failures and stays quiet on noise. Use after judging an escalation when the
  gate itself fired too eagerly or too rarely, or is the wrong shape entirely.
---

# Evolving the deterministic gate (the self-improving loop)

The gate is one editable artifact — **`.evsys/triggers/policy.json`** — plus the
**fn code it points at**. Both are yours. The runtime re-reads the policy live
every cycle and hot-reloads the fn's file when it changes, so your edits take
effect on the next evaluation — no restart, and they survive one (the persisted
policy is authoritative, not the YAML that seeded it). This is how the cheap
stage gets smarter from your judgement: you just assessed a batch; if the gate
mis-fired, fix the gate — its knobs, or its logic.

## The policy

```json
{
  "kind": "recent_failures",
  "import_path": "gate.py",
  "params": {"threshold": 0.4, "min_traces": 5},
  "every_n": 20,
  "window": 100
}
```

- **`kind`** — which registered `@register_trigger` fn runs. Point it at a
  better one (or a new one you author, below) if the current gate is the wrong shape.
- **`import_path`** — the `.py` file (or dotted module) the fn lives in. The driver
  re-imports it whenever this path OR the file's contents change, so editing the
  file or repointing here is how new logic goes live.
- **`params`** — the fn's thresholds (validated against its `Config`). The most
  common tune: raise a `threshold` if it over-fires, lower it if it misses.
- **`every_n`** — evaluate every N ingested traces. Raise it if escalations are
  too frequent / noisy; lower it to react faster to a fresh regression.
- **`window`** — how many recent traces the state keeps. A *smaller* window makes
  the gate sensitive to recent regressions; a larger one smooths noise.

## Three levels of tune — reach for the smallest that fixes it

1. **Knobs (most common).** Over-fired on noise → raise `threshold` / `every_n`,
   or shrink `window`. Under-firing on real failures → lower them. A `params`
   tweak is almost always enough; prefer it.

2. **Swap the fn.** If the failure shape is wrong for the current fn (e.g. you
   need per-tool failure rate, not a global rate), point `kind` + `import_path`
   at a different registered fn.

3. **Author / rewrite the fn (you own the code).** If no existing fn captures the
   failure, write one. Edit the file at `import_path` in place, or write a new
   `.py` and repoint `kind` + `import_path` at it. Contract:

   ```python
   from pydantic import BaseModel, ConfigDict
   from evsys_sdk import register_trigger
   from evsys_sdk.protocols import TriggerDecision

   @register_trigger("my_gate")            # this name is the policy `kind`
   class MyGate:
       name = "my_gate"
       class Config(BaseModel):
           model_config = ConfigDict(extra="forbid")
           threshold: float = 0.4
       def __init__(self, **params): self.cfg = self.Config(**params)
       def evaluate(self, state) -> TriggerDecision:
           # state.window = the raw traces ({trace_id, messages, feedback, metadata});
           # YOU decide what 'failure' means by reducing them.
           # state.extras is yours — stash any rolling memory (EWMA, seen-hashes,
           # a threshold you're adapting) and it persists untouched across evals.
           ...
   ```

   The driver hot-reloads on the next eval; a broken rewrite is error-isolated
   (logged `reload_error`) and never kills ingestion — but it also means the old
   fn keeps running until your file imports cleanly, so make sure it does.

## How to edit

Read `policy.json`, change only the fields you intend, write it back as valid JSON
(keep the other keys). Make the *smallest* change that fixes the observed mis-fire.
When you author or rewrite a fn, keep the `@register_trigger` name in sync with the
policy `kind`. Always note the change (old → new + why) in your verdict so the
tuning is auditable.
