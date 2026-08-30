---
name: diagnose-experiment
description: >
  Diagnose a training experiment end-to-end: rendered training inputs (after
  transforms and templates), ground-truth labels, train/eval prompt parity,
  and final-model success/failure examples. Writes DIAGNOSIS.md with checks,
  bugs, and improvements. Use when debugging bad eval scores, train/eval
  mismatch, template bugs, coverage gaps, or when the user asks to diagnose
  or audit an experiment.
---

# Diagnose experiment

Full workflow: [../../skills/diagnose-experiment/SKILL.md](../../skills/diagnose-experiment/SKILL.md)

Report template: [../../skills/diagnose-experiment/report-template.md](../../skills/diagnose-experiment/report-template.md)

When hacking on the SDK, these modules implement the surfaces the skill audits:

| Step | Source |
|------|--------|
| Transforms | `src/evsys_sdk/runner.py::_apply_transforms` |
| SDFT user/demo templates | `src/evsys_sdk/algorithms/sdft.py`, `training/sdft_data.py` |
| Eval chat wrap | `src/evsys_sdk/inference/chat_templated.py` |
| Benchmark eval | `src/evsys_sdk/experiment.py::_eval_arm` |
| Harbor rollouts | `src/evsys_sdk/training/harbor_eval.py` |
