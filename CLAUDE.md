# Working in this repo

`evsys-sdk` is both a Python SDK and a Claude Code plugin. A few
conventions to keep in mind when editing.

## Two skills directories (deliberately not synced)

This repo maintains skills in two places. They serve different audiences and
are allowed to diverge.

| Path | Loaded when | Audience |
|---|---|---|
| `skills/` | This repo is consumed as a plugin (`claude --plugin-dir ../evsys-sdk` from a research project) | Researchers using the SDK to run experiments |
| `.claude/skills/` | Claude is launched inside this repo (`cd evsys-sdk && claude`) | SDK developers editing the library itself |

Adding a skill:
  * Put it in `skills/` if researchers (plugin users) need it — e.g.
    `set-up-research-project`, `using-the-sdk`.
  * Put it in `.claude/skills/` if it's only useful while hacking on the SDK
    — e.g. detailed dashboard-write docs for testing the OOP API.
  * Put it in both (with the same name) only if the content is genuinely the
    same for both audiences. The two dirs are independent — no symlinks, no
    auto-sync. Update each on its own as the relevant skill evolves.

The plugin manifest is `.claude-plugin/plugin.json`; the marketplace entry is
`.claude-plugin/marketplace.json`. Both reference `skills/` and `agents/`.

## Dev workflow

  * Environment: `uv` manages `.venv` in the repo root. Run `uv sync` after
    pulling.
  * Tests: `.venv/bin/python -m pytest tests/ -q`. Full suite must stay green
    before any commit lands on `dev`.
  * Coverage: `.venv/bin/python -m coverage run --include='src/evsys_sdk/<module>.py' -m pytest tests/test_<module>.py -q && .venv/bin/python -m coverage report -m`.
    New modules target ≥ 90% line coverage.
  * Commits: keep them minimal — one new class or one logical change per
    commit, code + tests together.

## Extension points: the registry + Config + `{kind, params}` pattern

Anything a user can define and select by name — algorithms, transforms,
callbacks, verifiers, metrics, data stores, log stores, backends, inference
clients — follows ONE convention. When you add a new extension *kind*, or a
new built-in of an existing kind, mirror this exactly:

  1. **A registry** in `src/evsys_sdk/registry.py`: a `Registry("<kind>")`
     instance plus `register_<kind>` / `get_<kind>` / `list_<kind>s` functions,
     and an entry in `_all_registries()`.
  2. **Each implementation** is a class carrying two ClassVars — `name`
     (the string used in YAML) and `Config` (a Pydantic model, `extra="forbid"`,
     describing its params) — and decorated with `@register_<kind>("<name>")`.
  3. **A YAML surface**: a `<Kind>Spec` model in `config.py` (`{kind, params}`)
     and a `list[<Kind>Spec]` field on whatever config owns it.
  4. **A factory** that resolves specs → instances: look up the class via
     `get_<kind>(spec.kind)`, validate `spec.params` against the class's
     `Config`, then construct. See `training/callbacks.py::build_callbacks` and
     `transforms` for the two reference implementations.

The payoff: a researcher enables a feature from `config.yaml` with
`{kind: <name>, params: {...}}`, and **registers their own** with the same
decorator in their project — no SDK edit, no subclassing the library. Keep new
extension points consistent with this so the whole surface stays predictable.

## Useful entry points

  * `src/evsys_sdk/__init__.py` — public surface; what researchers
    import.
  * `agents/training-decider.md` — the agent that materializes new
    experiments end-to-end via the SDK.
  * `docs/DESIGN.md` — layout + protocol rationale; researcher-project
    section explains the on-disk shape `evsys init-project` creates.

## Running on your own infrastructure

Three orthogonal choices, each its own extension point:

| Question | Extension point | Built-ins |
|---|---|---|
| Which protocol does training speak? | `backend` | `tinker` (hosted), `skyrl` (self-hosted), `local`, `mock` |
| Whose hardware speaks it? | `compute` | `skypilot` |
| Where does the *agent* run? | `sandbox` | `e2b`, `modal`, `local` |

The hosted default needs no `compute:` at all. To run the same `config.yaml`
on infrastructure you control:

```yaml
backend:
  kind: skyrl                    # serves the Tinker protocol on your GPUs
  params:
    compute:
      kind: skypilot             # provisions the machine it runs on
      params:
        infra: aws               # or k8s / gcp / runpod / …
        accelerators: "L4:1"     # omit for a CPU cluster (JAX backend)
        idle_minutes_to_autostop: 30
```

`SkyRLBackend.prepare()` brings the compute up, then exports `TINKER_BASE_URL`.
Every client in a run — the training client, sampling clients, and harbor's
`TinkerLLM` in the rollout engine — constructs `ServiceClient()` with no
arguments, so that one variable redirects the whole run. `teardown()` releases
the cluster; SkyPilot autostops it regardless if the host dies.

Two defaults are deliberate and should not be relaxed casually:
`idle_minutes_to_autostop` cannot be 0 (a leaked GPU cluster bills by the
hour), and `remote_identity` is `NO_UPLOAD` because SkyPilot otherwise copies
your cloud credentials onto a VM that runs agent-authored code.
