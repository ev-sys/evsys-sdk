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

## Useful entry points

  * `src/evsys_sdk/__init__.py` — public surface; what researchers
    import.
  * `agents/training-decider.md` — the agent that materializes new
    experiments end-to-end via the SDK.
  * `docs/DESIGN.md` — layout + protocol rationale; researcher-project
    section explains the on-disk shape `evsys init-project` creates.
