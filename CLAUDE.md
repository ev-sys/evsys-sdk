# Working in this repo

`evsys-sdk` is both a Python SDK and a coding-agent plugin (Claude Code and
Cursor, from one shared `skills/` source). A few conventions to keep in mind
when editing.

## Two skills directories (deliberately not synced)

This repo maintains skills in two places. They serve different audiences and
are allowed to diverge.

| Path | Loaded when | Audience |
|---|---|---|
| `skills/` | This repo is consumed as a plugin (Claude `--plugin-dir ../evsys-sdk`, or the Cursor plugin) from a research project | Researchers using the SDK to run experiments |
| `.claude/skills/` | Claude is launched inside this repo (`cd evsys-sdk && claude`) | SDK developers editing the library itself |

Adding a skill:
  * Put it in `skills/` if researchers (plugin users) need it — e.g.
    `set-up-research-project`, `using-the-sdk`.
  * Put it in `.claude/skills/` if it's only useful while hacking on the SDK
    — e.g. detailed dashboard-write docs for testing the OOP API.
  * Put it in both (with the same name) only if the content is genuinely the
    same for both audiences. The two dirs are independent — no symlinks, no
    auto-sync. Update each on its own as the relevant skill evolves.

The plugin ships to both coding agents from a single `skills/` source:
 * Claude Code: `.claude-plugin/plugin.json` (manifest) +
 `.claude-plugin/marketplace.json` (marketplace entry).
 * Cursor: `.cursor-plugin/plugin.json` (manifest) +
 `.cursor-plugin/marketplace.json` (marketplace entry).

All four point at `skills/` — edit a skill once and both agents pick it up. Skill
folder names and their frontmatter `name` must be identical kebab-case.

Agents live in `.claude/agents/<name>.md` (one Markdown file per agent:
frontmatter `name` + `description`, body = the agent's system prompt). The
filename must match the frontmatter `name` (kebab-case). Reusable decision/loop
logic an agent depends on should still live as a skill so both the agent and
plugin consumers share it.

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
  * `skills/autoresearch-launch/SKILL.md` — the skill that decides and
    materializes new experiments end-to-end via the SDK.
  * `docs/DESIGN.md` — layout + protocol rationale; researcher-project
    section explains the on-disk shape `evsys init-project` creates.
