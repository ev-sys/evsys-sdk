# Contributing to evsys-sdk

Thanks for contributing! `evsys-sdk` is both a Python SDK and a Claude Code
plugin. This guide covers how to set up, make a change, and get it merged.

For *internal architecture* conventions (the registry / `Config` /
`{kind, params}` extension pattern, the two skills directories, dev commands),
see [`CLAUDE.md`](./CLAUDE.md) and [`docs/DESIGN.md`](./docs/DESIGN.md). This
guide is the human-facing workflow that wraps those.

## Branching model

We use **short-lived feature branches off `main`, merged back via pull
request**. There is no longer a shared `dev` integration branch — branch
directly off `main` and open a PR against `main`.

```bash
git checkout main && git pull
git checkout -b <type>/<short-slug>
```

Branch name prefixes (match the commit type):

| Prefix | For |
|---|---|
| `feat/` | a new feature or capability |
| `fix/` | a bug fix |
| `refactor/` | internal change, no behavior difference |
| `docs/` | docs / skills / comments only |
| `chore/` | tooling, deps, CI, repo plumbing |
| `test/` | adding or reworking tests only |

**`main` is protected** — you cannot push to it directly. Every change lands
through a reviewed PR (see [Pull requests](#pull-requests)).

## Dev setup

`uv` manages a `.venv` in the repo root.

```bash
uv sync                 # create/refresh .venv from pyproject + uv.lock
```

Run anything through the venv, e.g. `.venv/bin/python -m pytest`.

## Making a change

- **Keep it minimal** — one new class or one logical change per commit, code +
  tests together. Avoid multi-class megacommits.
- **Follow the extension pattern.** Anything selectable by name (algorithms,
  transforms, callbacks, verifiers, metrics, stores, backends, …) goes through
  the registry + `Config` + `{kind, params}` convention. See the "Extension
  points" section of [`CLAUDE.md`](./CLAUDE.md) — mirror it exactly when adding
  a new *kind* or a new built-in.
- **Skills live in two places** (`skills/` for plugin users, `.claude/skills/`
  for SDK developers) and are intentionally not synced — update the right one.
  See [`CLAUDE.md`](./CLAUDE.md).

## Tests, coverage, and lint

The full suite must stay green before a PR can merge. CI runs these on every PR;
run them locally first.

```bash
# tests (same deselection as CI — integration markers need secrets)
.venv/bin/python -m pytest -q -m "not tinker and not supabase and not gpu"

# coverage for a module you changed (new modules target >= 90% line coverage)
.venv/bin/python -m coverage run --include='src/evsys_sdk/<module>.py' \
  -m pytest tests/test_<module>.py -q && .venv/bin/python -m coverage report -m

# lint + import-order (ruff is the formatter/linter of record)
.venv/bin/python -m ruff check .
```

Test markers (see `pyproject.toml`): `slow`, `tinker`, `supabase`, `gpu`.
Tests that hit external services must be marked so they can be deselected in CI.

## Commit messages

- Imperative subject, scoped: `fix(sdft): drop empty student rollouts`.
- One logical change per commit; include the tests in the same commit.
- If the change was co-authored with an AI assistant, end the message with the
  agreed `Co-Authored-By:` trailer.

## Pull requests

1. Push your branch and open a PR against `main`. Fill in the
   [PR template](./.github/PULL_REQUEST_TEMPLATE.md) — what changed, why, and
   how you tested it.
2. **CI must pass** (tests + lint).
3. **At least one approving review is required** before merge. Code owners
   (see [`.github/CODEOWNERS`](./.github/CODEOWNERS)) are requested
   automatically for the areas you touched.
4. Keep the PR focused — one migration / feature / fix per PR. Split unrelated
   changes.
5. Prefer **squash-merge** so `main` stays one-commit-per-change. Delete the
   branch after merge.

A reviewer should not approve their own PR; if you have write access and your PR
is approved + green, you may merge it yourself.

## Reporting issues

Open an issue using one of the templates:

- **🐛 Bug report** — a reproducible defect.
- **✨ Feature request** — a new capability or extension point.

For open-ended questions, use the linked discussion/contact channel rather than
an issue.
