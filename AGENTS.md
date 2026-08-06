# Agent instructions for evsys-sdk

Repo conventions for coding agents. Full project conventions (skills layout,
registry pattern, dev workflow) live in [CLAUDE.md](CLAUDE.md) — read that
too; this file covers the git/branching policy.

## Branching: PRs are made off `dev`, not off `main`

* `dev` is the integration branch. **Base every feature branch on `dev` and
  open PRs against `dev`** — never against `main`.
* `main` is the release branch; it only advances by merging `dev` (release
  flow), not by direct feature PRs.
* If your branch was accidentally forked from `main`, merge `origin/dev` into
  it (or rebase onto `origin/dev`) before opening the PR, so the PR diff
  contains only your changes.
* Keep the full test suite green before landing anything on `dev`:
  `.venv/bin/python -m pytest tests/ -q` (run `uv sync` after pulling; if the
  sync pruned test deps, restore them with
  `uv pip install --python .venv/bin/python pytest numpy coverage`).
* Never commit incidental `uv.lock` churn — if the lock file changed only
  because of tool-version noise, `git checkout -- uv.lock` before committing.
