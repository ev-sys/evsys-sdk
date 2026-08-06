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

## Onboarding a compute provider (router/queue)

Five registration points, all in `src/evsys_sdk/compute/` — mirror the
`nebius` files, which were built purely from docs as the reference for this
checklist (`verda` is the battle-tested original):

1. `providers_<name>.py` — credentials + auth + a `_call` REST transport
   (injected everywhere, so tests never need an account).
2. An `@register_availability("<name>")` probe in `availability.py`
   returning `Capacity` rows (state, region, sku, usd_hr, spot). This alone
   puts the provider into every `scan()`.
3. `provisioner_<name>.py` — `@register_provisioner("<name>")` class with
   `name`/`Config` ClassVars implementing provision / alive / terminate
   (`terminate(keep_volume=True)` must preserve the checkpoint disk).
4. Declare storage semantics in `portability.py::_STORAGE_CAPS` (or
   `register_storage_caps`). Providers without persistent storage never get
   checkpointed jobs.
5. Credential shape in `credentials.py::PROVIDERS` + side-effect import in
   `compute/__init__.py`.

`evsys queue run` then picks it up automatically: every registered
provisioner with saved credentials joins the router's pool.
