# AGENTS.md

See `CLAUDE.md` for repo conventions (the registry + `{kind, params}` extension
pattern, the two skills directories) and `README.md` / `docs/DESIGN.md` for the
architecture. This file only captures cloud-agent operating notes.

## Cursor Cloud specific instructions

`evsys-sdk` is a Python 3.12+ SDK + `evsys` CLI managed by `uv` (`.venv` in the
repo root). The core SDK, CLI, and full test suite run fully offline with the
`mock` backend — no external services, GPU, or API keys are required.

Non-obvious notes:

- **`uv sync` alone does NOT install the test/lint tools.** `pytest`, `ruff`, and
  `pytest-asyncio` live in the `dev` *extra* (`[project.optional-dependencies]`),
  not a default dependency group. Use `uv sync --extra dev` to get them (the
  startup update script already does this).
- **Standard commands** are in `CLAUDE.md` / `README.md`: tests
  `.venv/bin/python -m pytest tests/ -q`, lint `.venv/bin/python -m ruff check .`,
  offline hello-world `python examples/01_local_mock_sft.py`. The CLI supports
  `evsys validate <yaml> --deep`, `evsys run <yaml> [-o out.json]`, `evsys list`,
  `evsys schema <kind> <name>`.
- **Skipped tests are expected, not failures.** ~23 tests skip on a clean setup
  because they need optional extras (`tinker`, `anthropic`/`frontier`) or CUDA
  (`gpu`). Install the matching extra + set the relevant API key only if you need
  those paths.
- **The real `sft`/`rl`/`sdft` algorithms and `tinker`/`local` backends register
  only when their optional extra is installed.** With just `dev`, `evsys list`
  shows `mock_sft`, `mock_rl`, `combo`, `gepa_prompt` and the `mock` backend —
  use these for offline smoke tests (e.g. `evsys schema algorithm mock_sft`).
- **`ruff check .` is not clean on a fresh checkout.** The lockfile pins a recent
  ruff (0.15.x) whose newer rules flag many pre-existing violations across the
  repo (mostly in `tests/`). This is unrelated to any single change; don't treat
  a non-zero `ruff check` on unrelated files as something you introduced.
- Dashboard/Supabase, Tinker, and frontier-model APIs are optional and degrade
  gracefully (local offline mirror) when absent; set `EVSYS_OFFLINE=true` to skip
  dashboard calls entirely.
