<!-- Title: <type>(<scope>): <imperative summary>  e.g. fix(sdft): drop empty student rollouts -->

## What & why

<!-- What does this change do, and why? Link the motivating issue. -->
Closes #

## Type of change

- [ ] feat — new feature/capability
- [ ] fix — bug fix
- [ ] refactor — internal change, no behavior difference
- [ ] docs — docs / skills / comments only
- [ ] chore — tooling, deps, CI

## How tested

<!-- Commands run + result. Paste the relevant pytest summary line. -->
```
.venv/bin/python -m pytest -q
```

## Checklist

- [ ] Tests added/updated and the full suite is green locally
- [ ] New modules covered to >= 90% (`coverage report`)
- [ ] `ruff check .` is clean
- [ ] New extension *kinds*/built-ins follow the registry + `Config` + `{kind, params}` pattern (see CLAUDE.md)
- [ ] Docs / skills updated if behavior or surface changed (the right skills dir)
- [ ] One logical change; unrelated work split into separate PRs
