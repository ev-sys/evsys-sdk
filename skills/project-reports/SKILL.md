---
name: project-reports
description: >
  Push and browse project HTML reports on the EvolvingSystems dashboard via
  the evsys CLI / EvsysStore. Use when an agent (or researcher) needs to upload
  a static HTML report folder, inspect the project's report filesystem tree,
  pick a path, or share a viewer URL with a logged-in project member.
---

# Project HTML reports

Reports are **static site folders** (`index.html` + CSS/JS/images) stored in a
project-scoped virtual filesystem (folders + report leaves). The UI owns folder
create/rename/move/delete; the CLI pushes one report into a path (creating
missing parent folders).

## Env

```bash
export EVSYS_API_URL="https://<backend>"     # or http://localhost:8000
export EVSYS_API_KEY="sk_..."                # dashboard → Settings → API keys
export EVSYS_PROJECT_ID="<project-uuid>"
```

The API key user must be a **member** of that project.

## Inspect structure first (agents)

Before pushing, list what already exists so you place the report correctly:

```bash
# Nested tree (folders + reports). Each report has path + url.
evsys report tree

# Flat list of reports only — easiest for picking a path / sharing a link.
evsys report list
```

Python equivalent:

```python
from evsys_sdk import EvsysStore
store = EvsysStore()
tree = store.list_report_tree()   # nested
reports = store.list_reports()    # [{id, path, url, entry_file, ...}]
```

Typical path conventions (not enforced — any segments are allowed):

```text
experiments/<exp-slug>/<run-or-eval-name>
benchmarks/<name>/latest
notes/<topic>
```

## Push a report

Local directory **must** contain `index.html` (or pass `--entry`):

```bash
evsys report push ./out/eval-report \
  --path experiments/20260520_sft/eval
```

- Leaf of `--path` is the report name (`eval` above).
- Missing parent folders are created.
- Re-pushing the same path **replaces** that report’s files (siblings stay).

Optional flags:

| Flag | Meaning |
|------|---------|
| `--name NAME` | Report name; then `--path` is parent folders only |
| `--entry FILE` | Entry HTML inside the dir (default `index.html`) |
| `--project-id` | Override `EVSYS_PROJECT_ID` |

JSON stdout includes `report_id`, `path`, `content_hash`, `n_files`, and
**`url`** — a dashboard link a **logged-in project member** can open to view
the report:

```text
https://<frontend>/projects/<project_id>/reports?report=<report_id>
```

## Viewer URL

- Returned on every successful `push` and on `list` / `get_report` / tree leaves.
- Requires the opener to be signed into the dashboard **and** a member of the
  project (not a public/anonymous link).
- Deep-link selects that report in the project Reports UI.

```python
r = store.get_report(report_id)
print(r["url"])
```

## Agent checklist

1. `evsys report list` (or `tree`) — see existing paths.
2. Choose a clear `--path` under the right folder (create parents by including
   them in the path; or create an empty folder in the UI).
3. Build a local dir with `index.html` (+ assets with relative links).
4. `evsys report push <dir> --path <virtual/path>`.
5. Surface the returned `url` to the user / in experiment notes.

## Do not

- Do not upload secrets inside the HTML/assets (any project member can view).
- Do not treat the URL as public — it is membership-gated.
- Do not expect multipart push through `EvsysStore._call`; push uses
  `POST /api/dashboard/api/sdk/reports/push/` (the CLI wraps this). Tree/list
  go through the normal SDK data gateway.
