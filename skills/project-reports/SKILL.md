---
name: project-reports
description: >
  Push and browse project HTML reports on the EvolvingSystems dashboard via
  the evsys CLI / EvsysStore. Use when an agent (or researcher) needs to write
  or upload a static HTML report, name it, inspect the project's report
  filesystem tree, pick a path, or share a viewer URL with a logged-in project
  member. Reports are written in third person; names are a readable date
  followed by the title.
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
experiments/<exp-slug>/<YYYY-MM-DD> <short-name>
benchmarks/<name>/<YYYY-MM-DD> latest
notes/<YYYY-MM-DD> <topic>
```

## Writing the report

- Write in **third person**. The report is a record of the work, not a diary.
  Prefer "The run reached 0.81 on the held-out split" over "I ran an eval and
  we got 0.81." Do not address the reader as "you" in the findings.
- The **report name** (the leaf of `--path`, or `--name`) is a **readable
  calendar date**, then a space, then a short title. Use `YYYY-MM-DD`, not a
  compact stamp like `20260520` and not a time-of-day.

```text
2026-08-17 tool-search eval
2026-08-17 composio public tool search
```

```bash
evsys report push ./out/eval-report \
  --path "experiments/sft/2026-08-17 tool-search eval"
# or: --path experiments/sft --name "2026-08-17 tool-search eval"
```

## Structure and content

Write for a reader who knows the project's goal but has **not** seen the code,
the run scripts, or where any data lives. The report answers "what happened and
what it means," not "how it was produced."

Structure every report in this order:

1. **Title + one-sentence framing.** What was done and why it matters, in a
   single line under the title.
2. **TL;DR.** Always lead with a short bulleted summary (3–6 bullets) that
   stands on its own: the headline result, the key numbers, and the main
   takeaway. A reader should be able to stop after the TL;DR and still have the
   gist.
3. **Body.** The detail behind the TL;DR — comparisons, tables, per-run
   findings, analysis of why things happened, and concrete recommendations.
   Prefer tables and short bulleted findings over long prose.

Content rules:

- **Do the analysis, don't just tally.** Explain *why* a result happened and
  what could be done better, not only *what* the numbers were.
- **Quote primary material directly** (model outputs, logs, generations) when it
  makes a point sharper. Present quotes as content; do not caption them with
  meta-notes (see "Do not").
- **No implementation detail unless the user asks for it.** Leave out file
  paths, script names, function/verifier internals, where transcripts or
  conversations were read from, tool-call plumbing, and how the report itself
  was assembled. Describe behavior and outcomes in domain terms a
  non-code-reader understands.

## Push a report

Local directory **must** contain `index.html` (or pass `--entry`):

```bash
evsys report push ./out/eval-report \
  --path "experiments/sft/2026-08-17 tool-search eval"
```

- Leaf of `--path` is the report name (`2026-08-17 tool-search eval` above).
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
2. Choose a `--path` whose **leaf name** is `YYYY-MM-DD` plus a short title
   (create parents by including them in the path; or create an empty folder
   in the UI).
3. Write the HTML in **third person**, structured as title → TL;DR → body
   (see "Structure and content"). Build a local dir with `index.html` (+ assets
   with relative links).
4. `evsys report push <dir> --path <virtual/path>`.
5. Surface the returned `url` to the user / in experiment notes.

## Do not

- Do not write the report in first person ("I", "we") or as a how-to addressed
  to the reader.
- Do not skip the TL;DR — every report leads with one.
- Do not write meta-captions about the report itself. Never include lines like
  "Third-person summary", "Quoted text is taken verbatim…", "This report was
  generated by…", or a description of the methodology/sources used to write it.
  Just present the findings; let the content speak.
- Do not include implementation detail (file paths, script/function names,
  verifier internals, transcript locations, tool-call mechanics) unless the user
  explicitly asks for it.
- Do not name a report with a compact stamp (`20260520_eval`) or a time of day.
  The name is `YYYY-MM-DD` plus a short title.
- Do not upload secrets inside the HTML/assets (any project member can view).
- Do not treat the URL as public — it is membership-gated.
- Do not expect multipart push through `EvsysStore._call`; push uses
  `POST /api/dashboard/api/sdk/reports/push/` (the CLI wraps this). Tree/list
  go through the normal SDK data gateway.
