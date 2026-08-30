---
name: draft-paper
description: >
  Draft a short (~4 page) NeurIPS-format research paper from an EvolvingSystems
  research writeup, blog post, or experiment reports. Use when the user says
  /draft-paper, asks to convert research into NeurIPS format, or wants a paper
  draft from composio-bench / research blog notes.
disable-model-invocation: false
---

# /draft-paper

Turn research notes (blog post, experiment `REPORT.md`, teammate writeup) into a
**short NeurIPS-format paper draft** (~4 content pages). Do **not** compile PDF
unless the user explicitly asks.

## Inputs the user may point at

- A research blog path (e.g. `frontend` → `/research/<slug>`, often on `main`)
- Experiment dirs under `composio-bench/experiments/<run>/` (`REPORT.md`,
  `*_report.md`, matrices, configs)
- An existing shell: `papers/<topic>/draft.tex` + `neurips_2026.sty`

If the writeup lives only on a git branch, `git show origin/main:<path>` (or the
branch they name) and work from that content.

## Output layout

```
papers/<topic>/
├── draft.tex              # NeurIPS 2026 paper body
├── references.bib
├── neurips_2026.sty       # download if missing
└── figures/
    └── *.png              # PNG only — never PDF figures
```

Default topic folder from the subject (e.g. multi-teacher →
`papers/multi-teacher/`). Reuse an existing `draft.tex` shell if present;
replace the template boilerplate with the real paper.

## Workflow

```text
Task progress:
- [ ] 1. Read source writeup + experiment numbers
- [ ] 2. Ensure papers/<topic>/ + neurips_2026.sty
- [ ] 3. Write draft.tex (~4 pages)
- [ ] 4. Put full result tables in the paper body
- [ ] 5. Generate aesthetic PNG figures under figures/
- [ ] 6. Strip em-dashes; use plain axis labels
- [ ] 7. Stop (no compile unless asked)
```

### 1. Gather facts

Pull title, problem, recipes/methods, metrics, and **exact numbers** from the
source. Prefer experiment reports over paraphrased blog prose when they
disagree. Keep anonymous authors for double-blind NeurIPS submission unless the
user supplies names.

### 2. Style file

If `neurips_2026.sty` is missing next to `draft.tex`, fetch the official NeurIPS
2026 style (same package the template uses: `\usepackage{neurips_2026}`). Do not
tweak geometry or font sizes.

### 3. Paper shape (~4 pages)

Typical sections:

1. Abstract (one paragraph)
2. Introduction + short contributions
3. Method (recipes / algorithm)
4. Experiments + results (figures + **full tables**)
5. Conclusion (limitations in 1–2 sentences)
6. References (do not count toward page budget)

Omit NeurIPS checklist / long appendix unless asked. Keep `\ack` (hidden under
anonymous submission).

### 4. Tables (required)

Put the **full result tables in the paper body**, not only a tiny summary and
not appendix-only. Include every recipe's forgetting matrix (after each stage ×
eval splits) when the source has them, plus a short after-final-stage summary
table. Metric names like `val_s0` are fine **inside tables**; keep them
consistent with the source reports.

### 5. Figures (PNG only)

- Directory: `papers/<topic>/figures/`
- Format: **PNG only**. Delete any `.pdf` figure siblings.
- Reference in TeX as `figures/<name>.png`.
- Graphs must be **aesthetic and information-dense**: value labels on points/bars,
  clear legend, shared colormaps for heatmaps, annotations for the key takeaway
  when helpful.
- Prefer generating with `uv run --with matplotlib python ...`.

#### Axis / legend wording (strict)

Do **not** put underscore metric keys on graph axes or legends
(e.g. avoid `val_stage0`, `full_stage0`, `val_stage2`).

Use short plain labels, for example:

- Y-axis: `Val accuracy (%)`
- Legend: `Stage 0 (val)`, `Stage 0 (full)`, `Stage 2 (val)`
- Heatmap axes: `Stage 0` / `After 0`, colorbar: `Val accuracy (%)`

Captions may briefly clarify splits in words; axes stay short.

### 6. Prose rules (strict)

- **No em-dashes.** Do not write `---` in TeX body text. Prefer commas,
  parentheses, colons, or separate sentences.
- Academic but readable; match the source claims; do not invent numbers.
- Citations via `references.bib` + `\citep{...}`.

### 7. Compile

**Do not compile** (`pdflatex` / Docker / tectonic) unless the user asks.
Deliver `draft.tex` + figures + bib. If they later ask to compile, target
~4 content pages and trim if over.

## Minimal `draft.tex` preamble pattern

```latex
\documentclass{article}
\usepackage{neurips_2026}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{hyperref,url,booktabs,amsfonts,amsmath,amssymb,microtype,graphicx}
\title{...}
\author{Anonymous Authors \\ ...}
\begin{document}
\maketitle
\begin{abstract}...\end{abstract}
% body ...
{\small\bibliographystyle{plainnat}\bibliography{references}}
\end{document}
```

## Done when

- [ ] `draft.tex` is a real paper (not NeurIPS formatting boilerplate)
- [ ] Full result tables are in the body
- [ ] Figures are PNG-only under `figures/` with plain axis labels
- [ ] No em-dashes in the draft
- [ ] No compile unless requested
