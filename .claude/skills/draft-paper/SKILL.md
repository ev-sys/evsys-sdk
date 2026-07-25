---
name: draft-paper
description: >
  Draft a short (~4 page) NeurIPS-format research paper from an EvolvingSystems
  research writeup, blog post, or experiment reports. Use when the user says
  /draft-paper, asks to convert research into NeurIPS format, or wants a paper
  draft from composio-bench / research blog notes.
---

# /draft-paper

Full workflow: [../../skills/draft-paper/SKILL.md](../../skills/draft-paper/SKILL.md)

Hard rules (see full skill for details):

- Output under `papers/<topic>/` with `draft.tex`, `references.bib`, `figures/*.png`
- ~4 pages; NeurIPS 2026 style (`neurips_2026.sty`)
- Full result tables in the paper body (not appendix-only)
- PNG figures only; no underscore metric names on graph axes (use `Val accuracy (%)`)
- No em-dashes (`---` in prose)
- Do not compile PDF unless the user asks
