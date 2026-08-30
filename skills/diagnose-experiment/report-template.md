# Experiment diagnosis — `<experiment_name>`

Generated: `<date>`
Config: `<path/to/config.yaml>`
Run status: `<completed|failed|partial>`
Final checkpoint: `<uri or path>`

## Executive summary

`<2–4 sentences: is the experiment trustworthy? top issue? top fix?>`

---

## 1. Training input pipeline

### 1.1 Config surfaces

| Surface | Value |
|---------|-------|
| Data path | |
| Transforms | |
| `user_template` | |
| `demo_template` | |
| `system_prompt` | |
| `renderer_name` | |
| `enable_thinking` | |
| Algorithm | |

### 1.2 Sample rendered inputs (post-transform)

#### Row A — `<task_id or tool_slug>`

**Raw row:**
```json

```

**After transforms:**
```json

```

**Student / anchor prompt (decoded):**
```text

```

**Teacher prompt (decoded, SDFT only):**
```text

```

#### Row B — …

### 1.3 Training dynamics

| Stage | steps | loss start | loss end | avg completion tokens |
|-------|-------|------------|----------|----------------------|
| | | | | |

**Findings:** …

---

## 2. Ground truth

| Row | raw `expected` | training target | in train slugs? |
|-----|----------------|-----------------|-----------------|
| | | | |

### Tool coverage (benchmark vs training)

| toolkit | benchmark tools | train tools | intersection | bench \\ train |
|---------|-----------------|-------------|--------------|---------------|
| | | | | |

**Findings:** …

---

## 3. Train vs eval prompt parity

| Surface | system_prompt | user content (example) | enable_thinking | renderer |
|---------|---------------|------------------------|-----------------|----------|
| Train (student) | | | | |
| Train (anchor) | | | | |
| Eval | | | | |

**Match?** `<yes / no — explain>`

**Findings:** …

---

## 4. Final model results

Eval set: `<benchmark name>` | Checkpoint: `<stage/run>`

| Metric | Value |
|--------|-------|
| pass@1 | |
| n_tasks | |

### Successes (sample)

- **`<task_id>`** — instruction: … | expected: `…` | got: `…`

### Failures (sample)

- **`<task_id>`** — instruction: … | expected: `…` | got: `…`

### Failure breakdown

| Pattern | count |
|---------|-------|
| wrong_slug | |
| no_answer_tag | |
| uncovered_tool | |
| other | |

**Findings:** …

---

## 5. Issues and recommendations

| Severity | Issue | Evidence | Fix |
|----------|-------|----------|-----|
| CRITICAL | | | |
| WARNING | | | |
| INFO | | | |

### Recommended next steps

1. …
2. …
3. …
