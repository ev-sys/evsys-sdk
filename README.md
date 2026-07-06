<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/logo-dark.svg">
    <img src="assets/logo.svg" alt="EvSys" width="300">
  </picture>
</p>

<h3 align="center">
Infrastructure for thousands of task-specialised models that learn continuously from every interaction.
</h3>

<p align="center">
| <a href="https://ev-sys.github.io/evsys-sdk/"><b>Documentation</b></a> | <a href="https://ev-sys.github.io/evsys-sdk/docs/quickstart"><b>Quickstart</b></a> | <a href="https://ev-sys.github.io/evsys-sdk/docs/autoresearch"><b>Autoresearch</b></a> | <a href="https://ev-sys.github.io/evsys-sdk/docs/concepts/architecture"><b>Architecture</b></a> | <a href="https://ev-sys.github.io/evsys-sdk/whitepaper/"><b>Whitepaper - in depth</b></a> |
</p>

<p align="center">
  <a href="https://join.slack.com/t/evsys-community/shared_invite/zt-41tnfb6vb-qpVnWw59wP5LcViww_2A4Q">
    <img src="https://img.shields.io/badge/Slack-Join%20our%20community-4A154B?style=for-the-badge&logo=slack&logoColor=white" alt="Join our community on Slack">
  </a>
</p>

---

## About

**Easily create continually-learning models on your own data.**

**evsys-sdk turns your favourite coding agent into a model lab.** Every training
experiment is a single declarative YAML; a coding agent - Claude Code, Codex,
anything - writes a config, launches it on a Tinker-compatible backend, reads the
structured result, and writes the next one. Run a series of educated experiments
and you land a small, task-specific model that beats frontier on your task at a
fraction of the cost.

We believe the future of AI is not a handful of generalist models, but
**thousands of models adapted for every task, continually learning from every
interaction**. Realising that needs a new layer of infrastructure - the
*autoresearch* layer that lets a system discover what to learn, train on it, and
ship it. evsys-sdk is the first step: it **standardises and centralises** every
experiment so a coding agent always has the context of what's been tried, while
staying flexible enough to run **any algorithm on any data**.

evsys-sdk is **powerful** with:

- **One declarative artifact** - every experiment is a single `ExperimentConfig` (YAML); nothing hidden in scripts
- **SFT, RL, and self-distillation (SDFT)** behind one config shape and one runner
- **Autoresearch** - coding agents launch experiments, learn from the results, and write the next hypothesis
- **Multi-stage recipes (SFT → RL)** and **continual learning** - weights chain across stages, so models accumulate skills instead of restarting from base
- **In-loop validation + held-out benchmarks**, scored by verifiers and metrics you register
- **Agent harnesses** - train a model *inside* your own multi-turn tool-use harness, so it learns to use your tools
- **Structured per-run logging** - metrics, rollouts, predictions, hypothesis & conclusion on disk (plus an optional dashboard)

evsys-sdk is **flexible and easy to use** with:

- **Tinker-compatible backends** - train on anything that speaks the Tinker protocol: Tinker, Fireworks, TML, SkyRL - one line in the config switches provider; plus `local` (TRL + peft on your own GPU) and `mock` (tests)
- **Eight registries** - algorithms, backends, transforms, verifiers, metrics, data stores, log stores, inference clients - register your own with a one-line `@register_*` decorator, no library fork
- **A coding-agent plugin (Claude Code + Cursor)** - one shared `skills/` set drives the whole loop; install it into either agent from this repo
- **The `evsys` CLI** - `validate`, `run`, `list`, `schema`, `init-project`, `benchmark`, `eval`
- **LoRA on any Hugging Face model**
- **`--dry` runs** - a few steps per stage with rollout logging, to eyeball the data + rollouts before a full run

## Getting Started

Requires **Python 3.12+** and [`uv`](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/ev-sys/evsys-sdk.git
cd evsys-sdk
uv sync
source .venv/bin/activate
```

Run the no-GPU, no-network hello world:

```bash
python examples/01_local_mock_sft.py
```

For real training, export your backend key and drive it from a config:

```bash
export TINKER_API_KEY=...
evsys run config.yaml          # validate, expand, train, score, record
evsys run config.yaml --dry    # a few steps per stage, with rollout logging
```

Discover what's available and inspect any extension's params:

```bash
evsys list algorithms          # sft · rl · sdft · ...
evsys schema algorithm sft
```

Visit the [documentation](https://ev-sys.github.io/evsys-sdk/) to learn more:

- [Installation](https://ev-sys.github.io/evsys-sdk/docs/installation)
- [Quickstart](https://ev-sys.github.io/evsys-sdk/docs/quickstart)
- [Putting it all together - Autoresearch](https://ev-sys.github.io/evsys-sdk/docs/autoresearch)
- [Concepts & the eight registries](https://ev-sys.github.io/evsys-sdk/docs/concepts/architecture)

## Use it as a coding-agent plugin

The SDK ships a single set of **skills** (in `skills/`) that drive the
autoresearch loop: read the full history of past experiments (hypotheses,
conclusions, metrics), decide the next educated experiment, scaffold the config
plus any custom verifier / metric / transform, launch it, and write back a
conclusion. There are no subagents — everything is skills, so the same source
works in both Claude Code and Cursor.

**Claude Code.** Point Claude at the repo directly, or add it as a plugin
marketplace:

```bash
claude --plugin-dir /path/to/evsys-sdk           # load in place from a checkout
# or, as a marketplace:
/plugin marketplace add ev-sys/evsys-sdk
/plugin install evsys-sdk@evsys-sdk
```

**Cursor.** The same repo is a Cursor plugin (`.cursor-plugin/`). Add it from
Customize → Plugins (import the git repo as a marketplace), or drop the skills
into a skills directory Cursor scans. Cursor also reads Claude's skill
directories, so `.claude/skills/` and `~/.claude/skills/` are discovered too.

Both wrappers point at the same `skills/` directory — edit a skill once and both
agents pick it up.

## Contributing

We welcome contributions and collaborations. The dev workflow:

```bash
uv sync
.venv/bin/python -m pytest tests/ -q     # the full suite must stay green
.venv/bin/python -m ruff check .
```

See [`CLAUDE.md`](CLAUDE.md) for repo conventions - the registry + `{kind, params}`
extension pattern and the two skills directories - and [`docs/DESIGN.md`](docs/DESIGN.md)
for the layout + protocol rationale.

## Contact

- **Docs & guides:** https://ev-sys.github.io/evsys-sdk/
- **Issues & feature requests:** [GitHub Issues](https://github.com/ev-sys/evsys-sdk/issues)
