# trajectory-experiments

Declarative, modular experiment framework for LLM training. Built around a single YAML, with pluggable algorithms / verifiers / metrics / data stores / log stores / backends. Runs locally on TRL or remotely on Tinker.

```bash
uv pip install -e .[tinker,local,tensorboard]
trajex validate config.yaml --deep
trajex run config.yaml
```

See `docs/cookbook.md` for end-to-end walkthroughs and `examples/` for ready-to-run YAML + Python scripts.

## Highlights

- **Single YAML drives everything**: `trajex run experiments/composio_sft.yaml`.
- **Modular by design**: add a new algorithm with `@register_algorithm("dpo")` — no library fork required.
- **Backends**: `mock` (tests), `local` (TRL+peft on your GPU), `tinker` (Tinker hosted).
- **Stores**: `LocalDataStore`, `JSONLLogStore`, `TensorBoardLogStore`, `MultiplexLogStore` out of the box.
- **Matrix expansion**: declare axes once, get a campaign of N runs without copy-paste.

## Status

`v0.1.0` — first cut. Stable: protocols, registry, YAML loader, mock backend, Tinker SFT/RL adapters. In progress: Supabase adapters, evolution-loop port. See `docs/DESIGN.md`.
