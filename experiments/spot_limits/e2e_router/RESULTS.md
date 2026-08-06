# Live E2E: experiment YAML -> queue -> router -> SkyRL train -> preemption -> auto-resume (2026-08-06)

Ran the ACTUAL production path on real Verda H100 nodes, Qwen3-4B-Instruct-2507,
SkyRL tinker server (megatron), driven end-to-end by JobRouter + VerdaProvisioner
+ TopicEvents + the delta_snapshot callback. Nothing manual after `submit`.

## Run 1 — happy path (job 3eabdba39e7b)
- router placed on verda H100, fresh volume; agent synced SkyRL (80s), server
  healthy, `evsys run config.yaml` (backend: skyrl) trained 30 SFT steps.
- checkpoints at steps 8/16/24/final -> tinker:// state on the volume
  (server checkpoints_base + sqlite DB both on /data) -> events -> CheckpointMap.
- `done` event: router terminated the node itself, kept the volume. rc=0.

## Run 2 — preemption + auto-resume (job 31f973ef21d9)
- placed fresh; trained to step 16-24; node KILLED mid-run (injected preemption).
- router (same tick): "will resume from step 15 (local_dir:verda/7bdf3b35...)"
  -> requeued -> re-placed: **"reuse volume 7bdf3b35 (zero-copy restart)"**.
- new node mounted the surviving volume; `evsys run` found the manifest:
      runner: auto-resume from tinker://model_236b3217/weights/step_24
      TinkerBackend: resumed (with optimizer) from .../step_24
  trained to completion: status "completed". Router marked DONE, tore down.

## Fixes the E2E forced (all committed)
- BackendConfig.kind literal gains "skyrl"; skyrl backend + protocol guard
  ported from feat/skyrl-backend.
- runner auto-resume: restarted run_dir picks up the last manifest row.
- save_sampler knob + graceful degrade: sampler export needs an inference
  role a 1-GPU self-hosted server doesn't have; training-state saves are
  what resume needs and are unaffected.
- SDK delivered to nodes via ntfy attachment URL (startup scripts cap ~<340KB).
- Known-good pattern: server --checkpoints-base and sqlite DB ON the volume
  (symlink /tmp/skyrl_checkpoints -> /data/...; SKYRL_DATABASE_URL) makes
  tinker:// checkpoints preemption-portable with zero copying.

Resume semantics v1: weights+optimizer continue from the checkpoint; the step
loop replays its schedule (start_step not fast-forwarded). Data-cursor
fast-forward is the known follow-up.

Cost of the whole live E2E incl. 3 debug cycles: ~$1.7 (H100 on-demand $3.25/hr,
node lifetimes 5-15 min each).
