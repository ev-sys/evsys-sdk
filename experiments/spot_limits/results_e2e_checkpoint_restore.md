# E2E checkpoint/restore test — real Verda infrastructure (2026-08-05)

Ran the portable-checkpointed-jobs layer live on two Verda CPU spot nodes
($0.014/hr each) with a 50GB NVMe persistent volume, using the actual SDK
modules (`checkpoint_delta`, `checkpoint_store`) shipped to the nodes.

## Leg 1 — cross-node network stream (the cross-provider transfer path)
Node A wrote base + 2 XOR-delta checkpoints to its attached volume via
`DeltaCheckpointer` + `LocalDirStore`, published a manifest (keys + sha256),
and served the store over HTTP. Node B ran `stream_copy` against it through
an `HttpReadStore` adapter:

    B:NET_STREAM {"pass": true, "copied": ["base.evd","step-100.evd","step-200.evd"],
                  "bytes": 1314190, "seconds": 0.02, "digests_verified": true,
                  "reconstructed_w1_sum": 8.847647666931152, "expected": 8.847647666931152}

Byte-exact reconstruction after network transfer with digest verification.
The transport is provider-agnostic — a Prime/other-cloud node running the
same agent is the identical flow (Prime pods API access confirmed, 200).

## Leg 2 — node death, volume survives, restart on another node
Node A was deleted (the "preemption"). Its data volume auto-detached and
survived. Attached to node B (shutdown -> attach -> boot), whose reboot
agent mounted it read-only and verified:

    B:VOL_RESTORE {"pass": true, "device": "/dev/vdb", "digests_match": true,
                   "reconstructed_w1_sum": 8.847647666931152, "resume_step": 200}

## Operational facts learned (encoded for the router)
- Verda `PUT /volumes {action: attach}` works but requires the target
  instance to be SHUT DOWN ("Instance should be shutdown"); boot after.
- Startup scripts are FIRST-BOOT ONLY. Any node participating in the
  re-attach flow needs a reboot-persistent agent (an `@reboot` cron written
  by the first-boot script worked).
- Deleting an instance auto-detaches its extra volumes; they survive (and
  bill) — consistent with the OS-volume orphan behavior.
- `mkfs` must be conditional (mount first; format only if mount fails) or a
  re-attach destroys the checkpoints it came for.
- CPU spot nodes are ideal harnesses: the whole two-node test cost ~$0.10.

Total test cost: ~$0.15 including volume-hours and iterations.
