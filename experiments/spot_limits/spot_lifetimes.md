# Spot lifetime ledger (passive preemption telemetry)

Every launch appends a row; every ending appends the outcome. A box that
vanishes without an operator reap is a PREEMPTION and its lifetime is
(last heartbeat - boot). Collected passively during normal benchmark work, so
box-size vs preemption-rate accumulates for free across campaigns.

| date | sku | gpus | region | lifetime_min | outcome |
|---|---|---|---|---|---|
| 2026-08-04 | 1A100.22V | 1 | FIN-03 | ~9 | PREEMPTED (mid-sweep, only true preemption of session) |
| 2026-08-04 | 1L40S.20V | 1 | FIN-02 | 0 | never booted (provisioning fail or instant reclaim) |
| 2026-08-04 | 22 boxes: 2H100/2H200/1H100/1H200/1A100 | 1-2 | FIN-01/02/03 | 5-50 each | operator reap (all survived their window) |

Historical (SSH-era campaign, lifetimes not preserved): 12 launches OK,
11 preemptions, 6 refused, 1 DOA over multi-hour boxes.

Open question this ledger exists to answer: does GPUs-per-node correlate
with preemption rate? No conclusive data yet (n=1 true preemption).
