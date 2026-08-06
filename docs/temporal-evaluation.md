# Temporal for the job router — evaluated, deferred

**Question:** [Temporal](https://github.com/temporalio/temporal) (MIT, open
source) provides durable workflow execution — automatic retries, timers,
heartbeats, signals, replayable history. Our `JobRouter` hand-rolls a subset
(durable JSONL state, reconcile loop, attempt counting). Should Temporal own
this?

**The fit is real.** The job lifecycle maps cleanly onto Temporal primitives:
one workflow per job; `provision`/`alive`/`terminate` become activities with
retry policies; preemption detection becomes activity heartbeat timeouts;
agent checkpoint reports become signals; the placement poll becomes a workflow
timer. Retries, backoff, and "the process died mid-launch" semantics come for
free, with a UI showing every job's full history.

**Why not now — three concrete reasons:**

1. **Infrastructure floor.** Temporal requires a running server plus a
   persistence store (Postgres/MySQL/Cassandra) plus worker processes. Our
   entire router runs on a $0.01/hr CPU node with zero dependencies and
   file-grade durability; state is greppable JSONL, the same crash-model as
   the queue. At the current scale (one team, tens of jobs) the operational
   surface of a Temporal deployment exceeds the code it would delete.
2. **It deletes the easy 20%.** The loop/retry scaffolding Temporal replaces
   is ~250 tested lines. The hard 80% — the Verda shutdown→attach→boot dance,
   storage-aware placement, XOR-delta streaming, volume reuse — is domain
   logic Temporal does not provide and which stays identical under either
   orchestrator.
3. **Determinism constraints.** Workflow code must be deterministic
   (replay-safe); our placement logic reads live availability, which pushes
   most of the router into activities anyway — at which point Temporal is
   orchestrating three function calls.

**When to revisit — any of:** multiple teams sharing the router; hundreds of
concurrent jobs; audit/UI requirements on job history; human-in-the-loop
approvals (Temporal signals are the right tool); or a second long-running
saga (e.g. multi-stage continual-learning pipelines) appearing beside job
placement.

**Port path (kept deliberately cheap):** the router already isolates every
effect behind the `Provisioner` protocol and consumes events via an injected
callable. A Temporal port is: workflow = `JobRouter.tick` logic, activities =
the `Provisioner` methods, signals = the events channel. No domain code
changes.
