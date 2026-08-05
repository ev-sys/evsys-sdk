"""Snapshot cadence, resume detection and the tenant manifest.

Preemption arrives with no warning — SkyPilot polls for it every 15s and
consumes no provider interruption notice — so there is no "flush on notice"
path to test. What there is: how often to snapshot, which checkpoint to come
back from, and how to remember which adapter was which. All three are pure
functions of data someone else fetched, which is why they can be tested at
all.
"""

from __future__ import annotations

import math

import pytest

from evsys_sdk.compute.snapshot import (
    ManifestCorrupt,
    ResumeManifest,
    SnapshotPolicy,
    SnapshotScheduler,
    TenantState,
    checkpoint_id_for_step,
    latest_manifest,
    latest_resumable,
    manifest_generation,
    manifest_name,
    parse_checkpoints,
    resume_target,
    step_of,
)

HOUR = 3600.0


class _Clock:
    """A clock the test drives, because a scheduler that reads the wall clock
    can only be tested by sleeping."""

    def __init__(self, t: float = 0.0):
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


def _listing(*entries):
    """The shape `GET /training_runs/{id}/checkpoints` actually returns."""
    return {"checkpoints": [
        {"checkpoint_id": cid, "checkpoint_type": ctype,
         "tinker_path": f"tinker://model_a/{'weights' if ctype == 'training' else 'sampler_weights'}/{cid}",
         "time": t}
        for cid, ctype, t in entries]}


class TestCheckpointIds:
    def test_step_round_trips(self):
        assert step_of(checkpoint_id_for_step(4200)) == 4200

    def test_ids_are_legal_skyrl_checkpoint_ids(self):
        """SkyRL constrains checkpoint ids to ^[a-zA-Z0-9_-]+$ (api.py:53) and
        rejects the request outright otherwise — a naming scheme with a slash
        or a colon in it fails at save time, not at resume time."""
        import re
        assert re.fullmatch(r"[a-zA-Z0-9_-]+", checkpoint_id_for_step(7))

    def test_lexical_order_matches_numeric_order(self):
        """The listing comes back unordered and its only timestamp is a wall
        clock on a machine that has just been replaced. Zero-padding means a
        reader never has to trust that clock."""
        ids = [checkpoint_id_for_step(s) for s in (2, 11, 100, 1000)]
        assert sorted(ids) == ids

    def test_foreign_ids_have_no_step(self):
        assert step_of("ss0_seq3") is None
        assert step_of("") is None

    def test_negative_steps_are_rejected(self):
        with pytest.raises(ValueError):
            checkpoint_id_for_step(-1)


class TestCadence:
    def test_optimum_is_young_daly(self):
        p = SnapshotPolicy(snapshot_cost_s=20, mtbf_s=2 * HOUR, max_loss_s=None)
        assert p.optimal_interval_s == pytest.approx(math.sqrt(2 * 20 * 2 * HOUR))

    def test_the_hourly_rate_does_not_change_the_interval(self):
        """Both terms of the overhead are pure time, so $/hr cancels out of
        the derivative entirely. The rate decides whether to be on spot at
        all — snapshotting more often because the GPU is expensive buys
        overhead and nothing else."""
        p = SnapshotPolicy(snapshot_cost_s=20, mtbf_s=2 * HOUR, max_loss_s=None)
        cheap, dear = p.effective_hourly_cost(0.63), p.effective_hourly_cost(1.40)
        assert dear / cheap == pytest.approx(1.40 / 0.63)
        assert p.interval_s == p.interval_s   # unchanged by either

    def test_restart_latency_does_not_move_the_optimum(self):
        """R/M is paid per preemption whatever T is, so it cannot appear in
        the derivative — it just dominates the total. Confusing the two leads
        to snapshotting furiously to cover a slow respawn."""
        a = SnapshotPolicy(snapshot_cost_s=20, mtbf_s=2 * HOUR, max_loss_s=None)
        b = SnapshotPolicy(snapshot_cost_s=20, mtbf_s=2 * HOUR, max_loss_s=None,
                           restart_s=1800)
        assert a.interval_s == b.interval_s
        assert b.waste_fraction() > a.waste_fraction()

    def test_max_loss_clamps_the_optimum(self):
        """'At most a few minutes' is a requirement. It outranks the optimum,
        which at a long MTBF would happily choose half an hour."""
        p = SnapshotPolicy(snapshot_cost_s=20, mtbf_s=24 * HOUR, max_loss_s=300)
        assert p.optimal_interval_s > 300
        assert p.interval_s == 300

    def test_a_floor_survives_an_absurd_max_loss(self):
        """Asking to lose less than one snapshot costs is not a cadence, it is
        a server that only writes checkpoints."""
        p = SnapshotPolicy(snapshot_cost_s=60, mtbf_s=2 * HOUR, max_loss_s=1.0)
        assert p.interval_s == p.min_interval_s

    def test_expected_loss_is_half_an_interval(self):
        p = SnapshotPolicy(snapshot_cost_s=20, mtbf_s=2 * HOUR, max_loss_s=300)
        assert p.expected_loss_s() == 150

    def test_overhead_is_minimised_at_the_optimum(self):
        p = SnapshotPolicy(snapshot_cost_s=20, mtbf_s=2 * HOUR, max_loss_s=None)
        t = p.optimal_interval_s
        assert p.overhead_fraction(t) < p.overhead_fraction(t * 0.5)
        assert p.overhead_fraction(t) < p.overhead_fraction(t * 2)

    def test_effective_cost_marks_up_the_sticker_price(self):
        """The number to compare against on-demand: a spot instance that
        wastes a third of its wall time is not a two-thirds discount."""
        p = SnapshotPolicy(snapshot_cost_s=20, mtbf_s=2 * HOUR, max_loss_s=300,
                           restart_s=900)
        assert p.effective_hourly_cost(0.63) == pytest.approx(0.63 * (1 + p.waste_fraction()))
        assert p.effective_hourly_cost(0.63) > 0.63

    def test_nonsense_inputs_are_refused(self):
        with pytest.raises(ValueError):
            SnapshotPolicy(snapshot_cost_s=0, mtbf_s=HOUR)
        with pytest.raises(ValueError):
            SnapshotPolicy(snapshot_cost_s=10, mtbf_s=0)

    def test_describe_says_when_the_optimum_was_overridden(self):
        p = SnapshotPolicy(snapshot_cost_s=20, mtbf_s=24 * HOUR, max_loss_s=300)
        assert "clamped" in p.describe()


class TestScheduler:
    def test_not_due_before_the_interval_elapses(self):
        clock = _Clock()
        s = SnapshotScheduler(SnapshotPolicy(snapshot_cost_s=20, mtbf_s=2 * HOUR,
                                             max_loss_s=300), clock=clock)
        assert not s.due()
        clock.advance(299)
        assert not s.due()
        clock.advance(2)
        assert s.due()

    def test_recording_a_snapshot_restarts_the_timer(self):
        clock = _Clock()
        s = SnapshotScheduler(SnapshotPolicy(snapshot_cost_s=20, mtbf_s=2 * HOUR,
                                             max_loss_s=300), clock=clock)
        clock.advance(400)
        assert s.due()
        s.record(20)
        assert not s.due()

    def test_the_first_measurement_replaces_the_guess_outright(self):
        """The configured cost is a seed, and a seed that is 6x optimistic
        would otherwise take several intervals to decay away — during which
        the run snapshots far too often."""
        s = SnapshotScheduler(SnapshotPolicy(snapshot_cost_s=10, mtbf_s=2 * HOUR,
                                             max_loss_s=None), clock=_Clock())
        s.record(60)
        assert s.policy.snapshot_cost_s == 60

    def test_later_measurements_are_smoothed_not_believed(self):
        """One unlucky write must not halve the cadence for the rest of the
        run."""
        s = SnapshotScheduler(SnapshotPolicy(snapshot_cost_s=10, mtbf_s=2 * HOUR,
                                             max_loss_s=None), clock=_Clock())
        s.record(20)
        s.record(200)
        assert 20 < s.policy.snapshot_cost_s < 200

    def test_a_costlier_snapshot_lengthens_the_interval(self):
        s = SnapshotScheduler(SnapshotPolicy(snapshot_cost_s=10, mtbf_s=2 * HOUR,
                                             max_loss_s=None), clock=_Clock())
        before = s.interval_s
        s.record(90)
        assert s.interval_s > before

    def test_skipping_does_not_leave_it_due_forever(self):
        """Without this a failed snapshot makes due() true on every step and
        the loop retries in a tight loop against a store that is down."""
        clock = _Clock()
        s = SnapshotScheduler(SnapshotPolicy(snapshot_cost_s=20, mtbf_s=2 * HOUR,
                                             max_loss_s=300), clock=clock)
        clock.advance(400)
        s.skipped()
        assert not s.due()

    def test_a_zero_duration_is_a_bug_not_a_free_snapshot(self):
        s = SnapshotScheduler(SnapshotPolicy(snapshot_cost_s=20, mtbf_s=HOUR),
                              clock=_Clock())
        with pytest.raises(ValueError):
            s.record(0)


class TestResumeDetection:
    def test_picks_the_highest_step(self):
        got = resume_target(_listing(
            (checkpoint_id_for_step(10), "training", "2026-01-01T00:00:00Z"),
            (checkpoint_id_for_step(30), "training", "2026-01-01T00:10:00Z"),
            (checkpoint_id_for_step(20), "training", "2026-01-01T00:05:00Z")))
        assert got == (f"tinker://model_a/weights/{checkpoint_id_for_step(30)}", 30)

    def test_the_step_wins_over_the_timestamp(self):
        """completed_at is a wall clock on a machine that just got replaced.
        If the two disagree, the clock is the witness not to believe."""
        got = resume_target(_listing(
            (checkpoint_id_for_step(30), "training", "2026-01-01T00:00:00Z"),
            (checkpoint_id_for_step(10), "training", "2099-01-01T00:00:00Z")))
        assert got[1] == 30

    def test_sampler_checkpoints_are_never_resumed_from(self):
        """load_weights rejects any path that is not …/weights/… (api.py:979),
        and a sampler checkpoint is an inference-format export — resuming from
        one drops the optimizer at best and 400s at worst."""
        got = resume_target(_listing(
            (checkpoint_id_for_step(40), "sampler", "2026-01-01T00:20:00Z"),
            (checkpoint_id_for_step(10), "training", "2026-01-01T00:00:00Z")))
        assert got[1] == 10

    def test_checkpoints_we_did_not_mint_are_ignored(self):
        """An id with no step in it cannot be placed in the run, so resuming
        from it would mean guessing where training left off."""
        assert latest_resumable(parse_checkpoints(
            _listing(("ss0_seq3", "training", None)))) is None

    def test_nothing_to_resume_from_is_not_an_error(self):
        """The normal state of a run's first minute."""
        assert resume_target({"checkpoints": []}) is None
        assert resume_target({}) is None

    def test_a_bare_list_is_accepted_too(self):
        rows = _listing((checkpoint_id_for_step(5), "training", None))["checkpoints"]
        assert resume_target(rows)[1] == 5

    def test_malformed_rows_do_not_sink_the_listing(self):
        """A listing that gained a field, or lost one, must not make a run
        unresumable."""
        payload = _listing((checkpoint_id_for_step(5), "training", None))
        payload["checkpoints"].extend([{"checkpoint_id": "x"}, "junk", {}])
        assert resume_target(payload)[1] == 5

    def test_the_path_names_the_old_model_id(self):
        """Resume is never 'reattach to model_id': create_model mints a fresh
        random id (api.py:824) and a new server's backend holds no models, so
        the dead id survives only as load_weights' source_model_id inside the
        path."""
        path, _ = resume_target(_listing(
            (checkpoint_id_for_step(5), "training", None)))
        assert path.startswith("tinker://model_a/weights/")


class TestManifest:
    def _manifest(self, generation=1, **kw):
        t = TenantState(name="policy-b", model_id="model_9fa2c1b0",
                        checkpoint_path="tinker://model_9fa2c1b0/weights/step-0000004200",
                        step=4200, data_cursor=118000, seed=7, lora_rank=32,
                        base_model="Qwen/Qwen3-4B")
        return ResumeManifest(generation=generation, run_id="run-1",
                              tenants={t.name: t}, step=4200, **kw)

    def test_round_trips(self):
        m = self._manifest(owner="job-7")
        back = ResumeManifest.from_json(m.to_json())
        assert back.generation == 1 and back.run_id == "run-1"
        assert back.owner == "job-7" and back.step == 4200
        assert back.tenants["policy-b"].data_cursor == 118000
        assert back.tenants["policy-b"].model_id == "model_9fa2c1b0"

    def test_it_carries_what_the_server_forgets(self):
        """SkyRL's models table has model_id, base_model, lora_config, status,
        request_id, session_id, created_at — and no user-metadata column. The
        tenant's NAME, its STEP and its DATA CURSOR have nowhere else to live,
        and without them a restart is a fresh run wearing old weights."""
        t = ResumeManifest.from_json(self._manifest().to_json()).tenants["policy-b"]
        assert (t.name, t.step, t.data_cursor) == ("policy-b", 4200, 118000)

    def test_a_truncated_object_is_rejected_not_half_read(self):
        """A process killed mid-upload leaves exactly this. Half a manifest
        that parses is worse than none: it would silently resume the tenants
        that happened to fit."""
        text = self._manifest().to_json()
        with pytest.raises(ManifestCorrupt):
            ResumeManifest.from_json(text[: len(text) // 2])

    def test_a_tampered_payload_fails_its_checksum(self):
        text = self._manifest().to_json().replace("4200", "9999")
        with pytest.raises(ManifestCorrupt):
            ResumeManifest.from_json(text)

    def test_a_future_version_is_refused_rather_than_guessed_at(self):
        import json
        m = self._manifest()
        payload = m.payload()
        payload["version"] = ResumeManifest.VERSION + 1
        import hashlib
        body = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        text = json.dumps({"checksum": hashlib.sha256(body.encode()).hexdigest(),
                           "payload": payload})
        with pytest.raises(ManifestCorrupt, match="newer"):
            ResumeManifest.from_json(text)

    def test_succeeding_bumps_the_generation(self):
        m = self._manifest(generation=4)
        nxt = m.succeed(m.tenants, step=4300, owner="job-8")
        assert nxt.generation == 5 and nxt.step == 4300 and nxt.owner == "job-8"

    def test_generations_are_lexically_ordered(self):
        """A plain bucket listing is already in generation order, so finding
        the newest never needs a second round trip to read timestamps."""
        names = [manifest_name(g) for g in (2, 10, 100)]
        assert sorted(names) == names

    def test_generation_parses_out_of_a_full_key(self):
        assert manifest_generation(f"s3://bucket/run-1/{manifest_name(9)}") == 9
        assert manifest_generation("checkpoints/step-0000000001.tar.gz") is None


class TestLatestManifest:
    def _store(self, *manifests):
        return {m.object_name(): m.to_json() for m in manifests}

    def _m(self, generation, name="a"):
        t = TenantState(name=name, model_id=f"model_{generation}",
                        checkpoint_path=f"tinker://model_{generation}/weights/step-0000000001",
                        step=generation)
        return ResumeManifest(generation=generation, run_id="r",
                              tenants={t.name: t}, step=generation)

    def test_the_newest_generation_wins(self):
        store = self._store(self._m(1), self._m(2), self._m(3))
        got = latest_manifest(list(store), store.__getitem__)
        assert got.generation == 3

    def test_a_torn_top_generation_falls_back_one(self):
        """The crash that interrupts the manifest upload is precisely the
        crash the manifest exists to cover. Refusing to resume there would
        lose the run to the one write that got cut off — so a bad generation
        costs an interval of redone work, not the run."""
        store = self._store(self._m(1), self._m(2))
        store[manifest_name(3)] = '{"checksum": "deadbeef", "payload": {}}'
        got = latest_manifest(list(store), store.__getitem__)
        assert got.generation == 2

    def test_an_unreadable_object_is_skipped_not_raised(self):
        """A store that 404s one key mid-listing must not take the run down."""
        store = self._store(self._m(1))

        def load(name):
            if name == manifest_name(2):
                raise OSError("gone")
            return store[name]

        got = latest_manifest([manifest_name(1), manifest_name(2)], load)
        assert got.generation == 1

    def test_nothing_valid_reads_as_a_fresh_run(self):
        assert latest_manifest([], lambda n: "") is None
        assert latest_manifest(["checkpoints/", "tinker.db"], lambda n: "") is None

    def test_non_manifest_keys_are_ignored(self):
        store = self._store(self._m(4))
        names = [*store, "step-0000000004.tar.gz", "manifest.json"]
        assert latest_manifest(names, store.__getitem__).generation == 4
