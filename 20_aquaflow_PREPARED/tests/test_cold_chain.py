import unittest
from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta, timezone

from aquaflow.api import AquaFlowService
from aquaflow.lab.cold_chain import (
    ColdChainState,
    Handoff,
    Received,
    ReviewException,
    SampleCollected,
    TemperatureRecorded,
    apply,
    decide,
    is_releaseable,
    replay,
    sla_breaches,
)

UTC = timezone.utc
# Refrigerated drinking-water sample: 2..8 C, 360 minute preservation window.
T_MIN, T_MAX, HOLD = 2.0, 8.0, 360


def t(hour: float, minute: float = 0) -> datetime:
    return datetime(2026, 10, 2, int(hour), int(minute), tzinfo=UTC)


def collected(sample="S1", collector="sampler", lab="LAB"):
    return SampleCollected(sample, t(8), T_MIN, T_MAX, HOLD, lab, collector)


def valid_chain_events(sample="S1", collector="sampler", lab="LAB"):
    """Collector -> courier -> lab, received inside the preservation window."""
    return [
        Handoff(sample, 1, collector, "courier", t(8, 5), "seal-1"),
        Handoff(sample, 2, "courier", lab, t(11), "seal-2"),
        Received(sample, t(11, 30), accepted=True, by="lab-tech"),
    ]


class ColdChainTest(unittest.TestCase):
    # --- Scenario 1: valid chain is publishable -------------------------- #
    def test_valid_chain_is_releaseable(self):
        state = replay(collected(), valid_chain_events())
        decision = decide(state)
        self.assertTrue(state.chain_valid)
        self.assertTrue(state.received)
        self.assertFalse(state.holding_breached)
        self.assertTrue(decision.releaseable)
        self.assertEqual(decision.hard_blocks, ())
        self.assertEqual(decision.open_deviations, ())
        self.assertEqual(sla_breaches(state), ())

    # --- Scenario 2: temperature excursion, still within holding time ---- #
    def test_temperature_excursion_holds_until_exception(self):
        events = [
            Handoff("S1", 1, "sampler", "courier", t(8, 5), "seal-1"),
            # Reading out of the 2..8 C envelope while still in transit.
            TemperatureRecorded("S1", t(9, 30), 11.5, "courier"),
            Handoff("S1", 2, "courier", "LAB", t(11), "seal-2"),
            Received("S1", t(11, 30), accepted=True),
        ]
        state = replay(collected(), events)
        decision = decide(state)
        # Within the preservation window: this is a flagged soft deviation,
        # not a hard service-time-limit breach.
        self.assertFalse(state.holding_breached)
        self.assertEqual(len(state.temperature_excursions), 1)
        self.assertFalse(decision.releaseable)
        self.assertEqual(decision.hard_blocks, ())
        self.assertIn("temperature_excursion", decision.open_deviations)
        self.assertIn("temperature_excursion", sla_breaches(state))
        self.assertFalse(is_releaseable(state))

    def test_reviewer_exception_resolves_temperature_hold(self):
        events = valid_chain_events()
        events.insert(1, TemperatureRecorded("S1", t(9, 30), 11.5, "courier"))
        events.append(
            ReviewException(
                "S1", t(12), "reviewer-qc",
                "brief excursion in insulated shipper, integrity retained",
                codes=("temperature_excursion",),
            )
        )
        state = replay(collected(), events)
        decision = decide(state)
        self.assertTrue(decision.releaseable)
        self.assertEqual(decision.open_deviations, ())
        self.assertEqual(decision.resolved_deviations, ("temperature_excursion",))
        # The excursion reading is retained as original evidence.
        self.assertEqual(len(state.temperature_excursions), 1)
        self.assertEqual(len(state.exceptions), 1)

    # --- Scenario 3: late receipt stays non-publishable ------------------- #
    def test_late_receipt_is_never_releaseable(self):
        events = [
            Handoff("S1", 1, "sampler", "courier", t(8, 5), "seal-1"),
            Handoff("S1", 2, "courier", "LAB", t(14, 5), "seal-2"),
            # Deadline is 14:00 UTC; receipt at 15:00 breaches the limit.
            Received("S1", t(15), accepted=True),
        ]
        state = replay(collected(), events)
        decision = decide(state)
        self.assertTrue(state.holding_breached)
        self.assertIn("holding_time_breach", decision.hard_blocks)
        self.assertFalse(decision.releaseable)
        self.assertFalse(is_releaseable(state))
        self.assertIn("holding_time_breach", sla_breaches(state))
        self.assertEqual(state.holding_deadline, t(14))

    def test_late_receipt_cannot_be_overridden_by_exception(self):
        events = [
            Handoff("S1", 1, "sampler", "courier", t(8, 5), "seal-1"),
            Handoff("S1", 2, "courier", "LAB", t(14, 5), "seal-2"),
            Received("S1", t(15), accepted=True),
            # Reviewer tries to excuse the late arrival.
            ReviewException(
                "S1", t(15, 5), "reviewer-qc", "traffic delay, please accept",
                codes=("holding_time_breach",),
            ),
        ]
        state = replay(collected(), events)
        decision = decide(state)
        self.assertFalse(decision.releaseable)
        self.assertIn("holding_time_breach", decision.hard_blocks)
        # The exception is kept as evidence but does not clear the hard block.
        self.assertEqual(len(state.exceptions), 1)
        self.assertNotIn("holding_time_breach", decision.resolved_deviations)

    # --- Broken custody chain blocks release ------------------------------ #
    def test_broken_chain_blocks_release(self):
        events = [
            # First handoff claims custody from "nobody" != current holder.
            Handoff("S1", 1, "intruder", "courier", t(8, 5), "seal-1"),
            Handoff("S1", 2, "courier", "LAB", t(11), "seal-2"),
            Received("S1", t(11, 30), accepted=True),
        ]
        state = replay(collected(), events)
        self.assertFalse(state.chain_valid)
        self.assertIn("custody_gap", state.custody_breaks)
        self.assertFalse(is_releaseable(state))
        # Reusing a seal is also an integrity breach.
        reused = valid_chain_events()
        reused[1] = Handoff("S1", 2, "courier", "LAB", t(11), "seal-1")
        state2 = replay(collected(), reused)
        self.assertIn("duplicate_seal", state2.custody_breaks)
        self.assertFalse(is_releaseable(state2))

    def test_rejected_and_undelivered_are_blocked(self):
        # Reached the lab but rejected.
        rejected = replay(
            collected("R"),
            [
                Handoff("R", 1, "sampler", "courier", t(8, 5), "s1"),
                Handoff("R", 2, "courier", "LAB", t(11), "s2"),
                Received("R", t(11, 30), accepted=False, note="cracked bottle"),
            ],
        )
        self.assertIn("receipt_rejected", decide(rejected).hard_blocks)
        self.assertFalse(decide(rejected).releaseable)
        # Never received at all.
        lost = replay(
            collected("L"),
            [Handoff("L", 1, "sampler", "courier", t(8, 5), "s1")],
        )
        self.assertIn("not_received", decide(lost).hard_blocks)
        self.assertFalse(decide(lost).releaseable)

    # --- UTC handling ----------------------------------------------------- #
    def test_all_comparisons_use_utc(self):
        # 17:00+03:00 == 14:00 UTC; 17:30+03:00 == 14:30 UTC -> late.
        tz3 = timezone(timedelta(hours=3))
        events = [
            Handoff("S1", 1, "sampler", "courier",
                    datetime(2026, 10, 2, 11, 5, tzinfo=tz3), "seal-1"),
            Handoff("S1", 2, "courier", "LAB",
                    datetime(2026, 10, 2, 17, 5, tzinfo=tz3), "seal-2"),
            Received("S1", datetime(2026, 10, 2, 17, 30, tzinfo=tz3), True),
        ]
        state = replay(collected(), events)
        self.assertEqual(state.received_at, datetime(2026, 10, 2, 14, 30, tzinfo=UTC))
        self.assertTrue(state.holding_breached)
        self.assertFalse(decide(state).releaseable)

    def test_naive_timestamps_rejected(self):
        # Validation runs when the evidence enters the chain (start/replay).
        with self.assertRaises(ValueError):
            replay(
                SampleCollected("S", datetime(2026, 10, 2, 8), T_MIN, T_MAX, HOLD, "LAB", "c"),
                [],
            )
        with self.assertRaises(ValueError):
            apply(
                replay(collected(), []),
                Handoff("S1", 1, "sampler", "courier",
                        datetime(2026, 10, 2, 8, 5), "seal-1"),
            )

    # --- Replay / immutability ------------------------------------------- #
    def test_state_changes_are_replayable(self):
        events = valid_chain_events()
        # Incremental fold ...
        incremental = apply(apply(replay(collected(), []), events[0]), events[1])
        incremental = apply(incremental, events[2])
        # ... must equal a full replay from the log.
        full = replay(collected(), events)
        for attr in (
            "holder", "custody_breaks", "evidence_breaks", "received_at",
            "accepted", "holding_breached", "chain_valid", "version",
            "handoff_count",
        ):
            self.assertEqual(getattr(incremental, attr), getattr(full, attr), attr)
        self.assertTrue(is_releaseable(full))
        # Replaying the same log is deterministic.
        self.assertEqual(replay(collected(), list(events)), full)

    def test_evidence_is_immutable(self):
        state = replay(collected(), valid_chain_events())
        with self.assertRaises(FrozenInstanceError):
            state.custody_breaks = ("tampered",)  # type: ignore[misc]
        with self.assertRaises(FrozenInstanceError):
            state.received_at = t(9)  # type: ignore[misc]
        self.assertIsInstance(state, ColdChainState)

    def test_exception_requires_identity_and_reason(self):
        with self.assertRaises(ValueError):
            apply(
                replay(collected(), valid_chain_events()),
                ReviewException("S1", t(12), "", "reason", ("temperature_excursion",)),
            )


class ColdChainWorkflowTest(unittest.TestCase):
    def _sample(self, sid="S1"):
        return {
            "id": sid,
            "collected_at": "2026-10-02T08:00:00+00:00",
            "temp_min_c": 2.0,
            "temp_max_c": 8.0,
            "holding_minutes": 360,
            "lab": "LAB",
            "collector": "sampler",
        }

    def test_valid_workflow_releases(self):
        events = [
            {"type": "handoff", "sample": "S1", "seq": 1, "src": "sampler",
             "dst": "courier", "at": "2026-10-02T08:05:00Z", "seal": "seal-1"},
            {"type": "handoff", "sample": "S1", "seq": 2, "src": "courier",
             "dst": "LAB", "at": "2026-10-02T11:00:00Z", "seal": "seal-2"},
            {"type": "received", "sample": "S1",
             "at": "2026-10-02T11:30:00Z", "accepted": True},
        ]
        result = AquaFlowService().cold_chain_release(self._sample(), events)
        self.assertEqual(result["decision"], "release")
        self.assertTrue(result["value"])
        self.assertEqual(result["evidence"]["hard_blocks"], [])
        self.assertEqual(result["workflow"], "cold_chain_release")

    def test_excursion_then_exception_workflow(self):
        events = [
            {"type": "handoff", "sample": "S1", "seq": 1, "src": "sampler",
             "dst": "courier", "at": "2026-10-02T08:05:00Z", "seal": "seal-1"},
            {"type": "temperature", "sample": "S1",
             "at": "2026-10-02T09:30:00Z", "celsius": 11.5},
            {"type": "handoff", "sample": "S1", "seq": 2, "src": "courier",
             "dst": "LAB", "at": "2026-10-02T11:00:00Z", "seal": "seal-2"},
            {"type": "received", "sample": "S1",
             "at": "2026-10-02T11:30:00Z", "accepted": True},
        ]
        svc = AquaFlowService()
        blocked = svc.cold_chain_release(self._sample(), events)
        self.assertEqual(blocked["decision"], "block")
        self.assertIn("temperature_excursion", blocked["evidence"]["open_deviations"])

        events.append(
            {"type": "exception", "sample": "S1",
             "at": "2026-10-02T12:00:00Z", "reviewer": "reviewer-qc",
             "reason": "short excursion, integrity retained",
             "codes": ["temperature_excursion"]}
        )
        released = svc.cold_chain_release(self._sample(), events)
        self.assertEqual(released["decision"], "release")
        self.assertEqual(released["evidence"]["temperature_readings"], 1)

    def test_late_workflow_blocks_and_keeps_evidence(self):
        events = [
            {"type": "handoff", "sample": "S1", "seq": 1, "src": "sampler",
             "dst": "courier", "at": "2026-10-02T08:05:00Z", "seal": "seal-1"},
            {"type": "handoff", "sample": "S1", "seq": 2, "src": "courier",
             "dst": "LAB", "at": "2026-10-02T14:05:00Z", "seal": "seal-2"},
            {"type": "received", "sample": "S1",
             "at": "2026-10-02T15:00:00Z", "accepted": True},
            {"type": "exception", "sample": "S1",
             "at": "2026-10-02T15:05:00Z", "reviewer": "reviewer-qc",
             "reason": "traffic", "codes": ["holding_time_breach"]},
        ]
        result = AquaFlowService().cold_chain_release(self._sample(), events)
        self.assertEqual(result["decision"], "block")
        self.assertIn("holding_time_breach", result["evidence"]["hard_blocks"])
        self.assertTrue(result["evidence"]["holding_breached"])
        self.assertEqual(result["evidence"]["exceptions"], 1)


if __name__ == "__main__":
    unittest.main()
