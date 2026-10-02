import unittest
from datetime import datetime, timedelta, timezone

from aquaflow.api import AquaFlowService
from aquaflow.lab.coldchain import (
    COLLECTED, EXCEPTION, HANDOVER, RECEIVED, TEMPERATURE,
    PRESERVATION_EXPIRED, TEMPERATURE_EXCURSION, TRANSIT_SLA_BREACH,
    Event, Policy, apply, excursions, expired, history, open_chain,
    preservation_deadline, releasable, replay, transit_breached,
    transit_deadline, violations,
)

UTC = timezone.utc
POLICY = Policy(transit=timedelta(hours=6), preservation=timedelta(hours=24),
                temp_low=0.0, temp_high=6.0)
T0 = datetime(2026, 1, 1, 8, 0, tzinfo=UTC)


def valid_events():
    return [
        Event(1, COLLECTED, T0, actor="collector.a"),
        Event(2, HANDOVER, T0 + timedelta(minutes=30), actor="collector.a", dst="courier.b", seal="SEAL-1"),
        Event(3, TEMPERATURE, T0 + timedelta(hours=1), celsius=4.2),
        Event(4, HANDOVER, T0 + timedelta(hours=1, minutes=30), actor="courier.b", dst="lab.c", seal="SEAL-2"),
        Event(5, TEMPERATURE, T0 + timedelta(hours=1, minutes=45), celsius=3.8),
        Event(6, RECEIVED, T0 + timedelta(hours=2), actor="lab.c"),
    ]


class ValidChain(unittest.TestCase):
    def test_valid_chain_is_releasable(self):
        chain = replay("S-1", POLICY, valid_events())
        at = T0 + timedelta(hours=4)
        self.assertEqual(chain.state, "received")
        self.assertEqual(chain.version, 6)
        self.assertEqual(chain.custodian, "lab.c")
        self.assertEqual(violations(chain, at), frozenset())
        self.assertTrue(releasable(chain, at))

    def test_deadlines_are_utc(self):
        chain = replay("S-1", POLICY, valid_events())
        self.assertEqual(transit_deadline(chain), T0 + timedelta(hours=6))
        self.assertEqual(preservation_deadline(chain), T0 + timedelta(hours=24))
        self.assertEqual(chain.collected_at.tzinfo, UTC)
        self.assertEqual(chain.received_at.tzinfo, UTC)

    def test_receipt_exactly_at_transit_deadline_is_in_time(self):
        events = valid_events()[:-1] + [Event(6, RECEIVED, T0 + timedelta(hours=6), actor="lab.c")]
        chain = replay("S-1", POLICY, events)
        self.assertFalse(transit_breached(chain))
        self.assertTrue(releasable(chain, T0 + timedelta(hours=6)))


class TemperatureExcursion(unittest.TestCase):
    def test_excursion_within_preservation_is_flagged_but_releasable(self):
        events = valid_events()
        events[2] = Event(3, TEMPERATURE, T0 + timedelta(hours=1), celsius=9.5)
        chain = replay("S-1", POLICY, events)
        at = T0 + timedelta(hours=4)
        self.assertEqual(len(excursions(chain)), 1)
        self.assertEqual(violations(chain, at), frozenset({TEMPERATURE_EXCURSION}))
        self.assertTrue(releasable(chain, at))

    def test_excursion_after_preservation_expiry_is_not_releasable(self):
        events = valid_events()
        events[2] = Event(3, TEMPERATURE, T0 + timedelta(hours=1), celsius=9.5)
        chain = replay("S-1", POLICY, events)
        at = T0 + timedelta(hours=25)
        self.assertEqual(violations(chain, at),
                         frozenset({TEMPERATURE_EXCURSION, PRESERVATION_EXPIRED}))
        self.assertFalse(releasable(chain, at))


class LateReceipt(unittest.TestCase):
    def late_events(self):
        events = valid_events()[:-1]
        events.append(Event(6, RECEIVED, T0 + timedelta(hours=7), actor="lab.c"))
        return events

    def test_late_receipt_is_flagged_and_not_releasable(self):
        chain = replay("S-1", POLICY, self.late_events())
        at = T0 + timedelta(hours=8)
        self.assertTrue(transit_breached(chain))
        self.assertEqual(violations(chain, at), frozenset({TRANSIT_SLA_BREACH}))
        self.assertFalse(releasable(chain, at))

    def test_exception_is_recorded_but_never_unblocks(self):
        events = self.late_events()
        events.append(Event(7, EXCEPTION, T0 + timedelta(hours=8),
                            actor="qa.reviewer", reason="courier traffic delay",
                            flag=TRANSIT_SLA_BREACH))
        chain = replay("S-1", POLICY, events)
        at = T0 + timedelta(hours=9)
        self.assertEqual(len(chain.exceptions), 1)
        self.assertEqual(chain.exceptions[0].reviewer, "qa.reviewer")
        self.assertEqual(chain.exceptions[0].flag, TRANSIT_SLA_BREACH)
        # original evidence is untouched and the block stands
        self.assertEqual(chain.version, 7)
        self.assertEqual(len(chain.readings), 2)
        self.assertEqual(chain.seals, frozenset({"SEAL-1", "SEAL-2"}))
        self.assertTrue(transit_breached(chain))
        self.assertFalse(releasable(chain, at))


class Replay(unittest.TestCase):
    def test_replay_matches_incremental_apply(self):
        events = valid_events()
        chain = open_chain("S-1", POLICY)
        for event in events:
            chain = apply(chain, event)
        self.assertEqual(chain, replay("S-1", POLICY, events))

    def test_history_replays_every_prefix(self):
        events = valid_events()
        states = history("S-1", POLICY, events)
        self.assertEqual(len(states), len(events) + 1)
        self.assertEqual(states[-1], replay("S-1", POLICY, events))
        self.assertEqual(states[0].state, "new")
        self.assertEqual(states[2].custodian, "courier.b")
        self.assertEqual(replay("S-1", POLICY, events[:3]), states[3])

    def test_replay_is_order_independent_on_input(self):
        events = valid_events()
        self.assertEqual(replay("S-1", POLICY, list(reversed(events))),
                         replay("S-1", POLICY, events))


class Utc(unittest.TestCase):
    def test_offsets_are_normalised_to_utc(self):
        collected = Event(1, COLLECTED, datetime(2026, 1, 1, 10, 0,
                          tzinfo=timezone(timedelta(hours=2))), actor="collector.a")
        received = Event(2, RECEIVED, datetime(2026, 1, 1, 11, 0, tzinfo=UTC),
                         actor="collector.a")
        chain = replay("S-1", POLICY, [collected, received])
        self.assertEqual(chain.collected_at, datetime(2026, 1, 1, 8, 0, tzinfo=UTC))
        self.assertFalse(transit_breached(chain))  # 3h transit, limit 6h

    def test_naive_datetimes_are_treated_as_utc(self):
        events = [Event(1, COLLECTED, datetime(2026, 1, 1, 8, 0), actor="collector.a"),
                  Event(2, RECEIVED, datetime(2026, 1, 1, 9, 0), actor="collector.a")]
        chain = replay("S-1", POLICY, events)
        self.assertEqual(chain.received_at, datetime(2026, 1, 1, 9, 0, tzinfo=UTC))

    def test_time_reversal_across_offsets_is_rejected(self):
        events = [Event(1, COLLECTED, datetime(2026, 1, 1, 8, 0, tzinfo=UTC), actor="collector.a"),
                  Event(2, RECEIVED, datetime(2026, 1, 1, 9, 30,
                        tzinfo=timezone(timedelta(hours=2))), actor="collector.a")]
        with self.assertRaises(ValueError):
            replay("S-1", POLICY, events)


class InvalidChain(unittest.TestCase):
    def test_handover_from_wrong_custodian(self):
        events = [Event(1, COLLECTED, T0, actor="collector.a"),
                  Event(2, HANDOVER, T0 + timedelta(minutes=5), actor="courier.b", dst="lab.c", seal="S-1")]
        with self.assertRaises(ValueError):
            replay("S-1", POLICY, events)

    def test_seal_reuse(self):
        events = [Event(1, COLLECTED, T0, actor="collector.a"),
                  Event(2, HANDOVER, T0 + timedelta(minutes=5), actor="collector.a", dst="courier.b", seal="S-1"),
                  Event(3, HANDOVER, T0 + timedelta(minutes=10), actor="courier.b", dst="lab.c", seal="S-1")]
        with self.assertRaises(ValueError):
            replay("S-1", POLICY, events)

    def test_receipt_by_non_custodian(self):
        events = valid_events()[:-1]
        events.append(Event(6, RECEIVED, T0 + timedelta(hours=2), actor="lab.x"))
        with self.assertRaises(ValueError):
            replay("S-1", POLICY, events)

    def test_sequence_gap(self):
        events = valid_events()
        events[3] = Event(7, HANDOVER, T0 + timedelta(hours=1, minutes=30), actor="courier.b", dst="lab.c", seal="SEAL-2")
        with self.assertRaises(ValueError):
            replay("S-1", POLICY, events)

    def test_no_events_after_receipt_except_exception(self):
        events = valid_events()
        events.append(Event(7, TEMPERATURE, T0 + timedelta(hours=3), celsius=4.0))
        with self.assertRaises(ValueError):
            replay("S-1", POLICY, events)

    def test_unreceived_sample_is_not_releasable(self):
        chain = replay("S-1", POLICY, valid_events()[:3])
        self.assertFalse(releasable(chain, T0 + timedelta(hours=4)))


class Expiry(unittest.TestCase):
    def test_preservation_expiry_blocks_release(self):
        chain = replay("S-1", POLICY, valid_events())
        at = T0 + timedelta(hours=25)
        self.assertTrue(expired(chain, at))
        self.assertEqual(violations(chain, at), frozenset({PRESERVATION_EXPIRED}))
        self.assertFalse(releasable(chain, at))


class Workflow(unittest.TestCase):
    def payload(self, received_at, extra=None, at="2026-01-01T12:00:00Z"):
        events = [
            {"seq": 1, "kind": "collected", "at": "2026-01-01T08:00:00Z", "actor": "collector.a"},
            {"seq": 2, "kind": "handover", "at": "2026-01-01T08:30:00Z", "actor": "collector.a", "dst": "courier.b", "seal": "SEAL-1"},
            {"seq": 3, "kind": "temperature", "at": "2026-01-01T09:00:00Z", "celsius": 4.2},
            {"seq": 4, "kind": "handover", "at": "2026-01-01T09:30:00Z", "actor": "courier.b", "dst": "lab.c", "seal": "SEAL-2"},
            {"seq": 5, "kind": "received", "at": received_at, "actor": "lab.c"},
        ]
        events.extend(extra or [])
        return {"sample": "S-1",
                "policy": {"transit_hours": 6, "preservation_hours": 24, "temp_low": 0, "temp_high": 6},
                "events": events, "at": at}

    def test_valid_chain_releases(self):
        result = AquaFlowService().run_workflow("cold_chain", self.payload("2026-01-01T10:00:00Z"))
        self.assertEqual(result["decision"], "release")
        self.assertEqual(result["value"], [])
        self.assertEqual(result["evidence"]["state"], "received")

    def test_temperature_excursion_still_releases_with_flag(self):
        payload = self.payload("2026-01-01T10:00:00Z")
        payload["events"][2]["celsius"] = 9.5
        result = AquaFlowService().run_workflow("cold_chain", payload)
        self.assertEqual(result["decision"], "release")
        self.assertEqual(result["value"], [TEMPERATURE_EXCURSION])

    def test_late_receipt_holds_even_with_exception(self):
        extra = [{"seq": 6, "kind": "exception", "at": "2026-01-01T16:00:00Z",
                  "actor": "qa.reviewer", "reason": "traffic delay", "flag": TRANSIT_SLA_BREACH}]
        result = AquaFlowService().run_workflow("cold_chain", self.payload("2026-01-01T15:30:00Z", extra, at="2026-01-01T18:00:00Z"))
        self.assertEqual(result["decision"], "hold")
        self.assertEqual(result["value"], [TRANSIT_SLA_BREACH])
        self.assertEqual(result["evidence"]["exceptions"], 1)


if __name__ == "__main__":
    unittest.main()
