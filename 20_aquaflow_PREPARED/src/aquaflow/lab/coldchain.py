"""Cold-chain tracking from sampling to laboratory receipt.

The chain is an append-only event log: collection, custody handovers,
temperature readings, laboratory receipt and reviewer exceptions are
recorded as immutable events.  The chain state is a pure fold over the
log, so any prefix can be replayed to audit how a decision was reached.
There is intentionally no API to edit or remove events; a reviewer
exception is appended as new evidence and never alters the original
record.  All time comparisons are normalised to UTC.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone

COLLECTED = "collected"
HANDOVER = "handover"
TEMPERATURE = "temperature"
RECEIVED = "received"
EXCEPTION = "exception"

TEMPERATURE_EXCURSION = "temperature_excursion"
TRANSIT_SLA_BREACH = "transit_sla_breach"
PRESERVATION_EXPIRED = "preservation_expired"


def _utc(at: datetime) -> datetime:
    if at.tzinfo is None:
        return at.replace(tzinfo=timezone.utc)
    return at.astimezone(timezone.utc)


@dataclass(frozen=True)
class Policy:
    transit: timedelta        # service time limit: collection -> lab receipt
    preservation: timedelta   # preservation (holding) limit from collection
    temp_low: float           # cold-chain envelope, Celsius
    temp_high: float

    def __post_init__(self) -> None:
        if self.transit <= timedelta(0) or self.preservation <= timedelta(0):
            raise ValueError("window")
        if self.temp_low >= self.temp_high:
            raise ValueError("envelope")


@dataclass(frozen=True)
class Event:
    seq: int
    kind: str
    at: datetime
    actor: str = ""
    dst: str = ""
    seal: str = ""
    celsius: float | None = None
    reason: str = ""
    flag: str = ""


@dataclass(frozen=True)
class ExceptionRecord:
    reviewer: str
    reason: str
    flag: str
    at: datetime


@dataclass(frozen=True)
class Chain:
    sample_id: str
    policy: Policy
    state: str = "new"
    version: int = 0
    custodian: str = ""
    collected_at: datetime | None = None
    received_at: datetime | None = None
    updated_at: datetime | None = None
    seals: frozenset[str] = field(default_factory=frozenset)
    readings: tuple[tuple[datetime, float], ...] = ()
    exceptions: tuple[ExceptionRecord, ...] = ()


def open_chain(sample_id: str, policy: Policy) -> Chain:
    if not sample_id:
        raise ValueError("sample")
    return Chain(sample_id, policy)


def apply(chain: Chain, event: Event) -> Chain:
    if event.seq != chain.version + 1:
        raise ValueError("sequence")
    at = _utc(event.at)
    if chain.updated_at is not None and at < chain.updated_at:
        raise ValueError("time reversal")
    version = chain.version + 1
    if event.kind == COLLECTED:
        if chain.state != "new":
            raise ValueError("collect once")
        if not event.actor:
            raise ValueError("collector")
        return replace(chain, state="open", version=version,
                       custodian=event.actor, collected_at=at, updated_at=at)
    if event.kind == HANDOVER:
        if chain.state != "open":
            raise ValueError("not in transit")
        if event.actor != chain.custodian or not event.dst or event.dst == event.actor:
            raise ValueError("custody")
        if not event.seal or event.seal in chain.seals:
            raise ValueError("seal")
        return replace(chain, version=version, custodian=event.dst,
                       seals=chain.seals | {event.seal}, updated_at=at)
    if event.kind == TEMPERATURE:
        if chain.state != "open":
            raise ValueError("not in transit")
        if event.celsius is None:
            raise ValueError("temperature")
        return replace(chain, version=version,
                       readings=chain.readings + ((at, float(event.celsius)),),
                       updated_at=at)
    if event.kind == RECEIVED:
        if chain.state != "open":
            raise ValueError("not in transit")
        if event.actor != chain.custodian:
            raise ValueError("custody")
        return replace(chain, state="received", version=version,
                       received_at=at, updated_at=at)
    if event.kind == EXCEPTION:
        if chain.state == "new":
            raise ValueError("no chain")
        if not event.actor or not event.reason:
            raise ValueError("exception")
        record = ExceptionRecord(event.actor, event.reason, event.flag, at)
        return replace(chain, version=version,
                       exceptions=chain.exceptions + (record,), updated_at=at)
    raise ValueError("unknown event")


def replay(sample_id: str, policy: Policy, events: list[Event]) -> Chain:
    chain = open_chain(sample_id, policy)
    for event in sorted(events, key=lambda e: e.seq):
        chain = apply(chain, event)
    return chain


def history(sample_id: str, policy: Policy, events: list[Event]) -> list[Chain]:
    chain = open_chain(sample_id, policy)
    states = [chain]
    for event in sorted(events, key=lambda e: e.seq):
        chain = apply(chain, event)
        states.append(chain)
    return states


def transit_deadline(chain: Chain) -> datetime | None:
    if chain.collected_at is None:
        return None
    return chain.collected_at + chain.policy.transit


def preservation_deadline(chain: Chain) -> datetime | None:
    if chain.collected_at is None:
        return None
    return chain.collected_at + chain.policy.preservation


def transit_breached(chain: Chain) -> bool:
    deadline = transit_deadline(chain)
    if deadline is None or chain.received_at is None:
        return False
    return chain.received_at > deadline


def expired(chain: Chain, at: datetime) -> bool:
    deadline = preservation_deadline(chain)
    if deadline is None:
        return False
    return _utc(at) > deadline


def excursions(chain: Chain) -> tuple[tuple[datetime, float], ...]:
    low, high = chain.policy.temp_low, chain.policy.temp_high
    return tuple(r for r in chain.readings if r[1] < low or r[1] > high)


def violations(chain: Chain, at: datetime) -> frozenset[str]:
    found = set()
    if excursions(chain):
        found.add(TEMPERATURE_EXCURSION)
    if transit_breached(chain):
        found.add(TRANSIT_SLA_BREACH)
    if expired(chain, at):
        found.add(PRESERVATION_EXPIRED)
    return frozenset(found)


def releasable(chain: Chain, at: datetime) -> bool:
    """A sample may be released only after a valid, in-time receipt.

    Temperature excursions are flagged for review but do not block
    release while the sample is inside its preservation window; a
    transit service-limit breach or an expired preservation window
    does.  Reviewer exceptions document deviations, they never lift
    a block.
    """
    return (chain.state == "received"
            and not transit_breached(chain)
            and not expired(chain, at))
