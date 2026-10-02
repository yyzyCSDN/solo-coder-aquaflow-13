"""Cold-chain tracking from sampling to laboratory receipt.

The module records an append-only evidence stream for every sample:

* :class:`SampleCollected` opens the chain and fixes the preservation window
  (maximum transit/holding time) and the acceptable temperature envelope;
* :class:`Handoff` records each custody transfer between organisations, with a
  tamper-evident seal (mirrors :mod:`aquaflow.lab.custody`);
* :class:`TemperatureRecorded` attaches thermometer readings taken in transit;
* :class:`Received` records laboratory acceptance or rejection;
* :class:`ReviewException` lets a reviewer annotate a deviation.  Exceptions
  are additional evidence: they never erase earlier events and can never
  override a hard service-time-limit (preservation) breach or an invalid chain.

State is derived exclusively by folding the events (:func:`apply` /
:func:`replay`), so every state change is reproducible by replaying the log.
All time comparisons are normalised to UTC on ingress -- naive datetimes are
rejected rather than silently interpreted.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from typing import Iterable, Union

__all__ = [
    "SampleCollected",
    "Handoff",
    "TemperatureRecorded",
    "Received",
    "ReviewException",
    "ColdChainEvent",
    "TemperaturePoint",
    "ExceptionRecord",
    "ColdChainState",
    "ReleaseDecision",
    "start",
    "apply",
    "replay",
    "decide",
    "deadline",
    "is_releaseable",
    "sla_breaches",
]


def _utc(value: datetime) -> datetime:
    """Normalise an aware datetime to UTC; naive timestamps are rejected."""
    if not isinstance(value, datetime):
        raise TypeError("timestamp must be a datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("naive timestamp: timezone-aware datetime required")
    return value.astimezone(timezone.utc)


def _append(items: tuple[str, ...], code: str) -> tuple[str, ...]:
    """Append a de-duplicated breach code while preserving first-seen order."""
    return items if code in items else items + (code,)


# --------------------------------------------------------------------------- #
# Evidence events (immutable, append-only)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SampleCollected:
    sample_id: str
    collected_at: datetime
    temp_min_c: float
    temp_max_c: float
    holding_minutes: float
    lab: str
    collector: str


@dataclass(frozen=True)
class Handoff:
    sample_id: str
    seq: int
    src: str
    dst: str
    at: datetime
    seal: str


@dataclass(frozen=True)
class TemperatureRecorded:
    sample_id: str
    at: datetime
    celsius: float
    recorded_by: str = ""


@dataclass(frozen=True)
class Received:
    sample_id: str
    at: datetime
    accepted: bool = True
    by: str = ""
    note: str = ""


@dataclass(frozen=True)
class ReviewException:
    sample_id: str
    at: datetime
    reviewer: str
    reason: str
    codes: tuple[str, ...] = ()


ColdChainEvent = Union[
    SampleCollected, Handoff, TemperatureRecorded, Received, ReviewException
]


@dataclass(frozen=True)
class TemperaturePoint:
    at: datetime
    celsius: float
    excursion: bool
    recorded_by: str = ""


@dataclass(frozen=True)
class ExceptionRecord:
    at: datetime
    reviewer: str
    reason: str
    codes: tuple[str, ...]


# --------------------------------------------------------------------------- #
# Derived state
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ColdChainState:
    sample_id: str
    collected_at: datetime
    temp_min_c: float
    temp_max_c: float
    holding_minutes: float
    lab: str
    collector: str
    holder: str
    custody_breaks: tuple[str, ...] = ()
    evidence_breaks: tuple[str, ...] = ()
    temperatures: tuple[TemperaturePoint, ...] = ()
    exceptions: tuple[ExceptionRecord, ...] = ()
    received_at: datetime | None = None
    accepted: bool = False
    # Replay-only bookkeeping (not part of the decision surface).
    seals: frozenset[str] = frozenset()
    handoff_count: int = 0
    last_handoff_at: datetime | None = None
    version: int = 1

    @property
    def chain_valid(self) -> bool:
        """True when the custody links form an unbroken, sealed path."""
        return not self.custody_breaks

    @property
    def received(self) -> bool:
        return self.received_at is not None

    @property
    def holding_deadline(self) -> datetime:
        return self.collected_at + timedelta(minutes=self.holding_minutes)

    @property
    def holding_breached(self) -> bool:
        """Service-time-limit breach: receipt (if any) after the deadline."""
        return (
            self.received_at is not None
            and self.received_at > self.holding_deadline
        )

    @property
    def temperature_excursions(self) -> tuple[TemperaturePoint, ...]:
        return tuple(t for t in self.temperatures if t.excursion)


def deadline(state: ColdChainState) -> datetime:
    """Return the UTC preservation deadline for the sample."""
    return state.holding_deadline


def _check_sample(event: ColdChainEvent, state: ColdChainState | None) -> None:
    if state is None:
        raise ValueError("chain must start with SampleCollected")
    if event.sample_id != state.sample_id:
        raise ValueError("event belongs to a different sample")


def start(event: SampleCollected) -> ColdChainState:
    """Open a chain from its immutable collection record."""
    if not isinstance(event, SampleCollected):
        raise TypeError("first event must be SampleCollected")
    if event.temp_min_c >= event.temp_max_c:
        raise ValueError("invalid temperature envelope")
    if event.holding_minutes <= 0:
        raise ValueError("holding time must be positive")
    if not event.lab or not event.collector:
        raise ValueError("lab and collector are required")
    return ColdChainState(
        sample_id=event.sample_id,
        collected_at=_utc(event.collected_at),
        temp_min_c=float(event.temp_min_c),
        temp_max_c=float(event.temp_max_c),
        holding_minutes=float(event.holding_minutes),
        lab=event.lab,
        collector=event.collector,
        holder=event.collector,
    )


def apply(state: ColdChainState | None, event: ColdChainEvent) -> ColdChainState:
    """Apply a single event, returning a new state (no mutation)."""
    if isinstance(event, SampleCollected):
        if state is not None:
            raise ValueError("sample already collected")
        return start(event)

    _check_sample(event, state)
    assert state is not None

    if isinstance(event, Handoff):
        at = _utc(event.at)
        breaks = state.custody_breaks
        # Handoff sequences are 1-based and must be gap-free on their own axis.
        if event.seq != state.handoff_count + 1:
            breaks = _append(breaks, "sequence_gap")
        if event.src != state.holder:
            breaks = _append(breaks, "custody_gap")
        if event.src == event.dst:
            breaks = _append(breaks, "self_transfer")
        if at < state.collected_at:
            breaks = _append(breaks, "time_before_collection")
        if state.last_handoff_at is not None and at < state.last_handoff_at:
            breaks = _append(breaks, "time_regression")
        if not event.seal:
            breaks = _append(breaks, "missing_seal")
        elif event.seal in state.seals:
            breaks = _append(breaks, "duplicate_seal")
        seals = state.seals | ({event.seal} if event.seal else set())
        # Custody still advances to the named destination so subsequent links
        # can be evaluated; the break remains recorded as hard evidence.
        return replace(
            state,
            holder=event.dst,
            custody_breaks=breaks,
            seals=seals,
            handoff_count=state.handoff_count + 1,
            last_handoff_at=at,
            version=state.version + 1,
        )

    if isinstance(event, TemperatureRecorded):
        at = _utc(event.at)
        if at < state.collected_at:
            breaks = _append(state.evidence_breaks, "reading_before_collection")
            state = replace(state, evidence_breaks=breaks)
        excursion = not (state.temp_min_c <= event.celsius <= state.temp_max_c)
        point = TemperaturePoint(
            at=at,
            celsius=float(event.celsius),
            excursion=excursion,
            recorded_by=event.recorded_by,
        )
        return replace(
            state,
            temperatures=state.temperatures + (point,),
            version=state.version + 1,
        )

    if isinstance(event, Received):
        at = _utc(event.at)
        breaks = state.evidence_breaks
        if at < state.collected_at:
            breaks = _append(breaks, "received_before_collection")
        if state.received_at is not None:
            breaks = _append(breaks, "duplicate_receipt")
        if state.holder != state.lab:
            breaks = _append(breaks, "lab_not_reached")
        if not event.accepted:
            breaks = _append(breaks, "receipt_rejected")
        return replace(
            state,
            received_at=at,
            accepted=bool(event.accepted),
            evidence_breaks=breaks,
            version=state.version + 1,
        )

    if isinstance(event, ReviewException):
        at = _utc(event.at)
        if at < state.collected_at:
            raise ValueError("exception cannot predate collection")
        if not event.reviewer or not event.reason:
            raise ValueError("exception requires reviewer and reason")
        codes = tuple(dict.fromkeys(event.codes))  # de-duplicate, keep order
        record = ExceptionRecord(
            at=at, reviewer=event.reviewer, reason=event.reason, codes=codes
        )
        return replace(
            state,
            exceptions=state.exceptions + (record,),
            version=state.version + 1,
        )

    raise TypeError("unknown cold-chain event")


def replay(
    collection: SampleCollected, events: Iterable[ColdChainEvent]
) -> ColdChainState:
    """Replay the full evidence log into a deterministic state.

    ``collection`` opens the chain; ``events`` are the subsequent handoffs,
    readings, receipt and exception records in append order.
    """
    state = start(collection)
    for event in events:
        state = apply(state, event)
    return state


# --------------------------------------------------------------------------- #
# Release decision
# --------------------------------------------------------------------------- #
# Soft deviations: recorded problems that a reviewer may accept with an
# exception.  Temperature excursions do not break the sample's analytical
# validity while it is still within its preservation window.
_SOFT_CODES = frozenset({"temperature_excursion"})

# Hard blockers: no exception, however senior, can publish these.
_HARD_RECEIPT_CODES = frozenset(
    {
        "lab_not_reached",
        "receipt_rejected",
        "received_before_collection",
        "duplicate_receipt",
        "reading_before_collection",
    }
)


@dataclass(frozen=True)
class ReleaseDecision:
    sample_id: str
    releaseable: bool
    hard_blocks: tuple[str, ...]
    open_deviations: tuple[str, ...]
    resolved_deviations: tuple[str, ...]
    exceptions: tuple[ExceptionRecord, ...]
    received_at: datetime | None
    holding_deadline: datetime
    holding_breached: bool

    @property
    def blocked(self) -> bool:
        return not self.releaseable

    @property
    def sla_breaches(self) -> tuple[str, ...]:
        """All flagged service-level breaches, hard and soft."""
        return tuple(dict.fromkeys(self.hard_blocks + self.open_deviations))


def _exception_codes(state: ColdChainState) -> frozenset[str]:
    codes: set[str] = set()
    for record in state.exceptions:
        codes.update(record.codes)
    return frozenset(codes)


def decide(state: ColdChainState) -> ReleaseDecision:
    """Decide whether the sample may be published for analysis.

    Hard blocks (invalid chain, evidence integrity problems, or receipt after
    the preservation deadline) always prevent release.  Temperature excursions
    are soft deviations: they hold the sample until a reviewer records an
    exception scoped to ``temperature_excursion``.  Reviewer exceptions can
    never clear a hard block or delete the underlying evidence.
    """
    hard_blocks = tuple(state.custody_breaks)
    hard_blocks += tuple(
        code for code in state.evidence_breaks if code in _HARD_RECEIPT_CODES
    )

    if not state.received:
        hard_blocks = _append(hard_blocks, "not_received")
    if state.holding_breached:
        hard_blocks = _append(hard_blocks, "holding_time_breach")

    temperature_code = "temperature_excursion"
    soft: tuple[str, ...] = (
        (temperature_code,)
        if state.temperature_excursions and temperature_code in _SOFT_CODES
        else ()
    )
    granted = _exception_codes(state)
    resolved = tuple(code for code in soft if code in granted)
    open_deviations = tuple(code for code in soft if code not in granted)

    releaseable = not hard_blocks and not open_deviations

    return ReleaseDecision(
        sample_id=state.sample_id,
        releaseable=releaseable,
        hard_blocks=hard_blocks,
        open_deviations=open_deviations,
        resolved_deviations=resolved,
        exceptions=state.exceptions,
        received_at=state.received_at,
        holding_deadline=state.holding_deadline,
        holding_breached=state.holding_breached,
    )


def is_releaseable(state: ColdChainState) -> bool:
    return decide(state).releaseable


def sla_breaches(state: ColdChainState) -> tuple[str, ...]:
    return decide(state).sla_breaches
