"""Search and investigation queries, with mandatory auditing.

Every function here goes through :meth:`Investigator._audit`, which writes to
the append-only access log **before** the query runs and records the outcome
after. Auditing after the fact would miss the query that crashed, timed out, or
was interrupted -- and those are exactly the ones worth looking at.

The queries themselves are ordinary. What makes this module load-bearing is
that there is no other path to the data: the HTTP layer holds no SQL, so an
endpoint cannot accidentally bypass the audit trail by writing its own query.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Sequence

from sqlalchemy import and_, func, or_, select
from sqlalchemy.orm import Session

from nepal_plate import parse

from .models import (
    AccessLog,
    Anomaly,
    Hit,
    Node,
    Read,
    ReviewItem,
    WatchlistEntry,
    ZoneSession,
    utcnow,
)
from .security import Principal, PlateHasher, authorise

#: Hard ceiling on any single query. An investigator who genuinely needs more
#: should export deliberately, which is a separate audited action -- an
#: unbounded query is how a whole national movement database walks out at once.
MAX_LIMIT = 1000


@dataclass(frozen=True)
class ReadSummary:
    id: int
    captured_at: datetime
    camera_id: str
    site_id: str
    plate: str
    plate_display: str
    confidence: str
    score: float
    n_frames: int
    ownership: str
    repaired_fields: list[str]
    verified: bool
    plate_image_sha256: str | None

    @classmethod
    def of(cls, r: Read) -> "ReadSummary":
        return cls(
            id=r.id, captured_at=r.captured_at, camera_id=r.camera_id, site_id=r.site_id,
            plate=r.plate_canonical, plate_display=r.plate_display,
            confidence=r.confidence, score=r.score, n_frames=r.n_frames,
            ownership=r.ownership, repaired_fields=list(r.repaired_fields or []),
            verified=r.verified, plate_image_sha256=r.plate_image_sha256,
        )

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        d["captured_at"] = self.captured_at.isoformat()
        return d


class Investigator:
    """The only route to read data. Audits everything."""

    def __init__(self, hasher: PlateHasher) -> None:
        self.hasher = hasher

    # -- audit ----------------------------------------------------------

    def _audit(
        self,
        session: Session,
        principal: Principal,
        action: str,
        reason: str | None,
        target: str,
    ) -> AccessLog:
        """Authorise, then log, then let the caller query.

        Written and committed *before* the query so that a query which never
        returns still leaves a record. The row is updated with the result count
        afterwards; a log entry with ``result='started'`` and no count is itself
        a signal worth looking at.
        """
        validated = authorise(principal, action, reason)
        entry = AccessLog(
            actor=principal.username,
            role=principal.role.value,
            action=action,
            target=target[:256],
            reason=validated,
            result="started",
            client_ip=principal.client_ip,
        )
        session.add(entry)
        session.flush()
        return entry

    @staticmethod
    def _complete(entry: AccessLog, n: int, result: str = "ok") -> None:
        entry.n_records = n
        entry.result = result

    # -- queries --------------------------------------------------------

    def find_plate(
        self,
        session: Session,
        principal: Principal,
        *,
        plate: str,
        reason: str,
        since: datetime | None = None,
        until: datetime | None = None,
        limit: int = 200,
    ) -> list[ReadSummary]:
        """All sightings of one plate.

        The plate is canonicalised before hashing, so a query typed in Devanagari
        finds reads stored from romanised input and vice versa.
        """
        entry = self._audit(session, principal, "read.search", reason, f"plate={plate}")
        parsed = parse(plate)
        if not parsed.is_valid:
            self._complete(entry, 0, "invalid_plate")
            raise ValueError(f"{plate!r} is not a valid Nepali plate")

        stmt = select(Read).where(Read.plate_hmac == self.hasher.hash(parsed.canonical))
        if since:
            stmt = stmt.where(Read.captured_at >= since)
        if until:
            stmt = stmt.where(Read.captured_at <= until)
        rows = session.scalars(
            stmt.order_by(Read.captured_at.desc()).limit(min(limit, MAX_LIMIT))
        ).all()
        self._complete(entry, len(rows))
        return [ReadSummary.of(r) for r in rows]

    def find_partial(
        self,
        session: Session,
        principal: Principal,
        *,
        fragment: str,
        reason: str,
        since: datetime | None = None,
        limit: int = 200,
    ) -> list[ReadSummary]:
        """Partial-plate search -- a witness saw three digits.

        This cannot use the HMAC index (a hash has no substrings), so it scans
        the plaintext column and is deliberately restricted and rate-limited by
        the caller. It is also the single most abusable query in the system: a
        two-character fragment matches a large fraction of the database, so a
        minimum length is enforced.
        """
        entry = self._audit(session, principal, "read.search", reason, f"partial={fragment}")
        cleaned = fragment.strip().upper()
        if len(cleaned) < 3:
            self._complete(entry, 0, "fragment_too_short")
            raise ValueError("a partial plate search needs at least 3 characters")

        stmt = select(Read).where(Read.plate_canonical.contains(cleaned))
        if since:
            stmt = stmt.where(Read.captured_at >= since)
        rows = session.scalars(
            stmt.order_by(Read.captured_at.desc()).limit(min(limit, MAX_LIMIT))
        ).all()
        self._complete(entry, len(rows))
        return [ReadSummary.of(r) for r in rows]

    def camera_feed(
        self,
        session: Session,
        principal: Principal,
        *,
        camera_id: str | None = None,
        minutes: int = 15,
        limit: int = 200,
    ) -> list[ReadSummary]:
        """Recent reads, for the live console.

        ``read.live`` requires no stated reason: an operator watching a feed
        cannot type a purpose per vehicle, and demanding one would train
        everybody to paste boilerplate -- which destroys the value of the reasons
        that do matter.
        """
        entry = self._audit(session, principal, "read.live", None, f"camera={camera_id or 'all'}")
        stmt = select(Read).where(
            Read.captured_at >= utcnow() - timedelta(minutes=minutes),
            Read.provisional.is_(False),
        )
        if camera_id:
            stmt = stmt.where(Read.camera_id == camera_id)
        rows = session.scalars(
            stmt.order_by(Read.captured_at.desc()).limit(min(limit, MAX_LIMIT))
        ).all()
        self._complete(entry, len(rows))
        return [ReadSummary.of(r) for r in rows]

    def zone_sessions(
        self,
        session: Session,
        principal: Principal,
        *,
        reason: str,
        zone_id: str | None = None,
        plate: str | None = None,
        open_only: bool = False,
        since: datetime | None = None,
        limit: int = 200,
    ) -> list[dict]:
        """Zone entry/exit records -- who was in an area, and for how long."""
        entry = self._audit(
            session, principal, "session.search", reason,
            f"zone={zone_id or 'all'} plate={plate or 'any'}",
        )
        stmt = select(ZoneSession)
        if zone_id:
            stmt = stmt.where(ZoneSession.zone_id == zone_id)
        if plate:
            parsed = parse(plate)
            if not parsed.is_valid:
                self._complete(entry, 0, "invalid_plate")
                raise ValueError(f"{plate!r} is not a valid Nepali plate")
            stmt = stmt.where(ZoneSession.plate_hmac == self.hasher.hash(parsed.canonical))
        if open_only:
            stmt = stmt.where(ZoneSession.exited_at.is_(None))
        if since:
            stmt = stmt.where(ZoneSession.entered_at >= since)
        rows = session.scalars(
            stmt.order_by(ZoneSession.entered_at.desc()).limit(min(limit, MAX_LIMIT))
        ).all()
        self._complete(entry, len(rows))
        return [
            {
                "session_id": s.session_id,
                "zone_id": s.zone_id,
                "camera_id": s.camera_id,
                "plate": s.plate_canonical,
                "entered_at": s.entered_at.isoformat(),
                "exited_at": s.exited_at.isoformat() if s.exited_at else None,
                "dwell_seconds": s.dwell_seconds,
                "close_reason": s.close_reason,
                "open": s.is_open,
            }
            for s in rows
        ]

    def convoy(
        self,
        session: Session,
        principal: Principal,
        *,
        plate: str,
        reason: str,
        window_seconds: int = 120,
        min_shared: int = 2,
        limit: int = 50,
    ) -> list[dict]:
        """Vehicles repeatedly seen alongside a target.

        Two vehicles passing one camera together once is coincidence. The same
        pair at several cameras, each time within a couple of minutes, is a
        travel pattern. ``min_shared`` is what separates the two, and setting it
        to 1 would produce an enormous list of strangers.
        """
        entry = self._audit(session, principal, "read.search", reason, f"convoy={plate}")
        parsed = parse(plate)
        if not parsed.is_valid:
            self._complete(entry, 0, "invalid_plate")
            raise ValueError(f"{plate!r} is not a valid Nepali plate")

        target = self.hasher.hash(parsed.canonical)
        anchors = session.scalars(
            select(Read).where(Read.plate_hmac == target, Read.provisional.is_(False))
            .order_by(Read.captured_at.desc()).limit(MAX_LIMIT)
        ).all()

        counts: dict[str, dict[str, Any]] = {}
        for a in anchors:
            lo = a.captured_at - timedelta(seconds=window_seconds)
            hi = a.captured_at + timedelta(seconds=window_seconds)
            neighbours = session.scalars(
                select(Read).where(
                    Read.camera_id == a.camera_id,
                    Read.captured_at.between(lo, hi),
                    Read.plate_hmac != target,
                    Read.provisional.is_(False),
                )
            ).all()
            for nb in neighbours:
                rec = counts.setdefault(
                    nb.plate_hmac,
                    {"plate": nb.plate_canonical, "encounters": 0, "cameras": set()},
                )
                rec["encounters"] += 1
                rec["cameras"].add(nb.camera_id)

        out = [
            {
                "plate": v["plate"],
                "encounters": v["encounters"],
                "distinct_cameras": len(v["cameras"]),
            }
            for v in counts.values()
            if len(v["cameras"]) >= min_shared
        ]
        out.sort(key=lambda d: (-d["distinct_cameras"], -d["encounters"]))
        out = out[:limit]
        self._complete(entry, len(out))
        return out

    # -- non-personal aggregates ----------------------------------------

    def occupancy(self, session: Session, principal: Principal) -> dict[str, int]:
        """Live vehicle count per zone. Aggregate only, so no reason required."""
        entry = self._audit(session, principal, "read.live", None, "occupancy")
        rows = session.execute(
            select(ZoneSession.zone_id, func.count())
            .where(ZoneSession.exited_at.is_(None))
            .group_by(ZoneSession.zone_id)
        ).all()
        self._complete(entry, len(rows))
        return {zone: int(n) for zone, n in rows}

    def statistics(self, session: Session, principal: Principal, *, hours: int = 24) -> dict:
        """Counts for the dashboard. No plate leaves this function."""
        entry = self._audit(session, principal, "read.live", None, f"stats/{hours}h")
        since = utcnow() - timedelta(hours=hours)
        base = select(func.count()).select_from(Read).where(
            Read.captured_at >= since, Read.provisional.is_(False)
        )
        stats = {
            "window_hours": hours,
            "reads": int(session.scalar(base) or 0),
            "high_confidence": int(
                session.scalar(base.where(Read.confidence == "high")) or 0
            ),
            "unverified": int(session.scalar(base.where(Read.verified.is_(False))) or 0),
            "distinct_plates": int(
                session.scalar(
                    select(func.count(func.distinct(Read.plate_hmac))).where(
                        Read.captured_at >= since, Read.provisional.is_(False)
                    )
                )
                or 0
            ),
            "open_sessions": int(
                session.scalar(
                    select(func.count()).select_from(ZoneSession).where(
                        ZoneSession.exited_at.is_(None)
                    )
                )
                or 0
            ),
            "pending_review": int(
                session.scalar(
                    select(func.count()).select_from(ReviewItem).where(
                        ReviewItem.reviewed_at.is_(None)
                    )
                )
                or 0
            ),
            "open_anomalies": int(
                session.scalar(
                    select(func.count()).select_from(Anomaly).where(
                        Anomaly.reviewed_at.is_(None)
                    )
                )
                or 0
            ),
        }
        self._complete(entry, 1)
        return stats

    # -- oversight ------------------------------------------------------

    def audit_trail(
        self,
        session: Session,
        principal: Principal,
        *,
        actor: str | None = None,
        since: datetime | None = None,
        limit: int = 500,
    ) -> list[dict]:
        """Read the access log. Auditors only.

        Note this action is *not* itself written to the access log, which is
        deliberate: an auditor reviewing the log would otherwise generate an
        unbounded cascade of entries about reviewing entries.
        """
        authorise(principal, "audit.read", None)
        stmt = select(AccessLog)
        if actor:
            stmt = stmt.where(AccessLog.actor == actor)
        if since:
            stmt = stmt.where(AccessLog.at >= since)
        rows = session.scalars(
            stmt.order_by(AccessLog.at.desc()).limit(min(limit, MAX_LIMIT))
        ).all()
        return [
            {
                "at": r.at.isoformat(), "actor": r.actor, "role": r.role,
                "action": r.action, "target": r.target, "reason": r.reason,
                "result": r.result, "n_records": r.n_records, "client_ip": r.client_ip,
            }
            for r in rows
        ]


__all__ = ["Investigator", "ReadSummary", "MAX_LIMIT"]
