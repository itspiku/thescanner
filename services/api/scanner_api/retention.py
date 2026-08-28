"""Retention and erasure.

Nepal's Privacy Act 2075 gives data subjects a right to erasure and requires
storage limitation. Both are implemented here, and both are implemented as
*jobs that run regardless of operator action* -- retention that depends on
somebody remembering is not retention.

Two design choices worth stating.

**Every row carries its own deadline.** ``expires_at`` is a column, not a
computation. That means an overdue row is visible in an ordinary query rather
than only in a cron log, so "is retention actually happening?" is a question
anyone can answer without trusting a scheduler.

**Erasure is audited, and the audit outlives the data.** A right-to-erasure that
cannot be shown to have been carried out has not been honoured. The
``ErasureRequest`` row survives the reads it deleted, recording who asked, under
what authority, when it ran, and how many records went — while holding only the
HMAC of the plate, never the plaintext, so the erasure record does not itself
become a register of erased vehicles.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Sequence

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from nepal_plate import parse

from .models import (
    AccessLog,
    Anomaly,
    Blob,
    ErasureRequest,
    Hit,
    Read,
    ReviewItem,
    UnreadablePassage,
    ZoneSession,
    utcnow,
)
from .ingest import RetentionPolicy
from .security import PlateHasher, Principal, authorise


@dataclass
class SweepResult:
    reads: int = 0
    sessions: int = 0
    unreadable: int = 0
    blobs: int = 0
    audit: int = 0

    def to_dict(self) -> dict:
        return self.__dict__.copy()

    @property
    def total(self) -> int:
        return self.reads + self.sessions + self.unreadable + self.blobs + self.audit


class RetentionService:
    def __init__(self, policy: RetentionPolicy | None = None) -> None:
        self.policy = policy or RetentionPolicy()

    # -- scheduled expiry ------------------------------------------------

    def sweep(self, session: Session, *, now: datetime | None = None) -> SweepResult:
        """Delete everything past its retention deadline.

        Ordered so that dependants go before the rows they reference: hits and
        review items reference reads, so a read deleted first would either
        orphan them or trip a foreign key, and on SQLite (where FKs are off by
        default without the pragma) the orphan is the silent outcome.
        """
        now = now or utcnow()
        res = SweepResult()

        stale_reads = session.scalars(
            select(Read.id).where(Read.expires_at.is_not(None), Read.expires_at <= now)
        ).all()
        if stale_reads:
            session.execute(delete(Hit).where(Hit.read_id.in_(stale_reads)))
            session.execute(delete(ReviewItem).where(ReviewItem.read_id.in_(stale_reads)))
            res.reads = session.execute(
                delete(Read).where(Read.id.in_(stale_reads))
            ).rowcount or 0

        res.sessions = session.execute(
            delete(ZoneSession).where(
                ZoneSession.expires_at.is_not(None), ZoneSession.expires_at <= now
            )
        ).rowcount or 0

        res.unreadable = session.execute(
            delete(UnreadablePassage).where(
                UnreadablePassage.expires_at.is_not(None),
                UnreadablePassage.expires_at <= now,
            )
        ).rowcount or 0

        res.blobs = self._sweep_blobs(session, now)

        res.audit = session.execute(
            delete(AccessLog).where(
                AccessLog.at <= now - timedelta(days=self.policy.audit_days)
            )
        ).rowcount or 0

        return res

    def _sweep_blobs(self, session: Session, now: datetime) -> int:
        """Delete expired images that nothing still references.

        Images expire faster than the reads that point at them (90 days against
        365), which is deliberate data minimisation -- the fact of a passage is
        far less intrusive than a photograph of it. So a read outliving its
        image is normal and must not delete the read; only the *reference* goes
        stale, and the UI shows the read without a picture.
        """
        expired = session.scalars(
            select(Blob.sha256).where(Blob.expires_at.is_not(None), Blob.expires_at <= now)
        ).all()
        if not expired:
            return 0

        # Anything a surviving read still points at is kept: a read whose
        # retention was extended for a case must keep its evidence.
        referenced = set(
            session.scalars(
                select(Read.plate_image_sha256).where(
                    Read.plate_image_sha256.in_(expired),
                    Read.expires_at > now,
                )
            ).all()
        ) | set(
            session.scalars(
                select(Read.context_image_sha256).where(
                    Read.context_image_sha256.in_(expired),
                    Read.expires_at > now,
                )
            ).all()
        )
        removable = [s for s in expired if s not in referenced]
        if not removable:
            return 0
        return session.execute(
            delete(Blob).where(Blob.sha256.in_(removable))
        ).rowcount or 0

    def overdue(self, session: Session, *, now: datetime | None = None) -> dict[str, int]:
        """How many rows are past their deadline but still present.

        The health check for retention itself. A non-zero, growing count means
        the sweep is not running, which is a compliance failure that would
        otherwise be invisible until somebody audited.
        """
        now = now or utcnow()
        return {
            "reads": int(session.scalar(
                select(func.count()).select_from(Read).where(
                    Read.expires_at.is_not(None), Read.expires_at <= now
                )
            ) or 0),
            "sessions": int(session.scalar(
                select(func.count()).select_from(ZoneSession).where(
                    ZoneSession.expires_at.is_not(None), ZoneSession.expires_at <= now
                )
            ) or 0),
            "blobs": int(session.scalar(
                select(func.count()).select_from(Blob).where(
                    Blob.expires_at.is_not(None), Blob.expires_at <= now
                )
            ) or 0),
        }

    # -- extension -------------------------------------------------------

    def extend_for_case(
        self,
        session: Session,
        principal: Principal,
        *,
        plate: str,
        reason: str,
        hasher: PlateHasher,
        days: int = 365,
    ) -> int:
        """Hold a vehicle's records beyond normal retention, for a live case.

        Legal holds are the legitimate exception to storage limitation, and they
        need to be explicit, attributable and time-bounded -- an indefinite hold
        is retention policy repealed by the back door.
        """
        authorise(principal, "admin.retention", reason)
        parsed = parse(plate)
        if not parsed.is_valid:
            raise ValueError(f"{plate!r} is not a valid Nepali plate")
        digest = hasher.hash(parsed.canonical)
        new_expiry = utcnow() + timedelta(days=days)

        n = 0
        for read in session.scalars(select(Read).where(Read.plate_hmac == digest)).all():
            read.expires_at = new_expiry
            n += 1
        for s in session.scalars(
            select(ZoneSession).where(ZoneSession.plate_hmac == digest)
        ).all():
            s.expires_at = new_expiry

        session.add(AccessLog(
            actor=principal.username, role=principal.role.value,
            action="admin.retention", target=f"extend:{parsed.canonical}",
            reason=reason, n_records=n, client_ip=principal.client_ip,
        ))
        return n

    # -- erasure ---------------------------------------------------------

    def request_erasure(
        self,
        session: Session,
        principal: Principal,
        *,
        plate: str,
        reason: str,
        hasher: PlateHasher,
        authority_ref: str = "",
    ) -> ErasureRequest:
        authorise(principal, "erasure.request", reason)
        parsed = parse(plate)
        if not parsed.is_valid:
            raise ValueError(f"{plate!r} is not a valid Nepali plate")
        req = ErasureRequest(
            plate_hmac=hasher.hash(parsed.canonical),
            requested_by=principal.username,
            authority_ref=authority_ref,
        )
        session.add(req)
        session.add(AccessLog(
            actor=principal.username, role=principal.role.value,
            action="erasure.request", target="erasure", reason=reason,
            client_ip=principal.client_ip,
        ))
        session.flush()
        return req

    def execute_erasure(self, session: Session, request_id: int) -> ErasureRequest:
        """Carry out an erasure across reads, sessions, hits and review items.

        Deliberately does **not** delete the access log entries recording who
        looked at the data. Those are the record of processing, not the personal
        data itself, and destroying them would erase the accountability trail
        that makes the erasure meaningful.
        """
        req = session.get(ErasureRequest, request_id)
        if req is None:
            raise ValueError(f"no erasure request {request_id}")
        if req.status == "done":
            return req

        read_ids = session.scalars(
            select(Read.id).where(Read.plate_hmac == req.plate_hmac)
        ).all()
        if read_ids:
            session.execute(delete(Hit).where(Hit.read_id.in_(read_ids)))
            session.execute(delete(ReviewItem).where(ReviewItem.read_id.in_(read_ids)))
        req.reads_deleted = session.execute(
            delete(Read).where(Read.plate_hmac == req.plate_hmac)
        ).rowcount or 0
        req.sessions_deleted = session.execute(
            delete(ZoneSession).where(ZoneSession.plate_hmac == req.plate_hmac)
        ).rowcount or 0
        session.execute(delete(Anomaly).where(Anomaly.plate_hmac == req.plate_hmac))

        req.executed_at = utcnow()
        req.status = "done"
        return req


__all__ = ["RetentionService", "SweepResult", "RetentionPolicy"]
