"""Ingest: accepting signed events from edge nodes.

This is the trust boundary. Everything arriving here was produced outside the
platform's control, so the rules are strict:

**Verify before storing, and record the verdict.** Every event's Ed25519
signature and payload hash are checked against the node's enrolled public key.
An event that fails is *stored with ``verified=False``*, not discarded --
silently dropping unverifiable events would let an attacker who can corrupt the
link erase reads at will, and the fact that a node produced something
unverifiable is itself worth knowing. Only verified reads are eligible for
watch-list alerts.

**Idempotent on ``(node_id, sequence)``.** The uplink acknowledges only on
confirmation, so a lost acknowledgement means a batch is re-sent. Duplicates
must be harmless.

**Chain gaps are recorded, never closed.** If an arriving event's sequence skips
ahead of the node's stored head, that gap is a fact -- a lost event, a deletion,
or a node that reset its chain. The platform notes it and continues; it does not
renumber, because renumbering destroys the evidence the chain exists to provide.

**Unknown nodes are refused.** A node must be enrolled with its public key
before its events are accepted. This is what stops fabricated reads from a node
that never existed.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Iterable, Sequence

from sqlalchemy import select
from sqlalchemy.orm import Session

from scanner_evidence import SignedEvent, verify

from .models import (
    Blob,
    Hit,
    Node,
    Read,
    ReviewItem,
    UnreadablePassage,
    ZoneSession,
    utcnow,
)
from .security import PlateHasher


@dataclass(frozen=True)
class RetentionPolicy:
    """Tiered retention. Defaults follow the UK NAS precedent and must be
    confirmed against Nepali policy before deployment."""

    reads_days: int = 365
    sessions_days: int = 365
    images_days: int = 90
    unreadable_days: int = 90
    #: Far longer than the data itself: the record of an access must outlive
    #: the data that was accessed.
    audit_days: int = 365 * 7


@dataclass
class IngestResult:
    accepted: list[int] = field(default_factory=list)
    rejected: list[dict] = field(default_factory=list)
    duplicates: list[int] = field(default_factory=list)
    unverified: list[int] = field(default_factory=list)
    chain_gaps: list[dict] = field(default_factory=list)
    hits: list[int] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "accepted": self.accepted,
            "rejected": self.rejected,
            "duplicates": self.duplicates,
            "unverified": self.unverified,
            "chain_gaps": self.chain_gaps,
            "hits": self.hits,
        }


class UnknownNode(Exception):
    """Events from a node the platform has never enrolled."""


class Ingestor:
    """Applies a batch of signed events to the database."""

    def __init__(
        self,
        hasher: PlateHasher,
        *,
        retention: RetentionPolicy | None = None,
        screener=None,          # screening.Screener, injected to avoid a cycle
        review_below: str = "high",
    ) -> None:
        self.hasher = hasher
        self.retention = retention or RetentionPolicy()
        self.screener = screener
        #: Reads below this confidence band go to the human review queue. That
        #: queue is also the active-learning loop: corrections become training
        #: labels on real imagery.
        self.review_below = review_below

    # -- enrolment ------------------------------------------------------

    def enrol(self, session: Session, record: dict) -> Node:
        """Register or update a node's public key.

        Re-enrolling with a *different* key is refused. A node whose key
        silently changed would invalidate every read it had already produced,
        and an attacker who could re-enrol a key could retroactively make forged
        events verify.
        """
        node_id = str(record["node_id"])
        public_key = str(record["public_key"])
        node = session.get(Node, node_id)
        if node is None:
            node = Node(
                node_id=node_id,
                public_key=public_key,
                site_id=str(record.get("site_id", "unknown")),
                camera_id=record.get("camera_id"),
                name=str(record.get("name", "")),
                mounting_notes=str(record.get("mounting_notes", "")),
            )
            session.add(node)
            return node

        if node.public_key != public_key:
            raise PermissionError(
                f"node {node_id!r} is already enrolled with a different key. "
                f"Rotating a node key invalidates its existing chain and must be "
                f"done deliberately, via key rotation, not by re-enrolment."
            )
        node.last_seen_at = utcnow()
        return node

    # -- events ---------------------------------------------------------

    def ingest(
        self, session: Session, node_id: str, events: Sequence[dict]
    ) -> IngestResult:
        result = IngestResult()
        node = session.get(Node, node_id)
        if node is None:
            raise UnknownNode(
                f"node {node_id!r} is not enrolled; refusing its events"
            )
        if not node.active:
            raise UnknownNode(f"node {node_id!r} is deactivated")

        for raw in events:
            try:
                event = SignedEvent.from_dict(raw)
            except (KeyError, TypeError, ValueError) as exc:
                result.rejected.append({"error": f"malformed event: {exc}"})
                continue

            if event.node_id != node_id:
                result.rejected.append({
                    "sequence": event.sequence,
                    "error": f"event claims node {event.node_id!r} in a {node_id!r} batch",
                })
                continue

            existing = session.scalar(
                select(Read).where(Read.node_id == node_id, Read.sequence == event.sequence)
            )
            if existing is not None:
                # Idempotent: acknowledge so the node stops resending.
                result.duplicates.append(event.sequence)
                result.accepted.append(event.sequence)
                continue

            ok = verify(event, node.public_key)
            if not ok:
                result.unverified.append(event.sequence)

            if event.sequence > node.chain_sequence + 1:
                result.chain_gaps.append({
                    "expected": node.chain_sequence + 1,
                    "received": event.sequence,
                    "missing": event.sequence - node.chain_sequence - 1,
                })

            self._apply(session, node, event, ok, result)

            if event.sequence > node.chain_sequence:
                node.chain_sequence = event.sequence
                node.chain_head = event.event_hash()
            node.last_seen_at = utcnow()
            result.accepted.append(event.sequence)

        return result

    def _apply(
        self, session: Session, node: Node, event: SignedEvent, verified: bool,
        result: IngestResult,
    ) -> None:
        kind = event.payload.get("kind", "")
        handler = {
            "plate_read": self._apply_read,
            "zone_entry": self._apply_zone,
            "zone_exit": self._apply_zone,
            "session_plate": self._apply_session_plate,
            "unreadable_passage": self._apply_unreadable,
        }.get(kind)
        if handler is None:
            result.rejected.append({
                "sequence": event.sequence, "error": f"unknown event kind {kind!r}"
            })
            return
        handler(session, node, event, verified, result)

    def _apply_read(
        self, session: Session, node: Node, event: SignedEvent, verified: bool,
        result: IngestResult,
    ) -> None:
        p = event.payload
        canonical = str(p.get("plate", ""))
        if not canonical:
            result.rejected.append({"sequence": event.sequence, "error": "read has no plate"})
            return

        captured = _parse_ts(event.captured_at)
        read = Read(
            node_id=node.node_id,
            sequence=event.sequence,
            captured_at=captured,
            expires_at=captured + timedelta(days=self.retention.reads_days),
            camera_id=str(p.get("camera_id", node.camera_id or node.node_id)),
            site_id=str(p.get("site_id", node.site_id)),
            track_id=p.get("track_id"),
            plate_hmac=self.hasher.hash(canonical),
            plate_canonical=canonical,
            plate_display=str(p.get("plate_display", "")),
            plate_system=str(p.get("system", "unknown")),
            ownership=str(p.get("ownership", "unknown")),
            size_class=str(p.get("size_class", "unknown")),
            confidence=str(p.get("confidence", "reject")),
            score=float(p.get("score", 0.0)),
            n_frames=int(p.get("n_frames", 0)),
            agreement=float(p.get("agreement", 0.0)),
            provisional=bool(p.get("provisional", False)),
            repaired_fields=list(p.get("repaired_fields", [])),
            warnings=list(p.get("warnings", [])),
            alternatives=list(p.get("alternatives", [])),
            plate_image_sha256=p.get("plate_image_sha256"),
            context_image_sha256=p.get("context_image_sha256"),
            prev_hash=event.prev_hash,
            payload_hash=event.payload_hash,
            signature=event.signature,
            verified=verified,
        )
        session.add(read)
        session.flush()  # need read.id for hits and review items

        # A final read supersedes the provisional one emitted mid-passage. The
        # provisional is kept rather than overwritten: an alert may already have
        # fired on it, and the record should show what the system knew then.
        if not read.provisional and read.track_id is not None:
            prior = session.scalars(
                select(Read).where(
                    Read.node_id == node.node_id,
                    Read.track_id == read.track_id,
                    Read.provisional.is_(True),
                    Read.superseded_by.is_(None),
                )
            ).all()
            for prov in prior:
                prov.superseded_by = read.id

        # Only verified reads may raise an alert. An unverifiable read is a
        # security event, not a sighting.
        if verified and self.screener is not None:
            for hit in self.screener.screen(session, read):
                result.hits.append(hit.id)

        if _band_rank(read.confidence) < _band_rank(self.review_below) and not read.provisional:
            session.add(
                ReviewItem(
                    read_id=read.id,
                    reason=f"confidence_{read.confidence}",
                    machine_plate=canonical,
                )
            )

    def _apply_zone(
        self, session: Session, node: Node, event: SignedEvent, verified: bool,
        result: IngestResult,
    ) -> None:
        p = event.payload
        session_id = str(p.get("session_id", ""))
        if not session_id:
            result.rejected.append({"sequence": event.sequence, "error": "zone event has no session_id"})
            return

        entered = _parse_ts(p.get("entered_at")) or _parse_ts(event.captured_at)
        row = session.get(ZoneSession, session_id)
        if row is None:
            row = ZoneSession(
                session_id=session_id,
                zone_id=str(p.get("zone_id", "")),
                camera_id=str(p.get("camera_id", node.camera_id or node.node_id)),
                node_id=node.node_id,
                track_id=p.get("track_id"),
                entered_at=entered,
                expires_at=entered + timedelta(days=self.retention.sessions_days),
            )
            session.add(row)

        exited = _parse_ts(p.get("exited_at"))
        if exited is not None:
            row.exited_at = exited
            row.dwell_seconds = p.get("dwell_seconds")
            row.close_reason = p.get("close_reason")

        plate = p.get("plate")
        if plate:
            row.plate_canonical = str(plate)
            row.plate_hmac = self.hasher.hash(str(plate))

    def _apply_session_plate(
        self, session: Session, node: Node, event: SignedEvent, verified: bool,
        result: IngestResult,
    ) -> None:
        """Back-fill a plate onto sessions emitted before recognition finished."""
        p = event.payload
        canonical = str(p.get("plate", ""))
        if not canonical:
            return
        digest = self.hasher.hash(canonical)
        for sid in p.get("sessions", []):
            row = session.get(ZoneSession, str(sid))
            if row is not None:
                row.plate_canonical = canonical
                row.plate_hmac = digest

    def _apply_unreadable(
        self, session: Session, node: Node, event: SignedEvent, verified: bool,
        result: IngestResult,
    ) -> None:
        captured = _parse_ts(event.captured_at)
        session.add(
            UnreadablePassage(
                node_id=node.node_id,
                camera_id=str(event.payload.get("camera_id", node.node_id)),
                captured_at=captured,
                n_crops=int(event.payload.get("n_crops", 0)),
                best_crop_quality=float(event.payload.get("best_crop_quality", 0.0)),
                expires_at=captured + timedelta(days=self.retention.unreadable_days),
            )
        )

    # -- blobs ----------------------------------------------------------

    def have_blobs(self, session: Session, digests: Iterable[str]) -> list[str]:
        wanted = list(digests)
        if not wanted:
            return []
        rows = session.scalars(select(Blob.sha256).where(Blob.sha256.in_(wanted))).all()
        return list(rows)

    def store_blob(
        self, session: Session, sha256: str, data_b64: str, content_type: str = "image/jpeg"
    ) -> Blob:
        """Store an image, verifying the digest matches the bytes.

        Trusting the claimed digest would let a node substitute one image for
        another under an existing hash, breaking the link between a read and its
        evidence.
        """
        import hashlib

        raw = base64.b64decode(data_b64)
        actual = hashlib.sha256(raw).hexdigest()
        if actual != sha256:
            raise ValueError(
                f"blob digest mismatch: claimed {sha256[:12]}..., actual {actual[:12]}..."
            )
        existing = session.get(Blob, sha256)
        if existing is not None:
            return existing
        blob = Blob(
            sha256=sha256,
            content_type=content_type,
            bytes=len(raw),
            data=raw,
            expires_at=utcnow() + timedelta(days=self.retention.images_days),
        )
        session.add(blob)
        return blob


_BANDS = {"reject": 0, "low": 1, "medium": 2, "high": 3}


def _band_rank(band: str) -> int:
    return _BANDS.get(band, 0)


def _parse_ts(value) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    s = str(value).replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


__all__ = ["Ingestor", "IngestResult", "RetentionPolicy", "UnknownNode"]
