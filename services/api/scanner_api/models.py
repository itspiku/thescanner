"""Database schema.

One PostgreSQL instance carries everything: relational entities, the time-series
of reads (TimescaleDB hypertable), and vehicle appearance vectors (pgvector).
The alternative -- Postgres plus a time-series store plus a vector store -- is
three systems to operate, back up, secure and staff, which for a government
deployment in a country with a small pool of specialist operators costs more
than the marginal performance it buys.

The same models run on SQLite for development and tests. Timescale and pgvector
features are applied conditionally (see ``db.apply_postgres_extensions``), so a
developer needs no infrastructure and production loses nothing.

Two schema decisions carry most of the privacy weight:

**Plates are indexed by keyed HMAC, not by plaintext.** ``plate_hmac`` is what
equality lookups and watch-list matching use. A database disclosure therefore
does not immediately yield a searchable movement history for a named vehicle --
an attacker needs the key as well. The plaintext is retained alongside because
operators must be able to read it, but it is not the index.

**Every table that holds a read carries its retention deadline as a column.**
Retention that lives only in a scheduled job is retention that silently stops
happening. Storing ``expires_at`` per row means the deletion is verifiable, and
a row that has outlived its deadline is visible in a query rather than only in a
cron log.
"""

from __future__ import annotations

import enum
from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    JSON,
    LargeBinary,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.types import TypeDecorator


class Base(DeclarativeBase):
    pass


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class UTCDateTime(TypeDecorator):
    """A timestamp that is always timezone-aware UTC, on every dialect.

    PostgreSQL round-trips ``timestamptz`` faithfully; SQLite silently drops the
    offset and hands back a naive datetime. Left alone, that means the same code
    raises ``can't compare offset-naive and offset-aware datetimes`` on one
    backend and not the other -- and, worse, any comparison that *does* succeed
    is comparing a UTC value against a local-time one.

    In a system where a read's timestamp is legally significant and gets signed
    into an evidence chain, a timezone that depends on the database backend is
    not an inconvenience. So every datetime is normalised to aware UTC on the
    way in and re-attached on the way out.
    """

    impl = DateTime(timezone=True)
    cache_ok = True

    def process_bind_param(self, value: datetime | None, dialect):
        if value is None:
            return None
        if value.tzinfo is None:
            # A naive datetime reaching the database is a bug upstream, but
            # assuming UTC is far safer than assuming the server's local zone.
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    def process_result_value(self, value: datetime | None, dialect):
        if value is None:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)


class Role(str, enum.Enum):
    """Least privilege, four roles.

    ``AUDITOR`` deliberately cannot read reads -- only the access log. An
    oversight role that can also perform the activity it oversees is not
    oversight.
    """

    OPERATOR = "operator"          # live feed, alerts; no bulk search
    INVESTIGATOR = "investigator"  # search and export, always with a reason
    ADMIN = "admin"                # cameras, zones, watch-lists, users
    AUDITOR = "auditor"            # the access log, and nothing else


class Confidence(str, enum.Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    REJECT = "reject"


# ---------------------------------------------------------------------------
# Estate
# ---------------------------------------------------------------------------

class Node(Base):
    """An enrolled edge camera.

    ``public_key`` is what makes a read verifiable, and ``chain_head`` /
    ``chain_sequence`` are what make a *gap* detectable. Without the running
    sequence the platform would accept a node that silently restarted its chain.
    """

    __tablename__ = "nodes"

    node_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    public_key: Mapped[str] = mapped_column(String(128), nullable=False)
    site_id: Mapped[str] = mapped_column(String(128), default="unknown")
    camera_id: Mapped[str | None] = mapped_column(String(128))
    name: Mapped[str] = mapped_column(String(256), default="")
    #: Recorded at commissioning. A read's evidential value depends on knowing
    #: where the camera pointed, and that is not recoverable afterwards.
    mounting_notes: Mapped[str] = mapped_column(Text, default="")
    latitude: Mapped[float | None] = mapped_column(Float)
    longitude: Mapped[float | None] = mapped_column(Float)
    enrolled_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)
    last_seen_at: Mapped[datetime | None] = mapped_column(UTCDateTime)
    chain_head: Mapped[str] = mapped_column(String(64), default="0" * 64)
    chain_sequence: Mapped[int] = mapped_column(Integer, default=0)
    active: Mapped[bool] = mapped_column(Boolean, default=True)

    reads: Mapped[list["Read"]] = relationship(back_populates="node")


class Zone(Base):
    __tablename__ = "zones"

    zone_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    camera_id: Mapped[str] = mapped_column(String(128), index=True)
    name: Mapped[str] = mapped_column(String(256), default="")
    polygon: Mapped[list] = mapped_column(JSON, default=list)
    site_id: Mapped[str] = mapped_column(String(128), default="unknown")


# ---------------------------------------------------------------------------
# Reads
# ---------------------------------------------------------------------------

class Read(Base):
    """One vehicle passage, as reported by one camera.

    Note this is a *passage*, not a frame: the edge agent fuses across the
    track and emits once. That is what keeps national volume in the millions
    rather than the hundreds of millions.
    """

    __tablename__ = "reads"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    node_id: Mapped[str] = mapped_column(ForeignKey("nodes.node_id"), index=True)
    #: Per-node monotonic sequence. Together with node_id this is the natural
    #: key, and the uniqueness constraint below is what makes ingest idempotent
    #: -- a re-sent batch after a lost acknowledgement must not duplicate reads.
    sequence: Mapped[int] = mapped_column(Integer, nullable=False)

    captured_at: Mapped[datetime] = mapped_column(UTCDateTime, index=True)
    received_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)
    expires_at: Mapped[datetime | None] = mapped_column(UTCDateTime, index=True)

    camera_id: Mapped[str] = mapped_column(String(128), index=True)
    site_id: Mapped[str] = mapped_column(String(128), default="unknown", index=True)
    track_id: Mapped[int | None] = mapped_column(Integer)

    #: Keyed HMAC of the canonical plate. This is the index, and the only value
    #: matching ever uses.
    plate_hmac: Mapped[str] = mapped_column(String(64), index=True)
    plate_canonical: Mapped[str] = mapped_column(String(64))
    plate_display: Mapped[str] = mapped_column(String(64), default="")
    plate_system: Mapped[str] = mapped_column(String(32), default="unknown")
    ownership: Mapped[str] = mapped_column(String(32), default="unknown")
    size_class: Mapped[str] = mapped_column(String(32), default="unknown")

    confidence: Mapped[str] = mapped_column(String(16), index=True)
    score: Mapped[float] = mapped_column(Float, default=0.0)
    n_frames: Mapped[int] = mapped_column(Integer, default=0)
    agreement: Mapped[float] = mapped_column(Float, default=0.0)
    #: A provisional read is emitted mid-passage so alerts can fire in time; the
    #: final fused read supersedes it. Kept rather than overwritten, because the
    #: alert that fired was based on the provisional value and the record should
    #: show what the system actually knew at the time.
    provisional: Mapped[bool] = mapped_column(Boolean, default=False)
    superseded_by: Mapped[int | None] = mapped_column(Integer)

    #: Fields where grammar repair overrode the pixels. An operator must be able
    #: to see that the system inferred rather than observed.
    repaired_fields: Mapped[list] = mapped_column(JSON, default=list)
    warnings: Mapped[list] = mapped_column(JSON, default=list)
    alternatives: Mapped[list] = mapped_column(JSON, default=list)

    plate_image_sha256: Mapped[str | None] = mapped_column(String(64))
    context_image_sha256: Mapped[str | None] = mapped_column(String(64))

    # Evidence chain, carried verbatim from the node.
    prev_hash: Mapped[str] = mapped_column(String(64))
    payload_hash: Mapped[str] = mapped_column(String(64))
    signature: Mapped[str] = mapped_column(String(128))
    #: Whether the platform verified the signature and chain link on ingest.
    verified: Mapped[bool] = mapped_column(Boolean, default=False)

    node: Mapped[Node] = relationship(back_populates="reads")

    __table_args__ = (
        UniqueConstraint("node_id", "sequence", name="uq_read_node_sequence"),
        Index("ix_reads_plate_time", "plate_hmac", "captured_at"),
        Index("ix_reads_camera_time", "camera_id", "captured_at"),
    )


class ZoneSession(Base):
    """A vehicle's stay inside a zone: the entry/exit record."""

    __tablename__ = "zone_sessions"

    session_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    zone_id: Mapped[str] = mapped_column(String(128), index=True)
    camera_id: Mapped[str] = mapped_column(String(128), index=True)
    node_id: Mapped[str] = mapped_column(String(128), index=True)
    track_id: Mapped[int | None] = mapped_column(Integer)

    entered_at: Mapped[datetime] = mapped_column(UTCDateTime, index=True)
    exited_at: Mapped[datetime | None] = mapped_column(UTCDateTime)
    dwell_seconds: Mapped[float | None] = mapped_column(Float)
    #: exited / track_lost / timed_out. Never collapse these: "went in and never
    #: came out" is a different fact from "left normally", and at a car park or
    #: border post it is the one worth alerting on.
    close_reason: Mapped[str | None] = mapped_column(String(32), index=True)

    #: Back-filled once the passage has been recognised. Null means the vehicle
    #: entered but was never identified -- recorded, not discarded.
    plate_hmac: Mapped[str | None] = mapped_column(String(64), index=True)
    plate_canonical: Mapped[str | None] = mapped_column(String(64))
    expires_at: Mapped[datetime | None] = mapped_column(UTCDateTime, index=True)

    @property
    def is_open(self) -> bool:
        return self.exited_at is None


class UnreadablePassage(Base):
    """A vehicle that passed but could not be identified.

    Recorded deliberately. A hole in the record is indistinguishable from a
    missed detection, and an investigator needs to know the difference between
    "no vehicle" and "a vehicle we could not read".
    """

    __tablename__ = "unreadable_passages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    node_id: Mapped[str] = mapped_column(String(128), index=True)
    camera_id: Mapped[str] = mapped_column(String(128), index=True)
    captured_at: Mapped[datetime] = mapped_column(UTCDateTime, index=True)
    n_crops: Mapped[int] = mapped_column(Integer, default=0)
    best_crop_quality: Mapped[float] = mapped_column(Float, default=0.0)
    expires_at: Mapped[datetime | None] = mapped_column(UTCDateTime, index=True)


# ---------------------------------------------------------------------------
# Screening
# ---------------------------------------------------------------------------

class WatchlistEntry(Base):
    """A vehicle of interest.

    Watch-lists are among the most sensitive data here -- they reveal who is
    under investigation -- so entries carry an owner, a reason and an expiry.
    An entry with no expiry is a standing surveillance authorisation nobody
    reviews, which is exactly the failure mode oversight exists to prevent.
    """

    __tablename__ = "watchlist"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    plate_hmac: Mapped[str] = mapped_column(String(64), index=True)
    plate_canonical: Mapped[str] = mapped_column(String(64))
    category: Mapped[str] = mapped_column(String(64), default="general", index=True)
    reason: Mapped[str] = mapped_column(Text, nullable=False)
    #: Free-text case or warrant reference tying this entry to an authorisation.
    authority_ref: Mapped[str] = mapped_column(String(128), default="")
    added_by: Mapped[str] = mapped_column(String(128))
    added_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)
    expires_at: Mapped[datetime | None] = mapped_column(UTCDateTime, index=True)
    active: Mapped[bool] = mapped_column(Boolean, default=True, index=True)

    __table_args__ = (
        CheckConstraint("length(reason) > 0", name="ck_watchlist_reason_present"),
    )


class Hit(Base):
    """A read that matched an active watch-list entry."""

    __tablename__ = "hits"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    read_id: Mapped[int] = mapped_column(ForeignKey("reads.id"), index=True)
    watchlist_id: Mapped[int] = mapped_column(ForeignKey("watchlist.id"), index=True)
    matched_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow, index=True)
    confidence: Mapped[str] = mapped_column(String(16))
    #: False when the read was below the confidence threshold for automatic
    #: action, so it was raised for human review instead of alerting.
    actionable: Mapped[bool] = mapped_column(Boolean, default=False)
    acknowledged_by: Mapped[str | None] = mapped_column(String(128))
    acknowledged_at: Mapped[datetime | None] = mapped_column(UTCDateTime)


class Anomaly(Base):
    """A suspected cloned or altered plate.

    Three independent signals, recorded with the evidence that produced them:
    the plate's own class letter disagreeing with the vehicle the detector saw,
    the plate colour disagreeing with the class letter, and movement between
    sites that is physically impossible in the elapsed time.
    """

    __tablename__ = "anomalies"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    kind: Mapped[str] = mapped_column(String(64), index=True)
    plate_hmac: Mapped[str] = mapped_column(String(64), index=True)
    plate_canonical: Mapped[str] = mapped_column(String(64))
    detected_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow, index=True)
    severity: Mapped[str] = mapped_column(String(16), default="info")
    detail: Mapped[dict] = mapped_column(JSON, default=dict)
    read_ids: Mapped[list] = mapped_column(JSON, default=list)
    reviewed_by: Mapped[str | None] = mapped_column(String(128))
    reviewed_at: Mapped[datetime | None] = mapped_column(UTCDateTime)


# ---------------------------------------------------------------------------
# Review queue -- the active-learning loop
# ---------------------------------------------------------------------------

class ReviewItem(Base):
    """A low-confidence read queued for a human.

    This is also the data-collection loop: corrections here become training
    labels on real Nepali imagery, which is the only route out of the cold-start
    problem described in ``docs/research/datasets.md``.
    """

    __tablename__ = "review_queue"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    read_id: Mapped[int] = mapped_column(ForeignKey("reads.id"), unique=True, index=True)
    queued_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow, index=True)
    reason: Mapped[str] = mapped_column(String(64), default="low_confidence")
    machine_plate: Mapped[str] = mapped_column(String(64))
    corrected_plate: Mapped[str | None] = mapped_column(String(64))
    #: True when the human confirmed the machine read was right. Distinct from a
    #: correction, and needed to measure real-world precision.
    confirmed: Mapped[bool | None] = mapped_column(Boolean)
    reviewed_by: Mapped[str | None] = mapped_column(String(128))
    reviewed_at: Mapped[datetime | None] = mapped_column(UTCDateTime)


# ---------------------------------------------------------------------------
# Identity, access and audit
# ---------------------------------------------------------------------------

class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    username: Mapped[str] = mapped_column(String(128), unique=True, index=True)
    full_name: Mapped[str] = mapped_column(String(256), default="")
    role: Mapped[str] = mapped_column(String(32), default=Role.OPERATOR.value)
    password_hash: Mapped[str] = mapped_column(String(256), default="")
    #: Enforced for every role. An unattributable action cannot be audited.
    mfa_enrolled: Mapped[bool] = mapped_column(Boolean, default=False)
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)


class AccessLog(Base):
    """Append-only record of who looked at what, and why.

    The most likely real attack on a system like this is not an external
    intrusion but an insider looking up a spouse, a journalist or a political
    rival. No technical control prevents that outright; what this table does is
    make it *visible*, and make the reason a mandatory field rather than an
    optional one.

    Retention is seven years -- far longer than the reads themselves -- because
    the record of an access must outlive the data that was accessed.
    """

    __tablename__ = "access_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow, index=True)
    actor: Mapped[str] = mapped_column(String(128), index=True)
    role: Mapped[str] = mapped_column(String(32), default="")
    action: Mapped[str] = mapped_column(String(64), index=True)
    target: Mapped[str] = mapped_column(String(256), default="")
    #: Mandatory. Requests without one are rejected before they reach data.
    reason: Mapped[str] = mapped_column(Text, nullable=False)
    result: Mapped[str] = mapped_column(String(32), default="ok")
    n_records: Mapped[int] = mapped_column(Integer, default=0)
    client_ip: Mapped[str] = mapped_column(String(64), default="")

    __table_args__ = (
        CheckConstraint("length(reason) > 0", name="ck_access_reason_present"),
        Index("ix_access_actor_time", "actor", "at"),
    )


class ErasureRequest(Base):
    """A Privacy Act 2075 erasure request, and its execution.

    The execution is itself audited: a right-to-erasure that cannot be shown to
    have been carried out has not been honoured.
    """

    __tablename__ = "erasure_requests"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    plate_hmac: Mapped[str] = mapped_column(String(64), index=True)
    requested_by: Mapped[str] = mapped_column(String(128))
    requested_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)
    authority_ref: Mapped[str] = mapped_column(String(128), default="")
    executed_at: Mapped[datetime | None] = mapped_column(UTCDateTime)
    reads_deleted: Mapped[int] = mapped_column(Integer, default=0)
    sessions_deleted: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(String(32), default="pending", index=True)


# ---------------------------------------------------------------------------
# Media and registry
# ---------------------------------------------------------------------------

class Blob(Base):
    __tablename__ = "blobs"

    sha256: Mapped[str] = mapped_column(String(64), primary_key=True)
    content_type: Mapped[str] = mapped_column(String(64), default="image/jpeg")
    bytes: Mapped[int] = mapped_column(Integer, default=0)
    data: Mapped[bytes | None] = mapped_column(LargeBinary)
    #: Set when the object lives in MinIO/S3 rather than inline. Small crops are
    #: cheaper inline than as separate objects; context frames are not.
    object_key: Mapped[str | None] = mapped_column(String(256))
    stored_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)
    expires_at: Mapped[datetime | None] = mapped_column(UTCDateTime, index=True)


class RegistryVehicle(Base):
    """Cached registration details, if a registry integration is available.

    Deliberately a cache with a pluggable source. Without DoTM access the system
    still works -- it just cannot cross-check the plate against the vehicle, so
    the cloned-plate signal degrades to what the vision model alone can infer.
    """

    __tablename__ = "registry_vehicles"

    plate_hmac: Mapped[str] = mapped_column(String(64), primary_key=True)
    plate_canonical: Mapped[str] = mapped_column(String(64))
    make: Mapped[str] = mapped_column(String(64), default="")
    model: Mapped[str] = mapped_column(String(64), default="")
    colour: Mapped[str] = mapped_column(String(32), default="")
    vehicle_type: Mapped[str] = mapped_column(String(32), default="")
    registered_province: Mapped[int | None] = mapped_column(Integer)
    fetched_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)
    source: Mapped[str] = mapped_column(String(64), default="unknown")


ALL_TABLES = (
    Node, Zone, Read, ZoneSession, UnreadablePassage, WatchlistEntry, Hit,
    Anomaly, ReviewItem, User, AccessLog, ErasureRequest, Blob, RegistryVehicle,
)

__all__ = [
    "Base", "Role", "Confidence", "utcnow",
    "Node", "Zone", "Read", "ZoneSession", "UnreadablePassage",
    "WatchlistEntry", "Hit", "Anomaly", "ReviewItem",
    "User", "AccessLog", "ErasureRequest", "Blob", "RegistryVehicle",
    "ALL_TABLES",
]
