"""Tests for the platform.

The two that matter most:

``test_watchlist_matches_across_scripts`` -- a plate typed by an officer in
romanised form must match a read the camera emitted in Devanagari. This is the
single most common way a national ANPR silently fails: the lists never match,
and a non-match looks exactly like an absent vehicle, so nobody notices.

``test_end_to_end_edge_to_platform`` -- a real edge queue, signed with a real
key, drained through the real uplink into the real API. Everything else here
tests a component in isolation; this one is the only thing that proves the
seams line up.
"""

from __future__ import annotations

import base64
import hashlib
import json
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import select

from nepal_plate import parse
from scanner_evidence import NodeIdentity

from scanner_api.db import Database, DbConfig
from scanner_api.ingest import Ingestor, RetentionPolicy, UnknownNode
from scanner_api.models import (
    AccessLog,
    Anomaly,
    Blob,
    Hit,
    Node,
    Read,
    ReviewItem,
    Role,
    WatchlistEntry,
    ZoneSession,
)
from scanner_api.retention import RetentionService
from scanner_api.screening import AnomalyDetector, Screener, haversine_km
from scanner_api.search import Investigator
from scanner_api.security import (
    PermissionDenied,
    PlateHasher,
    Principal,
    ReasonRequired,
    TokenConfig,
    authorise,
    hash_password,
    issue_token,
    verify_password,
    verify_token,
)

KEY = "k" * 64
PLATE = "बा १ च १२३४"
CANONICAL = "NP-L:BA-1-CHA-1234"


@pytest.fixture
def hasher() -> PlateHasher:
    return PlateHasher(KEY)


@pytest.fixture
def db(tmp_path) -> Database:
    d = Database(DbConfig(url=f"sqlite:///{tmp_path/'t.db'}"))
    d.init()
    yield d
    d.dispose()


@pytest.fixture
def identity(tmp_path) -> NodeIdentity:
    return NodeIdentity.load_or_create(tmp_path / "node.key", "SITE:CAM-1")


@pytest.fixture
def ingestor(hasher) -> Ingestor:
    return Ingestor(hasher, screener=Screener())


def _enrol(session, ingestor, identity, **extra):
    node = ingestor.enrol(session, {**identity.enrolment_record(), **extra})
    session.flush()
    return node


def _read_payload(plate: str = CANONICAL, **over) -> dict:
    return {
        "camera_id": "CAM-1", "site_id": "SITE", "track_id": 7,
        "plate": plate, "plate_display": "बा १ च १२३४",
        "system": "devanagari", "ownership": "private", "size_class": "light",
        "confidence": "high", "score": 0.96, "n_frames": 9, "agreement": 1.0,
        "provisional": False, "repaired_fields": [], "warnings": [],
        "alternatives": [], **over,
    }


# ---------------------------------------------------------------------------
# Security primitives
# ---------------------------------------------------------------------------

def test_plate_hasher_requires_a_real_key():
    with pytest.raises(Exception):
        PlateHasher("short")


def test_hmac_is_stable_and_key_dependent():
    a, b = PlateHasher("a" * 64), PlateHasher("b" * 64)
    assert a.hash(CANONICAL) == a.hash(CANONICAL)
    assert a.hash(CANONICAL) != b.hash(CANONICAL)


def test_auditor_cannot_read_vehicle_data():
    """An oversight role that can perform the activity it oversees is not
    oversight."""
    auditor = Principal("aud", Role.AUDITOR)
    for action in ("read.live", "read.search", "read.image", "session.search"):
        with pytest.raises(PermissionDenied):
            authorise(auditor, action, "checking")
    assert authorise(auditor, "audit.read", None)


def test_nobody_else_can_read_the_audit_log():
    for role in (Role.OPERATOR, Role.INVESTIGATOR, Role.ADMIN):
        with pytest.raises(PermissionDenied):
            authorise(Principal("u", role), "audit.read", None)


def test_search_requires_a_stated_reason():
    inv = Principal("i", Role.INVESTIGATOR)
    with pytest.raises(ReasonRequired):
        authorise(inv, "read.search", None)
    with pytest.raises(ReasonRequired):
        authorise(inv, "read.search", "x")
    assert authorise(inv, "read.search", "case 2026/114 vehicle trace")


def test_live_feed_needs_no_reason():
    """Requiring one per vehicle would train everybody to paste boilerplate,
    which destroys the value of the reasons that do matter."""
    assert authorise(Principal("o", Role.OPERATOR), "read.live", None)


def test_operator_cannot_bulk_search():
    with pytest.raises(PermissionDenied):
        authorise(Principal("o", Role.OPERATOR), "read.search", "curiosity is not a reason")


def test_tokens_round_trip_and_expire():
    cfg = TokenConfig(secret="s" * 32, ttl_minutes=60)
    p = verify_token(cfg, issue_token(cfg, "alice", Role.ADMIN, mfa=True))
    assert p.username == "alice" and p.role is Role.ADMIN and p.mfa

    expired = TokenConfig(secret="s" * 32, ttl_minutes=-1)
    with pytest.raises(Exception):
        verify_token(cfg, issue_token(expired, "alice", Role.ADMIN, mfa=True))


def test_token_signature_cannot_be_forged():
    cfg = TokenConfig(secret="s" * 32)
    token = issue_token(cfg, "alice", Role.OPERATOR, mfa=False)
    # Escalate the role in the body and keep the old signature.
    forged = token.replace("|operator|", "|admin|")
    with pytest.raises(Exception):
        verify_token(cfg, forged)


def test_passwords_use_a_slow_hash():
    h = hash_password("correct horse battery staple")
    assert h.startswith("scrypt$")
    assert verify_password("correct horse battery staple", h)
    assert not verify_password("wrong", h)


# ---------------------------------------------------------------------------
# Ingest
# ---------------------------------------------------------------------------

def test_unknown_node_is_refused(db, ingestor, identity):
    with db.session() as s:
        with pytest.raises(UnknownNode):
            ingestor.ingest(s, identity.node_id, [identity.sign(_read_payload() | {"kind": "plate_read"}).to_dict()])


def test_read_is_stored_and_verified(db, ingestor, identity):
    with db.session() as s:
        _enrol(s, ingestor, identity)
        ev = identity.sign({**_read_payload(), "kind": "plate_read"})
        result = ingestor.ingest(s, identity.node_id, [ev.to_dict()])
        assert result.accepted == [ev.sequence]
        assert not result.unverified

        read = s.scalar(select(Read))
        assert read.plate_canonical == CANONICAL
        assert read.verified
        assert read.plate_hmac == PlateHasher(KEY).hash(CANONICAL)


def test_tampered_read_is_stored_but_flagged_unverified(db, ingestor, identity):
    """Discarding it would let anyone who can corrupt the link erase reads."""
    with db.session() as s:
        _enrol(s, ingestor, identity)
        ev = identity.sign({**_read_payload(), "kind": "plate_read"}).to_dict()
        ev["payload"]["plate"] = "NP-L:BA-1-CHA-9999"

        result = ingestor.ingest(s, identity.node_id, [ev])
        assert result.unverified == [ev["sequence"]]
        read = s.scalar(select(Read))
        assert read is not None, "an unverifiable read must still be recorded"
        assert not read.verified


def test_unverified_reads_never_raise_alerts(db, hasher, identity):
    """An unverifiable read is a security event, not a sighting."""
    ingestor = Ingestor(hasher, screener=Screener())
    with db.session() as s:
        _enrol(s, ingestor, identity)
        Screener().add_entry(
            s, hasher, plate=PLATE, reason="stolen vehicle, case 9", added_by="inv"
        )
        ev = identity.sign({**_read_payload(), "kind": "plate_read"}).to_dict()
        ev["payload"]["camera_id"] = "TAMPERED"
        result = ingestor.ingest(s, identity.node_id, [ev])
        assert result.unverified
        assert not result.hits
        assert s.scalar(select(Hit)) is None


def test_ingest_is_idempotent(db, ingestor, identity):
    """The uplink acknowledges only on confirmation, so a lost ack means the
    batch is re-sent."""
    with db.session() as s:
        _enrol(s, ingestor, identity)
        ev = identity.sign({**_read_payload(), "kind": "plate_read"}).to_dict()
        ingestor.ingest(s, identity.node_id, [ev])
        again = ingestor.ingest(s, identity.node_id, [ev])
        assert again.duplicates == [ev["sequence"]]
        assert len(s.scalars(select(Read)).all()) == 1


def test_chain_gap_is_recorded_not_closed(db, ingestor, identity):
    """A gap is information -- renumbering would destroy the evidence the chain
    exists to provide."""
    with db.session() as s:
        _enrol(s, ingestor, identity)
        first = identity.sign({**_read_payload(), "kind": "plate_read"})
        identity.sign({"dropped": True})          # never delivered
        identity.sign({"dropped": True})
        later = identity.sign({**_read_payload(), "kind": "plate_read"})

        ingestor.ingest(s, identity.node_id, [first.to_dict()])
        result = ingestor.ingest(s, identity.node_id, [later.to_dict()])
        assert result.chain_gaps
        assert result.chain_gaps[0]["missing"] == 2
        assert {r.sequence for r in s.scalars(select(Read)).all()} == {1, 4}


def test_reenrolling_a_different_key_is_refused(db, ingestor, identity, tmp_path):
    """Otherwise an attacker who could re-enrol could make forged events verify."""
    with db.session() as s:
        _enrol(s, ingestor, identity)
        other = NodeIdentity.load_or_create(tmp_path / "other.key", identity.node_id)
        with pytest.raises(PermissionError):
            ingestor.enrol(s, other.enrolment_record())


def test_low_confidence_reads_go_to_the_review_queue(db, ingestor, identity):
    with db.session() as s:
        _enrol(s, ingestor, identity)
        ev = identity.sign({**_read_payload(confidence="low"), "kind": "plate_read"})
        ingestor.ingest(s, identity.node_id, [ev.to_dict()])
        item = s.scalar(select(ReviewItem))
        assert item is not None and item.reason == "confidence_low"


def test_zone_entry_and_exit_build_one_session(db, ingestor, identity):
    with db.session() as s:
        _enrol(s, ingestor, identity)
        t0 = datetime(2026, 8, 1, 10, 0, tzinfo=timezone.utc)
        entry = identity.sign({
            "kind": "zone_entry", "camera_id": "CAM-1", "zone_id": "z1",
            "session_id": "S1", "track_id": 7,
            "entered_at": t0.isoformat().replace("+00:00", "Z"),
        })
        exit_ = identity.sign({
            "kind": "zone_exit", "camera_id": "CAM-1", "zone_id": "z1",
            "session_id": "S1", "track_id": 7,
            "entered_at": t0.isoformat().replace("+00:00", "Z"),
            "exited_at": (t0 + timedelta(minutes=4)).isoformat().replace("+00:00", "Z"),
            "dwell_seconds": 240.0, "close_reason": "exited",
        })
        ingestor.ingest(s, identity.node_id, [entry.to_dict(), exit_.to_dict()])

        row = s.get(ZoneSession, "S1")
        assert row.dwell_seconds == 240.0
        assert row.close_reason == "exited"
        assert not row.is_open


def test_session_plate_is_backfilled(db, ingestor, identity):
    """Recognition finishes after the entry event has already been emitted."""
    with db.session() as s:
        _enrol(s, ingestor, identity)
        ingestor.ingest(s, identity.node_id, [
            identity.sign({
                "kind": "zone_entry", "camera_id": "CAM-1", "zone_id": "z1",
                "session_id": "S9", "track_id": 3,
            }).to_dict(),
            identity.sign({
                "kind": "session_plate", "camera_id": "CAM-1",
                "plate": CANONICAL, "sessions": ["S9"], "confidence": "high",
            }).to_dict(),
        ])
        assert s.get(ZoneSession, "S9").plate_canonical == CANONICAL


def test_unreadable_passage_is_recorded(db, ingestor, identity):
    """A hole in the record is indistinguishable from a missed detection."""
    with db.session() as s:
        _enrol(s, ingestor, identity)
        ingestor.ingest(s, identity.node_id, [
            identity.sign({
                "kind": "unreadable_passage", "camera_id": "CAM-1",
                "n_crops": 5, "best_crop_quality": 0.21,
            }).to_dict()
        ])
        from scanner_api.models import UnreadablePassage

        assert s.scalar(select(UnreadablePassage)) is not None


def test_blob_digest_is_verified(db, ingestor):
    """Trusting the claimed digest would let a node substitute one image for
    another under an existing hash."""
    with db.session() as s:
        data = b"\xff\xd8fake-jpeg"
        good = hashlib.sha256(data).hexdigest()
        ingestor.store_blob(s, good, base64.b64encode(data).decode())
        assert s.get(Blob, good) is not None

        with pytest.raises(ValueError):
            ingestor.store_blob(s, "0" * 64, base64.b64encode(data).decode())


# ---------------------------------------------------------------------------
# Screening
# ---------------------------------------------------------------------------

def test_watchlist_matches_across_scripts(db, hasher, identity):
    """An officer types romanised; the camera emits Devanagari. If these do not
    produce the same key the watch-list silently never fires."""
    screener = Screener()
    ingestor = Ingestor(hasher, screener=screener)
    with db.session() as s:
        _enrol(s, ingestor, identity)
        entry = screener.add_entry(
            s, hasher, plate="BA 1 CHA 1234",
            reason="stolen vehicle, case 2026/114", added_by="inv",
        )
        assert entry.plate_canonical == CANONICAL

        ev = identity.sign({**_read_payload(plate=CANONICAL), "kind": "plate_read"})
        result = ingestor.ingest(s, identity.node_id, [ev.to_dict()])
        assert result.hits, "romanised watch-list entry did not match a Devanagari read"


def test_watchlist_rejects_an_invalid_plate(db, hasher):
    with db.session() as s:
        with pytest.raises(ValueError):
            Screener().add_entry(s, hasher, plate="NOT A PLATE", reason="x" * 10, added_by="i")


def test_watchlist_requires_a_reason(db, hasher):
    with db.session() as s:
        with pytest.raises(ValueError):
            Screener().add_entry(s, hasher, plate=PLATE, reason="   ", added_by="i")


def test_expired_watchlist_entries_do_not_match(db, hasher, identity):
    screener = Screener()
    ingestor = Ingestor(hasher, screener=screener)
    with db.session() as s:
        _enrol(s, ingestor, identity)
        screener.add_entry(
            s, hasher, plate=PLATE, reason="old case, now closed", added_by="inv",
            expires_at=datetime.now(timezone.utc) - timedelta(days=1),
        )
        ev = identity.sign({**_read_payload(), "kind": "plate_read"})
        assert not ingestor.ingest(s, identity.node_id, [ev.to_dict()]).hits


def test_low_confidence_hits_are_raised_but_not_actionable(db, hasher, identity):
    screener = Screener()
    ingestor = Ingestor(hasher, screener=screener)
    with db.session() as s:
        _enrol(s, ingestor, identity)
        screener.add_entry(s, hasher, plate=PLATE, reason="case 2026/200", added_by="i")
        ev = identity.sign({**_read_payload(confidence="medium"), "kind": "plate_read"})
        ingestor.ingest(s, identity.node_id, [ev.to_dict()])
        hit = s.scalar(select(Hit))
        assert hit is not None and not hit.actionable


def test_backfill_surfaces_prior_sightings(db, hasher, identity):
    """Adding a plate should show where it has already been, not only where it
    goes next."""
    screener = Screener()
    ingestor = Ingestor(hasher, screener=screener)
    with db.session() as s:
        _enrol(s, ingestor, identity)
        for _ in range(3):
            ingestor.ingest(s, identity.node_id, [
                identity.sign({**_read_payload(), "kind": "plate_read"}).to_dict()
            ])
        entry = screener.add_entry(s, hasher, plate=PLATE, reason="new case 2026/300", added_by="i")
        historical = screener.backfill(s, entry, days=30)
        assert len(historical) == 3
        assert all(not h.actionable for h in historical), "historical hits must not alert"


# ---------------------------------------------------------------------------
# Anomalies
# ---------------------------------------------------------------------------

def test_haversine_is_sane():
    # Kathmandu to Pokhara, roughly 140 km.
    assert 130 < haversine_km(27.7172, 85.3240, 28.2096, 83.9856) < 160


def test_impossible_movement_is_flagged(db, hasher, tmp_path):
    ingestor = Ingestor(hasher, screener=Screener())
    a = NodeIdentity.load_or_create(tmp_path / "a.key", "KTM:CAM-A")
    b = NodeIdentity.load_or_create(tmp_path / "b.key", "PKR:CAM-B")
    with db.session() as s:
        _enrol(s, ingestor, a)
        _enrol(s, ingestor, b)
        s.get(Node, a.node_id).latitude, s.get(Node, a.node_id).longitude = 27.7172, 85.3240
        s.get(Node, b.node_id).latitude, s.get(Node, b.node_id).longitude = 28.2096, 83.9856
        s.flush()

        t0 = datetime(2026, 8, 1, 12, 0, tzinfo=timezone.utc)
        ingestor.ingest(s, a.node_id, [
            a.sign({**_read_payload(), "kind": "plate_read"}, captured_at=t0).to_dict()
        ])
        # Same plate 140 km away, ten minutes later.
        ingestor.ingest(s, b.node_id, [
            b.sign({**_read_payload(), "kind": "plate_read"},
                   captured_at=t0 + timedelta(minutes=10)).to_dict()
        ])

        read_b = s.scalar(select(Read).where(Read.node_id == b.node_id))
        found = AnomalyDetector().check_impossible_movement(s, read_b)
        assert found is not None
        assert found.kind == "impossible_movement"
        assert found.detail["implied_speed_kmh"] > 160


def test_impossible_movement_ignores_low_confidence(db, hasher, tmp_path):
    """A misread digit produces exactly this signal. Building an accusation on
    a low-confidence read is how an ANPR generates a wrongful stop."""
    ingestor = Ingestor(hasher, screener=Screener())
    a = NodeIdentity.load_or_create(tmp_path / "a.key", "KTM:CAM-A")
    with db.session() as s:
        _enrol(s, ingestor, a)
        s.get(Node, a.node_id).latitude, s.get(Node, a.node_id).longitude = 27.7, 85.3
        s.flush()
        ingestor.ingest(s, a.node_id, [
            a.sign({**_read_payload(confidence="low"), "kind": "plate_read"}).to_dict()
        ])
        read = s.scalar(select(Read))
        assert AnomalyDetector().check_impossible_movement(s, read) is None


def test_colour_class_mismatch_is_flagged(db, ingestor, identity):
    with db.session() as s:
        _enrol(s, ingestor, identity)
        ingestor.ingest(s, identity.node_id, [
            identity.sign({**_read_payload(), "kind": "plate_read"}).to_dict()
        ])
        read = s.scalar(select(Read))
        # Read says private (class च), but the plate photographed black = public.
        found = AnomalyDetector().check_colour_consistency(read, "black_white")
        assert found is not None and found.kind == "colour_class_mismatch"
        assert AnomalyDetector().check_colour_consistency(read, "red_white") is None


def test_plate_vehicle_mismatch_is_flagged(db, ingestor, identity):
    """A motorcycle plate on a truck is the classic clone signature."""
    with db.session() as s:
        _enrol(s, ingestor, identity)
        ingestor.ingest(s, identity.node_id, [
            identity.sign({
                **_read_payload(size_class="motorcycle"), "kind": "plate_read"
            }).to_dict()
        ])
        read = s.scalar(select(Read))
        det = AnomalyDetector()
        assert det.check_vehicle_consistency(read, "truck") is not None
        assert det.check_vehicle_consistency(read, "scooter") is None


# ---------------------------------------------------------------------------
# Search and audit
# ---------------------------------------------------------------------------

def test_search_is_audited_with_the_reason(db, hasher, ingestor, identity):
    inv = Investigator(hasher)
    who = Principal("alice", Role.INVESTIGATOR, client_ip="10.0.0.5")
    with db.session() as s:
        _enrol(s, ingestor, identity)
        ingestor.ingest(s, identity.node_id, [
            identity.sign({**_read_payload(), "kind": "plate_read"}).to_dict()
        ])
        rows = inv.find_plate(s, who, plate="BA 1 CHA 1234", reason="case 2026/114 trace")
        assert len(rows) == 1

        log = s.scalar(select(AccessLog))
        assert log.actor == "alice"
        assert log.reason == "case 2026/114 trace"
        assert log.action == "read.search"
        assert log.n_records == 1
        assert log.client_ip == "10.0.0.5"


def test_a_failed_search_still_leaves_a_record(db, hasher):
    """Auditing after the fact would miss the query that crashed."""
    inv = Investigator(hasher)
    who = Principal("mallory", Role.INVESTIGATOR)
    with db.session() as s:
        with pytest.raises(ValueError):
            inv.find_plate(s, who, plate="GARBAGE", reason="fishing expedition")
        log = s.scalar(select(AccessLog))
        assert log is not None and log.result == "invalid_plate"


def test_partial_search_enforces_a_minimum_length(db, hasher):
    inv = Investigator(hasher)
    who = Principal("alice", Role.INVESTIGATOR)
    with db.session() as s:
        with pytest.raises(ValueError):
            inv.find_partial(s, who, fragment="12", reason="witness statement")


def test_operator_search_is_denied_and_touches_no_data(db, hasher):
    inv = Investigator(hasher)
    with db.session() as s:
        with pytest.raises(PermissionDenied):
            inv.find_plate(s, Principal("bob", Role.OPERATOR), plate=PLATE, reason="curious")


def test_statistics_leak_no_plates(db, hasher, ingestor, identity):
    inv = Investigator(hasher)
    with db.session() as s:
        _enrol(s, ingestor, identity)
        ingestor.ingest(s, identity.node_id, [
            identity.sign({**_read_payload(), "kind": "plate_read"}).to_dict()
        ])
        stats = inv.statistics(s, Principal("o", Role.OPERATOR))
        assert stats["reads"] == 1
        assert CANONICAL not in json.dumps(stats)


# ---------------------------------------------------------------------------
# Retention and erasure
# ---------------------------------------------------------------------------

def test_sweep_deletes_only_expired_rows(db, hasher, ingestor, identity):
    svc = RetentionService()
    with db.session() as s:
        _enrol(s, ingestor, identity)
        ingestor.ingest(s, identity.node_id, [
            identity.sign({**_read_payload(), "kind": "plate_read"}).to_dict()
        ])
        assert svc.sweep(s).reads == 0          # not yet due

        s.scalar(select(Read)).expires_at = datetime.now(timezone.utc) - timedelta(days=1)
        s.flush()
        assert svc.sweep(s).reads == 1
        assert s.scalar(select(Read)) is None


def test_overdue_reports_unswept_rows(db, hasher, ingestor, identity):
    """A growing overdue count means retention stopped running -- a compliance
    failure that is otherwise invisible."""
    svc = RetentionService()
    with db.session() as s:
        _enrol(s, ingestor, identity)
        ingestor.ingest(s, identity.node_id, [
            identity.sign({**_read_payload(), "kind": "plate_read"}).to_dict()
        ])
        s.scalar(select(Read)).expires_at = datetime.now(timezone.utc) - timedelta(days=1)
        s.flush()
        assert svc.overdue(s)["reads"] == 1


def test_erasure_removes_reads_and_sessions_but_keeps_the_audit_trail(
    db, hasher, ingestor, identity
):
    """Destroying the record of who looked at the data would erase the
    accountability that makes the erasure meaningful."""
    svc = RetentionService()
    inv = Investigator(hasher)
    admin = Principal("admin", Role.ADMIN)
    with db.session() as s:
        _enrol(s, ingestor, identity)
        ingestor.ingest(s, identity.node_id, [
            identity.sign({**_read_payload(), "kind": "plate_read"}).to_dict()
        ])
        inv.find_plate(s, Principal("alice", Role.INVESTIGATOR),
                       plate=PLATE, reason="case 2026/114 trace")
        n_log_before = len(s.scalars(select(AccessLog)).all())

        req = svc.request_erasure(s, admin, plate=PLATE,
                                  reason="privacy act erasure request 12",
                                  hasher=hasher)
        done = svc.execute_erasure(s, req.id)
        assert done.reads_deleted == 1
        assert s.scalar(select(Read)) is None
        assert len(s.scalars(select(AccessLog)).all()) >= n_log_before


def test_legal_hold_extends_retention(db, hasher, ingestor, identity):
    svc = RetentionService()
    with db.session() as s:
        _enrol(s, ingestor, identity)
        ingestor.ingest(s, identity.node_id, [
            identity.sign({**_read_payload(), "kind": "plate_read"}).to_dict()
        ])
        n = svc.extend_for_case(
            s, Principal("admin", Role.ADMIN), plate=PLATE,
            reason="live prosecution 2026/114", hasher=hasher, days=730,
        )
        assert n == 1
        assert s.scalar(select(Read)).expires_at > datetime.now(timezone.utc) + timedelta(days=700)


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------

def test_end_to_end_edge_to_platform(tmp_path, monkeypatch):
    """A real edge queue, drained through the real uplink, into the real API.

    Everything else here tests a component in isolation. This is the only thing
    that proves the seams line up: the edge's canonical serialisation, the
    uplink's batching, the API's verification, and the domain layer's
    canonicalisation all have to agree, and any one of them drifting is silent.
    """
    from fastapi.testclient import TestClient

    from scanner_api.app import Settings, create_app
    from scanner_edge.queue import EventQueue
    from scanner_edge.uplink import Uplink, UplinkConfig

    monkeypatch.setenv("SCANNER_TOKEN_SECRET", "t" * 64)
    monkeypatch.setenv("SCANNER_NODE_TOKEN", "n" * 64)
    monkeypatch.setenv("SCANNER_DB_URL", f"sqlite:///{tmp_path/'e2e.db'}")

    database = Database(DbConfig(url=f"sqlite:///{tmp_path/'e2e.db'}"))
    app = create_app(settings=Settings(), hasher=PlateHasher(KEY), database=database)
    client = TestClient(app)

    identity = NodeIdentity.load_or_create(tmp_path / "e2e.key", "SITE:CAM-E2E")
    queue = EventQueue(tmp_path / "q", identity)

    def transport(path, body, headers):
        r = client.post(path, content=body, headers=headers)
        return r.status_code, r.content

    up = Uplink(
        UplinkConfig(base_url="", node_id=identity.node_id, token="n" * 64),
        queue,
        transport=transport,
    )

    with TestClient(app):  # trigger lifespan so the schema exists
        up.enrol()

        # A watch-list entry typed in romanised form, before any read arrives.
        with database.session() as s:
            Screener().add_entry(
                s, PlateHasher(KEY), plate="BA 1 CHA 1234",
                reason="stolen vehicle case 2026/114", added_by="inv",
            )

        queue.append("plate_read", _read_payload())
        queue.append("zone_entry", {
            "camera_id": "CAM-1", "zone_id": "z1", "session_id": "E2E-1", "track_id": 7,
        })
        assert queue.stats().unsent == 2

        sent = up.drain()
        assert sent == 2
        assert queue.stats().unsent == 0, "acknowledged events must leave the queue"

        with database.session() as s:
            read = s.scalar(select(Read))
            assert read is not None
            assert read.plate_canonical == CANONICAL
            assert read.verified, "a genuinely signed read must verify at the platform"
            assert s.get(ZoneSession, "E2E-1") is not None
            assert s.scalar(select(Hit)) is not None, (
                "a romanised watch-list entry must match a read that travelled "
                "the full edge-to-platform path"
            )

    queue.close()
    database.dispose()
