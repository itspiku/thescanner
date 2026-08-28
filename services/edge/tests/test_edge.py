"""Tests for the edge agent.

The tests that matter most here are the ones about *evidence*: that a tampered
read is detected, that a node which restarts does not fork its own chain, and
that an unsent read survives a crash. Those are the properties a prosecution
would rest on, and all three fail silently if they fail at all.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from scanner_evidence import GENESIS, NodeIdentity, SignedEvent, verify, verify_chain
from scanner_edge.queue import EventQueue
from scanner_edge.select import Candidate, select, sharpness
from scanner_edge.tracker import ByteTracker, Detection, TrackState, iou_matrix
from scanner_edge.zones import (
    SessionCloseReason,
    Zone,
    ZoneEngine,
    ZoneEventKind,
    reference_point,
    zones_from_config,
)


# ---------------------------------------------------------------------------
# Identity and the evidence chain
# ---------------------------------------------------------------------------

@pytest.fixture
def identity(tmp_path) -> NodeIdentity:
    return NodeIdentity.load_or_create(tmp_path / "node.key", "TEST-01")


def test_key_is_persisted_and_stable(tmp_path):
    """A node that regenerates its key on restart invalidates its own history."""
    a = NodeIdentity.load_or_create(tmp_path / "k.pem", "N1")
    b = NodeIdentity.load_or_create(tmp_path / "k.pem", "N1")
    assert a.public_key_b64() == b.public_key_b64()


def test_signed_event_verifies(identity):
    ev = identity.sign({"plate": "NP-L:BA-1-CHA-1234"})
    assert verify(ev, identity.public_key_b64())


def test_altered_payload_fails_verification(identity):
    """The core evidential property: a read cannot be changed after signing."""
    ev = identity.sign({"plate": "NP-L:BA-1-CHA-1234", "camera_id": "C1"})
    tampered = SignedEvent(
        **{**ev.to_dict(), "payload": {"plate": "NP-L:BA-1-CHA-9999", "camera_id": "C1"}}
    )
    assert not verify(tampered, identity.public_key_b64())


def test_signature_over_a_mismatched_payload_hash_fails(identity):
    """A valid signature over a header describing a *different* payload proves
    nothing, so the payload hash is checked independently."""
    ev = identity.sign({"a": 1})
    forged = SignedEvent(**{**ev.to_dict(), "payload": {"a": 2}})
    assert not verify(forged, identity.public_key_b64())


def test_chain_links_and_detects_deletion(identity):
    events = [identity.sign({"n": i}) for i in range(6)]
    ok, reason = verify_chain(events, identity.public_key_b64())
    assert ok, reason

    # Remove a middle event: both the gap and the broken link are detectable.
    ok, reason = verify_chain(events[:2] + events[3:], identity.public_key_b64())
    assert not ok
    assert "3" in reason, reason


def test_chain_reports_where_it_broke(identity):
    """"The evidence is invalid" is not useful to a court; naming the read is."""
    events = [identity.sign({"n": i}) for i in range(5)]
    bad = SignedEvent(**{**events[2].to_dict(), "payload": {"n": 99}})
    ok, reason = verify_chain(events[:2] + [bad] + events[3:], identity.public_key_b64())
    assert not ok
    assert str(events[2].sequence) in reason


def test_another_nodes_key_cannot_verify(tmp_path):
    a = NodeIdentity.load_or_create(tmp_path / "a.pem", "A")
    b = NodeIdentity.load_or_create(tmp_path / "b.pem", "B")
    assert not verify(a.sign({"x": 1}), b.public_key_b64())


def test_canonical_serialisation_is_order_independent(identity):
    """Signatures only verify if both sides serialise identically."""
    from scanner_evidence import digest

    assert digest({"a": 1, "b": 2}) == digest({"b": 2, "a": 1})


# ---------------------------------------------------------------------------
# Queue
# ---------------------------------------------------------------------------

def test_queue_persists_and_orders(tmp_path, identity):
    with EventQueue(tmp_path, identity) as q:
        for i in range(5):
            q.append("plate_read", {"n": i})
        pending = q.pending()
        assert [e.payload["n"] for e in pending] == [0, 1, 2, 3, 4]
        assert q.stats().unsent == 5


def test_queue_resumes_the_chain_after_restart(tmp_path):
    """The most dangerous failure in the design: a node that restarts at
    sequence 0 forks its own history and every later read is unverifiable."""
    ident = NodeIdentity.load_or_create(tmp_path / "k.pem", "N")
    with EventQueue(tmp_path, ident) as q:
        for i in range(4):
            q.append("plate_read", {"n": i})

    # Fresh identity object, same key file -- exactly what a restart looks like.
    ident2 = NodeIdentity.load_or_create(tmp_path / "k.pem", "N")
    with EventQueue(tmp_path, ident2) as q2:
        assert ident2.sequence == 4
        q2.append("plate_read", {"n": 4})
        events = list(q2.iter_all())

    assert [e.sequence for e in events] == [1, 2, 3, 4, 5]
    ok, reason = verify_chain(events, ident2.public_key_b64())
    assert ok, reason


def test_acknowledged_events_leave_the_pending_set(tmp_path, identity):
    with EventQueue(tmp_path, identity) as q:
        for i in range(3):
            q.append("plate_read", {"n": i})
        q.mark_sent([1, 2])
        assert [e.sequence for e in q.pending()] == [3]
        assert q.stats().unsent == 1


def test_blobs_are_deduplicated(tmp_path, identity):
    """Frames of one track often yield identical crops."""
    with EventQueue(tmp_path, identity) as q:
        a = q.put_blob(b"same-bytes")
        b = q.put_blob(b"same-bytes")
        c = q.put_blob(b"other-bytes")
        assert a == b != c
        assert q.blob_path(a).read_bytes() == b"same-bytes"


def test_prune_never_discards_unsent_reads(tmp_path, identity):
    """Dropping an undelivered read is data loss. The operator should be told
    the disk is full, not silently lose evidence."""
    with EventQueue(tmp_path, identity) as q:
        q.append("plate_read", {"n": 0})
        q.append("plate_read", {"n": 1})
        q.mark_sent([1])
        # Backdate everything so retention applies.
        old = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
        q._db.execute("UPDATE events SET queued_at=?", (old,))
        q._db.commit()

        n_events, _ = q.prune(keep_days=7)
        assert n_events == 1
        remaining = [e.payload["n"] for e in q.iter_all()]
        assert remaining == [1], "the unsent read must survive"


def test_prune_keeps_blobs_that_surviving_events_reference(tmp_path, identity):
    with EventQueue(tmp_path, identity) as q:
        keep = q.put_blob(b"referenced")
        drop = q.put_blob(b"orphan")
        q.append("plate_read", {"plate_image_sha256": keep})
        _, n_blobs = q.prune(keep_days=7)
        assert n_blobs == 1
        assert q.blob_path(keep) is not None and q.blob_path(keep).is_file()
        assert q.blob_path(drop) is None


# ---------------------------------------------------------------------------
# Tracking
# ---------------------------------------------------------------------------

def _det(x, y, w=60, h=40, score=0.9):
    return Detection(box=(x, y, x + w, y + h), score=score)


def test_iou_matrix():
    m = iou_matrix([(0, 0, 10, 10)], [(0, 0, 10, 10), (100, 100, 110, 110)])
    assert m[0, 0] == pytest.approx(1.0)
    assert m[0, 1] == 0.0


def test_track_is_confirmed_after_min_hits():
    t = ByteTracker(min_hits=3)
    for i in range(2):
        assert not t.update([_det(10 + i * 5, 10)])
    confirmed = t.update([_det(20, 10)])
    assert len(confirmed) == 1
    assert confirmed[0].state is TrackState.CONFIRMED


def test_low_confidence_detections_keep_a_track_alive():
    """The BYTE contribution, and the reason it matters here.

    A vehicle that fragments into three tracks produces three reads of one
    passage, three zone sessions and three watch-list alerts.
    """
    t = ByteTracker(min_hits=2, high_threshold=0.6, low_threshold=0.1)
    for i in range(4):
        t.update([_det(10 + i * 8, 10, score=0.9)])
    track_id = t.tracks[0].track_id

    # Four frames where the vehicle is occluded and only weakly detected.
    for i in range(4, 8):
        t.update([_det(10 + i * 8, 10, score=0.25)])

    assert len(t.tracks) == 1, "the track fragmented"
    assert t.tracks[0].track_id == track_id


def test_only_confident_detections_start_a_track():
    t = ByteTracker(high_threshold=0.6, low_threshold=0.1)
    t.update([_det(10, 10, score=0.3)])
    assert not t.tracks


def test_lost_tracks_are_removed_after_max_age():
    t = ByteTracker(min_hits=1, max_age=3)
    t.update([_det(10, 10)])
    for _ in range(5):
        t.update([])
    assert not t.tracks


def test_direction_is_recovered_from_the_trail():
    t = ByteTracker(min_hits=1)
    for i in range(6):
        t.update([_det(10 + i * 20, 50)])
    d = t.tracks[0].direction()
    assert d is not None
    assert d[0] > 0.9, "should read as travelling right"


# ---------------------------------------------------------------------------
# Zones
# ---------------------------------------------------------------------------

SQUARE = Zone(
    zone_id="z1", name="box",
    polygon=((100, 100), (300, 100), (300, 300), (100, 300)),
    enter_frames=2, exit_frames=2,
)


class _FakeTrack:
    """Minimal stand-in: the engine only needs an id, a box and a direction."""

    def __init__(self, track_id: int, box):
        self.track_id = track_id
        self.box = box

    def direction(self):
        return None


def test_reference_point_is_the_bottom_centre():
    """A centroid puts a bus inside a zone while its wheels are still outside."""
    assert reference_point((100, 100, 200, 400)) == (150.0, 400.0)


def test_entry_and_exit_produce_a_dwell():
    eng = ZoneEngine([SQUARE])
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)

    events = []
    for f in range(4):
        events += eng.update([_FakeTrack(1, (150, 150, 200, 200))],
                             frame_index=f, now=now + timedelta(seconds=f))
    assert [e.kind for e in events] == [ZoneEventKind.ENTRY]

    for f in range(4, 9):
        events += eng.update([_FakeTrack(1, (500, 500, 550, 550))],
                             frame_index=f, now=now + timedelta(seconds=f))
    exits = [e for e in events if e.kind is ZoneEventKind.EXIT]
    assert len(exits) == 1
    s = exits[0].session
    assert s.close_reason is SessionCloseReason.EXITED
    assert s.dwell_seconds() == pytest.approx(4.0, abs=1.5)


def test_boundary_jitter_does_not_produce_spurious_events():
    """A box straddling the edge flickers; without hysteresis that is a burst
    of entries and exits."""
    eng = ZoneEngine([SQUARE])
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    events = []
    inside = (150, 150, 200, 200)
    outside = (150, 150, 200, 320)   # bottom-centre just past the edge
    for f in range(10):
        box = inside if f % 2 == 0 else outside
        events += eng.update([_FakeTrack(1, box)], frame_index=f,
                             now=now + timedelta(seconds=f))
    assert not events, f"hysteresis failed: {[e.kind for e in events]}"


def test_a_vanished_track_is_lost_not_exited():
    """"Went in and never came out" is exactly the event worth alerting on, so
    it must not be reported as a clean exit."""
    eng = ZoneEngine([SQUARE])
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    for f in range(3):
        eng.update([_FakeTrack(1, (150, 150, 200, 200))], frame_index=f, now=now)
    events = eng.update([], frame_index=4, now=now + timedelta(seconds=5))
    assert len(events) == 1
    assert events[0].session.close_reason is SessionCloseReason.TRACK_LOST


def test_session_times_out_after_max_dwell():
    zone = Zone(zone_id="z", name="z", polygon=SQUARE.polygon,
                enter_frames=1, exit_frames=2, max_dwell_seconds=10)
    eng = ZoneEngine([zone])
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    eng.update([_FakeTrack(1, (150, 150, 200, 200))], frame_index=0, now=now)
    events = eng.update([_FakeTrack(1, (150, 150, 200, 200))], frame_index=1,
                        now=now + timedelta(seconds=60))
    assert any(e.session.close_reason is SessionCloseReason.TIMED_OUT for e in events)


def test_occupancy_counts_open_sessions():
    eng = ZoneEngine([SQUARE])
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    for f in range(3):
        eng.update(
            [_FakeTrack(1, (150, 150, 200, 200)), _FakeTrack(2, (200, 200, 250, 250))],
            frame_index=f, now=now,
        )
    assert eng.occupancy()["z1"] == 2


def test_plate_is_back_filled_onto_earlier_sessions():
    """Recognition finishes after the entry event has already been emitted."""
    eng = ZoneEngine([SQUARE])
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    for f in range(3):
        eng.update([_FakeTrack(7, (150, 150, 200, 200))], frame_index=f, now=now)
    touched = eng.attach_plate(7, "NP-L:BA-1-CHA-1234")
    assert touched and all(s.plate == "NP-L:BA-1-CHA-1234" for s in touched)


def test_zone_config_rejects_a_degenerate_polygon():
    with pytest.raises(ValueError):
        zones_from_config([{"zone_id": "z", "polygon": [[0, 0], [1, 1]]}])


# ---------------------------------------------------------------------------
# Crop selection
# ---------------------------------------------------------------------------

def _candidate(frame_index: int, width: int = 90, blur: bool = False) -> Candidate:
    rng = np.random.default_rng(frame_index)
    img = rng.integers(0, 255, (40, width, 3), dtype=np.uint8)
    if blur:
        import cv2

        img = cv2.GaussianBlur(img, (15, 15), 0)
    return Candidate(
        frame_index=frame_index, image=img,
        box=(0.0, 0.0, float(width), 40.0),
        detector_score=0.9, frame_size=(1920, 1080),
    )


def test_sharpness_separates_blurred_from_sharp():
    assert sharpness(_candidate(1).image) > sharpness(_candidate(1, blur=True).image)


def test_selection_spreads_across_the_track():
    """Fusion gains come from independent looks. Adjacent frames share their
    blur and pose, so picking the top-N by quality alone wastes the diversity
    that makes fusion work.
    """
    cands = [_candidate(i) for i in range(40)]
    chosen = select(cands, k=6, diversity_window=6)
    assert len(chosen) == 6
    gaps = [
        b.candidate.frame_index - a.candidate.frame_index
        for a, b in zip(chosen, chosen[1:])
    ]
    assert min(gaps) >= 3, f"selected frames clustered: {gaps}"


def test_selection_prefers_larger_plates():
    small = [_candidate(i, width=30) for i in range(0, 20, 2)]
    large = [_candidate(i, width=120) for i in range(1, 21, 2)]
    chosen = select(small + large, k=5)
    widths = [c.candidate.box[2] for c in chosen]
    assert sum(w > 100 for w in widths) >= 4


def test_selection_drops_unreadably_small_crops():
    assert select([_candidate(i, width=12) for i in range(10)], k=5) == []
