"""Zone occupancy: when a vehicle entered an area, and when it left.

This is the feature the system exists for. A camera watches a region; the
engine records an **entry** when a vehicle crosses into it and an **exit** when
it crosses out, with the dwell time between.

Three details separate a usable implementation from a naive one.

**Hysteresis at the boundary.** A vehicle whose bounding box straddles a zone
edge will flicker in and out for several frames as the box jitters, producing a
burst of spurious entries and exits. Entry and exit therefore require a run of
consecutive frames on the new side, not a single crossing.

**A reference point, not the whole box.** Testing whether the box centroid is
inside a polygon puts a bus inside a zone while its wheels are still outside.
The engine uses the *bottom-centre* of the box -- roughly where the vehicle
touches the road -- which is the point a human would mean by "where the vehicle
is".

**Unclosed sessions are information, not errors.** A session that never closes
means the vehicle is still inside, or left by an unmonitored exit, or the exit
read was missed. Those are operationally different and all three matter, so
sessions are aged out with an explicit reason rather than silently dropped or
force-closed. For a car park or a border post, "went in and never came out" is
precisely the event worth alerting on.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Iterable, Mapping, Sequence

from .tracker import Box, Track

Point = tuple[float, float]


class ZoneEventKind(str, Enum):
    ENTRY = "entry"
    EXIT = "exit"


class SessionCloseReason(str, Enum):
    EXITED = "exited"
    #: The track ended inside the zone -- the vehicle parked, or tracking was
    #: lost. Distinct from a clean exit and must not be reported as one.
    TRACK_LOST = "track_lost"
    #: Open beyond the maximum plausible dwell. Worth alerting on.
    TIMED_OUT = "timed_out"


@dataclass(frozen=True, slots=True)
class Zone:
    """A polygonal region of the camera's view.

    Coordinates are in frame pixels. A zone belongs to one camera; correlating
    the same physical area across cameras is the platform's job, not the edge
    node's.
    """

    zone_id: str
    name: str
    polygon: tuple[Point, ...]
    #: Maximum plausible dwell. Beyond this a session is timed out and flagged.
    max_dwell_seconds: float = 6 * 3600.0
    #: Frames a vehicle must be consistently inside before entry is declared.
    enter_frames: int = 3
    #: Frames consistently outside before exit is declared. Higher than entry:
    #: a brief occlusion should not read as leaving.
    exit_frames: int = 5

    def contains(self, point: Point) -> bool:
        """Ray-casting point-in-polygon."""
        x, y = point
        poly = self.polygon
        inside = False
        n = len(poly)
        for i in range(n):
            x1, y1 = poly[i]
            x2, y2 = poly[(i + 1) % n]
            if (y1 > y) != (y2 > y):
                x_at = (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1
                if x < x_at:
                    inside = not inside
        return inside


def reference_point(box: Box) -> Point:
    """The point that represents where a vehicle *is*.

    Bottom-centre of the box: approximately the contact patch with the road.
    A centroid would place a tall vehicle inside a zone while its wheels are
    still outside it, which is wrong for anything involving a stop line, a car
    park boundary or a toll gantry.
    """
    x1, _, x2, y2 = box
    return ((x1 + x2) / 2.0, y2)


@dataclass(slots=True)
class ZoneSession:
    """One vehicle's stay inside one zone."""

    session_id: str
    zone_id: str
    track_id: int
    entered_at: datetime
    entry_frame: int
    exited_at: datetime | None = None
    exit_frame: int | None = None
    close_reason: SessionCloseReason | None = None
    #: Set by the pipeline once the plate has been read for this passage.
    plate: str | None = None
    #: Direction of travel at entry, for approach-lane analytics.
    entry_direction: tuple[float, float] | None = None

    @property
    def is_open(self) -> bool:
        return self.exited_at is None

    def dwell_seconds(self, now: datetime | None = None) -> float:
        end = self.exited_at or now or datetime.now(timezone.utc)
        return max(0.0, (end - self.entered_at).total_seconds())

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "zone_id": self.zone_id,
            "track_id": self.track_id,
            "entered_at": self.entered_at.isoformat().replace("+00:00", "Z"),
            "entry_frame": self.entry_frame,
            "exited_at": self.exited_at.isoformat().replace("+00:00", "Z") if self.exited_at else None,
            "exit_frame": self.exit_frame,
            "close_reason": self.close_reason.value if self.close_reason else None,
            "dwell_seconds": round(self.dwell_seconds(), 3) if not self.is_open else None,
            "plate": self.plate,
            "entry_direction": list(self.entry_direction) if self.entry_direction else None,
        }


@dataclass(frozen=True, slots=True)
class ZoneEvent:
    kind: ZoneEventKind
    session: ZoneSession
    at: datetime
    frame_index: int


@dataclass(slots=True)
class _Membership:
    """Per (track, zone) debounce state."""

    inside: bool = False
    inside_run: int = 0
    outside_run: int = 0
    session: ZoneSession | None = None


class ZoneEngine:
    """Tracks zone membership and emits entry/exit events."""

    def __init__(self, zones: Sequence[Zone]) -> None:
        self.zones = {z.zone_id: z for z in zones}
        self._state: dict[tuple[int, str], _Membership] = {}
        self.sessions: dict[str, ZoneSession] = {}

    # -- per frame ------------------------------------------------------

    def update(
        self,
        tracks: Sequence[Track],
        *,
        frame_index: int,
        now: datetime | None = None,
    ) -> list[ZoneEvent]:
        now = now or datetime.now(timezone.utc)
        events: list[ZoneEvent] = []
        live_ids = {t.track_id for t in tracks}

        for track in tracks:
            point = reference_point(track.box)
            for zone_id, zone in self.zones.items():
                key = (track.track_id, zone_id)
                st = self._state.setdefault(key, _Membership())
                if zone.contains(point):
                    st.inside_run += 1
                    st.outside_run = 0
                    if not st.inside and st.inside_run >= zone.enter_frames:
                        st.inside = True
                        session = ZoneSession(
                            session_id=uuid.uuid4().hex,
                            zone_id=zone_id,
                            track_id=track.track_id,
                            entered_at=now,
                            entry_frame=frame_index,
                            entry_direction=track.direction(),
                        )
                        st.session = session
                        self.sessions[session.session_id] = session
                        events.append(ZoneEvent(ZoneEventKind.ENTRY, session, now, frame_index))
                else:
                    st.outside_run += 1
                    st.inside_run = 0
                    if st.inside and st.outside_run >= zone.exit_frames:
                        st.inside = False
                        if st.session is not None and st.session.is_open:
                            ev = self._close(st.session, now, frame_index, SessionCloseReason.EXITED)
                            events.append(ev)
                        st.session = None

        # A track that has disappeared while inside a zone did not "exit" -- it
        # was lost. Recording that distinction honestly is the whole point.
        for (track_id, zone_id), st in list(self._state.items()):
            if track_id in live_ids:
                continue
            if st.inside and st.session is not None and st.session.is_open:
                events.append(
                    self._close(st.session, now, frame_index, SessionCloseReason.TRACK_LOST)
                )
            del self._state[(track_id, zone_id)]

        events.extend(self._time_out(now, frame_index))
        return events

    def _close(
        self, session: ZoneSession, now: datetime, frame_index: int, reason: SessionCloseReason
    ) -> ZoneEvent:
        session.exited_at = now
        session.exit_frame = frame_index
        session.close_reason = reason
        return ZoneEvent(ZoneEventKind.EXIT, session, now, frame_index)

    def _time_out(self, now: datetime, frame_index: int) -> list[ZoneEvent]:
        out: list[ZoneEvent] = []
        for session in self.sessions.values():
            if not session.is_open:
                continue
            zone = self.zones.get(session.zone_id)
            if zone is None:
                continue
            if session.dwell_seconds(now) > zone.max_dwell_seconds:
                out.append(self._close(session, now, frame_index, SessionCloseReason.TIMED_OUT))
        return out

    # -- queries --------------------------------------------------------

    def open_sessions(self, zone_id: str | None = None) -> list[ZoneSession]:
        return [
            s for s in self.sessions.values()
            if s.is_open and (zone_id is None or s.zone_id == zone_id)
        ]

    def occupancy(self) -> dict[str, int]:
        """Current vehicle count per zone -- the live dashboard number."""
        counts = {z: 0 for z in self.zones}
        for s in self.sessions.values():
            if s.is_open:
                counts[s.zone_id] = counts.get(s.zone_id, 0) + 1
        return counts

    def attach_plate(self, track_id: int, plate: str) -> list[ZoneSession]:
        """Attach a plate read to every session for this track.

        Recognition completes at the end of a passage, after fusion, which is
        usually *after* the entry event has already been emitted. So sessions
        are created plate-less and back-filled. The platform reconciles on
        ``session_id``, and a session that never receives a plate is reported as
        an unidentified passage rather than being dropped -- an unread vehicle
        still entered.
        """
        touched = [s for s in self.sessions.values() if s.track_id == track_id]
        for s in touched:
            s.plate = plate
        return touched

    def forget(self, session_ids: Iterable[str]) -> None:
        for sid in session_ids:
            self.sessions.pop(sid, None)


def zones_from_config(entries: Iterable[Mapping]) -> list[Zone]:
    """Build zones from configuration."""
    out: list[Zone] = []
    for e in entries:
        polygon = tuple((float(p[0]), float(p[1])) for p in e["polygon"])
        if len(polygon) < 3:
            raise ValueError(f"zone {e.get('zone_id')!r} needs at least 3 points")
        out.append(
            Zone(
                zone_id=str(e["zone_id"]),
                name=str(e.get("name", e["zone_id"])),
                polygon=polygon,
                max_dwell_seconds=float(e.get("max_dwell_seconds", 6 * 3600)),
                enter_frames=int(e.get("enter_frames", 3)),
                exit_frames=int(e.get("exit_frames", 5)),
            )
        )
    return out


__all__ = [
    "Zone",
    "ZoneEngine",
    "ZoneEvent",
    "ZoneEventKind",
    "ZoneSession",
    "SessionCloseReason",
    "reference_point",
    "zones_from_config",
]
