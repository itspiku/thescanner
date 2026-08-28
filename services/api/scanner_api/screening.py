"""Screening: matching reads against watch-lists, and detecting cloned plates.

Two jobs that look similar and are not.

**Screening** asks "is this vehicle on a list?" -- an exact lookup against
active watch-list entries, on the HMAC index so plaintext plates are never
required to match.

**Anomaly detection** asks "is this plate telling the truth?" It uses three
independent signals that fall out of the domain model for free:

1. *Colour disagrees with the class letter.* Legacy plates encode ownership
   twice -- once in the background colour, once in the class letter -- so the
   two disagreeing means one of them was altered. This redundancy exists only
   because of Nepal's legacy scheme, and it disappears as the fleet moves to
   embossed plates.
2. *Class letter disagrees with the vehicle.* A plate whose class says
   *motorcycle* on something the detector sized as a truck is the classic
   cloned-plate signature.
3. *Physically impossible movement.* The same plate read at two sites too far
   apart for the elapsed time means there are two vehicles wearing it.

Signal 3 is the strongest and the only one that needs no extra model, but it
also has the most benign explanations -- a mis-sited camera, a clock that never
synced, a misread digit. So anomalies are raised for human review with the
evidence attached, and are never treated as findings on their own.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Iterable, Sequence

from sqlalchemy import and_, or_, select
from sqlalchemy.orm import Session

from nepal_plate import Ownership, PlateColour, PlateSystem, parse, spec

from .models import Anomaly, Hit, Node, Read, RegistryVehicle, WatchlistEntry, utcnow
from .security import PlateHasher


@dataclass(frozen=True)
class ScreeningConfig:
    #: Confidence band at or above which a hit may trigger an automatic alert.
    #: Anything below raises the hit but marks it non-actionable, so a human
    #: sees it and a machine does not act on it.
    actionable_band: str = "high"


class Screener:
    """Matches reads against the active watch-list."""

    def __init__(self, cfg: ScreeningConfig | None = None) -> None:
        self.cfg = cfg or ScreeningConfig()

    def screen(self, session: Session, read: Read) -> list[Hit]:
        now = utcnow()
        entries = session.scalars(
            select(WatchlistEntry).where(
                WatchlistEntry.plate_hmac == read.plate_hmac,
                WatchlistEntry.active.is_(True),
                or_(WatchlistEntry.expires_at.is_(None), WatchlistEntry.expires_at > now),
            )
        ).all()

        hits: list[Hit] = []
        for entry in entries:
            hit = Hit(
                read_id=read.id,
                watchlist_id=entry.id,
                confidence=read.confidence,
                actionable=read.confidence == self.cfg.actionable_band,
            )
            session.add(hit)
            hits.append(hit)
        if hits:
            session.flush()
        return hits

    def add_entry(
        self,
        session: Session,
        hasher: PlateHasher,
        *,
        plate: str,
        reason: str,
        added_by: str,
        category: str = "general",
        authority_ref: str = "",
        expires_at: datetime | None = None,
    ) -> WatchlistEntry:
        """Add a vehicle of interest.

        The plate is canonicalised through the domain parser first, so an entry
        typed by an officer produces the same key as a read from a camera. This
        is the single most common place a national ANPR silently fails: the
        watch-list says ``BA 1 CHA 1234`` and the camera emits ``बा १ च १२३४``,
        they never match, and nobody notices because a non-match looks exactly
        like an absent vehicle.
        """
        parsed = parse(plate)
        if not parsed.is_valid:
            raise ValueError(
                f"{plate!r} is not a valid Nepali plate: {'; '.join(parsed.errors)}"
            )
        if not reason.strip():
            raise ValueError("a watch-list entry requires a reason")

        entry = WatchlistEntry(
            plate_hmac=hasher.hash(parsed.canonical),
            plate_canonical=parsed.canonical,
            category=category,
            reason=reason.strip(),
            authority_ref=authority_ref,
            added_by=added_by,
            expires_at=expires_at,
        )
        session.add(entry)
        session.flush()
        return entry

    def backfill(self, session: Session, entry: WatchlistEntry, *, days: int = 30) -> list[Hit]:
        """Match a new watch-list entry against recent history.

        Adding a plate to a watch-list should surface where it has already been,
        not only where it goes next. Without this, an investigator adding a
        vehicle at 9am learns nothing about the previous three weeks that are
        sitting in the database.
        """
        since = utcnow() - timedelta(days=days)
        reads = session.scalars(
            select(Read).where(
                Read.plate_hmac == entry.plate_hmac,
                Read.captured_at >= since,
                Read.verified.is_(True),
                Read.provisional.is_(False),
            )
        ).all()
        hits = []
        for read in reads:
            hit = Hit(
                read_id=read.id, watchlist_id=entry.id, confidence=read.confidence,
                actionable=False,  # historical: informational, never an alert
            )
            session.add(hit)
            hits.append(hit)
        if hits:
            session.flush()
        return hits


# ---------------------------------------------------------------------------
# Anomaly detection
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AnomalyConfig:
    #: Fastest plausible average speed between two sites, km/h. Generous: the
    #: point is to catch the physically impossible, not to infer speeding.
    max_speed_kmh: float = 160.0
    #: Ignore pairs closer together than this in time; GPS and clock jitter make
    #: very short intervals meaningless.
    min_interval_seconds: float = 60.0
    lookback_hours: int = 24
    #: Only HIGH-confidence reads feed impossible-movement detection. A misread
    #: digit produces exactly this signal, and building an accusation on a
    #: low-confidence read is how an ANPR generates a wrongful stop.
    require_confidence: str = "high"


class AnomalyDetector:
    """Flags plates whose observations do not hang together."""

    def __init__(self, cfg: AnomalyConfig | None = None) -> None:
        self.cfg = cfg or AnomalyConfig()

    # -- signal 1: colour vs class letter --------------------------------

    def check_colour_consistency(self, read: Read, observed_colour: str | None) -> Anomaly | None:
        """Legacy plates state ownership twice. Disagreement means alteration."""
        if read.plate_system != PlateSystem.DEVANAGARI.value or not observed_colour:
            return None
        try:
            colour = PlateColour(observed_colour)
        except ValueError:
            return None
        expected = spec.LEGACY_COLOUR_OWNERSHIP.get(colour)
        if expected is None or read.ownership == Ownership.UNKNOWN.value:
            return None
        if expected.value == read.ownership:
            return None
        return Anomaly(
            kind="colour_class_mismatch",
            plate_hmac=read.plate_hmac,
            plate_canonical=read.plate_canonical,
            severity="warning",
            detail={
                "observed_colour": colour.value,
                "colour_implies": expected.value,
                "class_letter_implies": read.ownership,
            },
            read_ids=[read.id],
        )

    # -- signal 2: plate class vs observed vehicle -----------------------

    def check_vehicle_consistency(self, read: Read, observed_type: str | None) -> Anomaly | None:
        """A motorcycle plate on a truck is the classic clone signature."""
        if not observed_type or read.size_class == "unknown":
            return None
        compatible = {
            "motorcycle": {"motorcycle", "scooter", "moped"},
            "light": {"car", "jeep", "van", "tempo", "auto", "pickup"},
            "heavy": {"truck", "bus", "lorry", "minibus", "tractor", "excavator", "crane"},
            "any": None,  # tourist and diplomatic plates go on anything
        }.get(read.size_class)
        if compatible is None or observed_type.lower() in compatible:
            return None
        return Anomaly(
            kind="plate_vehicle_mismatch",
            plate_hmac=read.plate_hmac,
            plate_canonical=read.plate_canonical,
            severity="warning",
            detail={
                "plate_size_class": read.size_class,
                "observed_vehicle_type": observed_type,
            },
            read_ids=[read.id],
        )

    # -- signal 3: impossible movement -----------------------------------

    def check_impossible_movement(
        self, session: Session, read: Read
    ) -> Anomaly | None:
        """The same plate in two places too far apart for the time between.

        Requires both cameras to have coordinates. Without them the check is
        skipped rather than guessed at -- an anomaly derived from unknown
        geography is noise, and noise in an alert queue trains operators to
        ignore it.
        """
        if read.confidence != self.cfg.require_confidence:
            return None
        here = session.get(Node, read.node_id)
        if here is None or here.latitude is None or here.longitude is None:
            return None

        since = read.captured_at - timedelta(hours=self.cfg.lookback_hours)
        others = session.scalars(
            select(Read).where(
                Read.plate_hmac == read.plate_hmac,
                Read.id != read.id,
                Read.captured_at >= since,
                Read.captured_at <= read.captured_at,
                Read.confidence == self.cfg.require_confidence,
                Read.verified.is_(True),
                Read.provisional.is_(False),
            )
        ).all()

        for other in others:
            node = session.get(Node, other.node_id)
            if node is None or node.latitude is None or node.longitude is None:
                continue
            if node.node_id == here.node_id:
                continue
            dt = abs((read.captured_at - other.captured_at).total_seconds())
            if dt < self.cfg.min_interval_seconds:
                continue
            km = haversine_km(here.latitude, here.longitude, node.latitude, node.longitude)
            speed = km / (dt / 3600.0)
            if speed <= self.cfg.max_speed_kmh:
                continue
            return Anomaly(
                kind="impossible_movement",
                plate_hmac=read.plate_hmac,
                plate_canonical=read.plate_canonical,
                severity="alert",
                detail={
                    "from_node": node.node_id,
                    "to_node": here.node_id,
                    "distance_km": round(km, 2),
                    "interval_seconds": round(dt, 1),
                    "implied_speed_kmh": round(speed, 1),
                    "threshold_kmh": self.cfg.max_speed_kmh,
                    "note": (
                        "Benign explanations exist: a mis-sited camera, an unsynced "
                        "clock, or a misread digit. Review before acting."
                    ),
                },
                read_ids=[other.id, read.id],
            )
        return None

    # -- registry cross-check --------------------------------------------

    def check_registry(self, session: Session, read: Read) -> Anomaly | None:
        """Compare the read against cached registration details, if available."""
        vehicle = session.get(RegistryVehicle, read.plate_hmac)
        if vehicle is None:
            return None
        mismatches = {}
        if (
            vehicle.registered_province
            and read.plate_system == PlateSystem.EMBOSSED.value
        ):
            parsed = parse(read.plate_canonical)
            if parsed.is_valid and parsed.province != vehicle.registered_province:
                mismatches["province"] = {
                    "plate": parsed.province, "registry": vehicle.registered_province
                }
        if not mismatches:
            return None
        return Anomaly(
            kind="registry_mismatch",
            plate_hmac=read.plate_hmac,
            plate_canonical=read.plate_canonical,
            severity="warning",
            detail=mismatches,
            read_ids=[read.id],
        )

    def run_all(
        self,
        session: Session,
        read: Read,
        *,
        observed_colour: str | None = None,
        observed_vehicle_type: str | None = None,
    ) -> list[Anomaly]:
        found = [
            self.check_colour_consistency(read, observed_colour),
            self.check_vehicle_consistency(read, observed_vehicle_type),
            self.check_impossible_movement(session, read),
            self.check_registry(session, read),
        ]
        out = [a for a in found if a is not None]
        for a in out:
            session.add(a)
        if out:
            session.flush()
        return out


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance.

    Straight-line, not road distance, so it *understates* how far a vehicle had
    to travel and therefore understates the implied speed. That bias is the safe
    direction: it makes the check conservative and reduces false accusations.
    """
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


__all__ = [
    "Screener",
    "ScreeningConfig",
    "AnomalyDetector",
    "AnomalyConfig",
    "haversine_km",
]
