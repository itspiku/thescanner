"""Seed a demo deployment: nodes, reads, zone sessions, watch-list, anomalies.

    python scripts/seed_demo.py --out demo

Produces a working system to look at without needing cameras. Everything is
generated through the *real* path -- plates sampled from the grammar, rendered
and degraded by ``synthplate``, signed with real Ed25519 keys, ingested through
the real ``Ingestor`` -- so the console shows genuine confidence bands, genuine
repaired-field flags and a genuine evidence chain rather than mock data.

That matters more than convenience. Demo data built by inserting rows directly
would hide exactly the integration bugs a demo is supposed to surface, and it
would show an operator a system that behaves better than the real one.
"""

from __future__ import annotations

import argparse
import io
import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nepal_plate import PlateColour  # noqa: E402
from scanner_api.db import Database, DbConfig  # noqa: E402
from scanner_api.ingest import Ingestor  # noqa: E402
from scanner_api.models import Node, Role, User  # noqa: E402
from scanner_api.screening import AnomalyDetector, Screener  # noqa: E402
from scanner_api.security import PlateHasher, hash_password  # noqa: E402
from scanner_evidence import NodeIdentity  # noqa: E402
from synthplate.degrade import degrade  # noqa: E402
from synthplate.render import render  # noqa: E402
from synthplate.sampling import PlateSampler  # noqa: E402

#: Real Kathmandu Valley junctions, so the impossible-movement check has
#: plausible geography to reason about.
SITES = [
    ("KTM-BAL", "Balkumari", 27.6710, 85.3410),
    ("KTM-MAH", "Mahalaxmisthan", 27.6640, 85.3220),
    ("KTM-DHO", "Dhobighat", 27.6790, 85.3060),
    ("KTM-MBH", "Munibhairabh", 27.6880, 85.3550),
]

CONFIDENCE_MIX = [("high", 0.68), ("medium", 0.18), ("low", 0.10), ("reject", 0.04)]


def _band(rng: random.Random) -> str:
    r = rng.random()
    acc = 0.0
    for name, p in CONFIDENCE_MIX:
        acc += p
        if r <= acc:
            return name
    return "high"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default="demo", help="directory for the database and keys")
    ap.add_argument("--reads", type=int, default=400)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--plate-key", default="d" * 64)
    args = ap.parse_args()

    out = Path(args.out)
    (out / "keys").mkdir(parents=True, exist_ok=True)
    db_path = out / "scanner.db"
    if db_path.exists():
        db_path.unlink()

    rng = random.Random(args.seed)
    hasher = PlateHasher(args.plate_key)
    db = Database(DbConfig(url=f"sqlite:///{db_path}"))
    db.init()

    screener = Screener()
    ingestor = Ingestor(hasher, screener=screener)
    detector = AnomalyDetector()
    sampler = PlateSampler(seed=args.seed, strategy="realistic", embossed_fraction=0.35)

    identities = {
        code: NodeIdentity.load_or_create(out / "keys" / f"{code}.key", f"{code}:CAM-1")
        for code, *_ in SITES
    }

    with db.session() as s:
        for username, role in (
            ("operator", Role.OPERATOR),
            ("investigator", Role.INVESTIGATOR),
            ("admin", Role.ADMIN),
            ("auditor", Role.AUDITOR),
        ):
            s.add(User(
                username=username, full_name=username.title(), role=role.value,
                password_hash=hash_password("demo-password-123"), mfa_enrolled=False,
            ))

        for (code, name, lat, lon) in SITES:
            ingestor.enrol(s, {
                **identities[code].enrolment_record(),
                "site_id": code, "camera_id": f"{code}-CAM-1", "name": name,
                "mounting_notes": f"{name} junction, pole-mounted, demo data",
            })
        s.flush()
        for (code, _, lat, lon) in SITES:
            node = s.get(Node, identities[code].node_id)
            node.latitude, node.longitude = lat, lon

        # A watch-list entry typed the way an officer would -- romanised -- so
        # the demo exercises the cross-script match that a real deployment
        # depends on.
        watched = sampler.sample_legacy()
        roman = (
            f"{watched.plate.zone} {watched.plate.lot or ''} "
            f"{watched.plate.vehicle_class} {watched.plate.serial}"
        ).replace("  ", " ")
        screener.add_entry(
            s, hasher, plate=roman,
            reason="demo: stolen vehicle, case 2026/114",
            added_by="investigator", category="stolen",
        )
        print(f"watch-list entry: {roman}  ->  {watched.plate.canonical}")

        now = datetime.now(timezone.utc)
        plates = [sampler.sample() for _ in range(max(1, args.reads // 3))]
        plates.append(watched)

        n_sessions = 0
        for i in range(args.reads):
            code, _, _, _ = SITES[rng.randrange(len(SITES))]
            ident = identities[code]
            sample = plates[rng.randrange(len(plates))]
            captured = now - timedelta(minutes=rng.uniform(0, 24 * 60))

            crop = degrade(render(sample, height=96, rng=rng), rng=rng)
            buf = io.BytesIO()
            crop.image.save(buf, format="JPEG", quality=88)
            sha = None
            band = _band(rng)
            if band != "reject":
                sha = ingestor.store_blob(
                    s,
                    __import__("hashlib").sha256(buf.getvalue()).hexdigest(),
                    __import__("base64").b64encode(buf.getvalue()).decode(),
                )
                sha = sha.sha256

            track_id = 1000 + i
            payload = {
                "kind": "plate_read",
                "camera_id": f"{code}-CAM-1", "site_id": code, "track_id": track_id,
                "plate": sample.plate.canonical, "plate_display": sample.plate.display,
                "system": sample.plate.system.value,
                "ownership": sample.plate.ownership.value,
                "size_class": sample.plate.size_class.value,
                "confidence": band,
                "score": round(rng.uniform(0.5, 0.99), 3),
                "n_frames": rng.randint(3, 18),
                "agreement": round(rng.uniform(0.5, 1.0), 3),
                "provisional": False,
                # A repaired field on some reads, so the console shows the
                # "inferred, not observed" flag it is meant to surface.
                "repaired_fields": ["class"] if rng.random() < 0.08 else [],
                "warnings": [], "alternatives": [],
                "plate_image_sha256": sha,
            }
            events = [ident.sign(payload, captured_at=captured).to_dict()]

            if rng.random() < 0.4:
                entered = captured
                sid = f"{code}-{track_id}"
                zone_common = {
                    "camera_id": f"{code}-CAM-1", "zone_id": f"{code}-junction",
                    "session_id": sid, "track_id": track_id,
                    "entered_at": entered.isoformat().replace("+00:00", "Z"),
                }
                events.append(ident.sign({"kind": "zone_entry", **zone_common},
                                         captured_at=entered).to_dict())
                # Most sessions close; a few stay open, which is what the
                # "still inside" state in the console is for.
                if rng.random() < 0.8:
                    dwell = rng.uniform(20, 900)
                    exited = entered + timedelta(seconds=dwell)
                    events.append(ident.sign({
                        "kind": "zone_exit", **zone_common,
                        "exited_at": exited.isoformat().replace("+00:00", "Z"),
                        "dwell_seconds": round(dwell, 1),
                        "close_reason": "exited" if rng.random() < 0.9 else "track_lost",
                        "plate": sample.plate.canonical,
                    }, captured_at=exited).to_dict())
                n_sessions += 1

            result = ingestor.ingest(s, ident.node_id, events)
            for seq in result.accepted:
                from sqlalchemy import select

                from scanner_api.models import Read

                read = s.scalar(
                    select(Read).where(Read.node_id == ident.node_id, Read.sequence == seq)
                )
                if read is not None and read.verified:
                    detector.run_all(
                        s, read,
                        observed_colour=(
                            PlateColour.BLACK_WHITE.value if rng.random() < 0.03 else None
                        ),
                        observed_vehicle_type="truck" if rng.random() < 0.02 else None,
                    )

    print(f"seeded {args.reads} reads, {n_sessions} zone sessions across {len(SITES)} cameras")
    print()
    print(f"  database   {db_path}")
    print(f"  users      operator / investigator / admin / auditor")
    print(f"  password   demo-password-123")
    print()
    print("Run it with:")
    print(f"  SCANNER_DB_URL=sqlite:///{db_path} \\")
    print(f"  SCANNER_PLATE_KEY={args.plate_key} \\")
    print("  SCANNER_TOKEN_SECRET=$(python -c \"print('s'*64)\") \\")
    print("  scanner-api serve")
    db.dispose()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
