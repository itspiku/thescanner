"""HTTP layer.

Thin by design: every endpoint authenticates, then delegates to
:mod:`scanner_api.search`, :mod:`scanner_api.ingest`, :mod:`scanner_api.screening`
or :mod:`scanner_api.retention`. **No SQL lives here.** That is the mechanism
that makes the audit trail unbypassable -- an endpoint cannot accidentally read
personal data without going through the audited query layer, because it has no
way to query at all.

Two authentication paths, deliberately separate:

* **Nodes** present a deployment token and a node id. This authenticates the
  *transport*. It is not what makes reads trustworthy -- the per-event Ed25519
  signatures do that -- so a leaked node token lets an attacker send data, but
  not forge reads that verify.
* **Users** present a bearer token carrying a role. Local HMAC tokens here; a
  production deployment fronts this with OIDC and mandatory MFA.
"""

# NB: deliberately no `from __future__ import annotations`.
#
# FastAPI resolves endpoint annotations with get_type_hints(), which looks names
# up in module globals. The dependency aliases below (`UserDep`, `Db`) are
# defined inside create_app(), so with string annotations they are unresolvable
# and FastAPI silently falls back to treating them as query parameters -- every
# endpoint then 422s asking for a `session` query argument.

import os
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Annotated, Any

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, Response
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from .db import Database, DbConfig
from .ingest import Ingestor, RetentionPolicy, UnknownNode
from .models import Anomaly, Blob, Hit, Node, Read, ReviewItem, Role, WatchlistEntry, utcnow
from .retention import RetentionService
from .screening import AnomalyDetector, Screener
from .search import Investigator
from .security import (
    PermissionDenied,
    PlateHasher,
    Principal,
    ReasonRequired,
    SecurityError,
    TokenConfig,
    authorise,
    hash_password,
    issue_token,
    verify_password,
    verify_token,
)


class Settings:
    """Runtime configuration, from the environment.

    The service refuses to start without a plate key and a token secret. A
    default for either would make every deployment's pseudonyms and tokens
    identical, which is worse than having none because it looks like security.
    """

    def __init__(self) -> None:
        self.db_url = os.environ.get("SCANNER_DB_URL", "sqlite:///./scanner.db")
        self.node_token = os.environ.get("SCANNER_NODE_TOKEN", "")
        secret = os.environ.get("SCANNER_TOKEN_SECRET", "")
        if not secret:
            raise RuntimeError(
                "SCANNER_TOKEN_SECRET is not set. Generate one with "
                "'python -m scanner_api.cli genkey'."
            )
        self.tokens = TokenConfig(secret=secret)
        self.retention = RetentionPolicy()


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class EnrolRequest(BaseModel):
    node_id: str
    public_key: str
    algorithm: str = "Ed25519"
    site_id: str = "unknown"
    camera_id: str | None = None
    name: str = ""
    mounting_notes: str = ""
    latitude: float | None = None
    longitude: float | None = None


class IngestRequest(BaseModel):
    node_id: str
    events: list[dict[str, Any]]


class BlobCheckRequest(BaseModel):
    sha256: list[str]


class BlobRequest(BaseModel):
    sha256: str
    data: str
    content_type: str = "image/jpeg"


class LoginRequest(BaseModel):
    username: str
    password: str


class WatchlistRequest(BaseModel):
    plate: str
    reason: str = Field(min_length=8)
    category: str = "general"
    authority_ref: str = ""
    expires_in_days: int | None = 90
    backfill_days: int = 30


class ReviewRequest(BaseModel):
    corrected_plate: str | None = None
    confirmed: bool = False


class ErasureRequestBody(BaseModel):
    plate: str
    reason: str = Field(min_length=8)
    authority_ref: str = ""


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

def create_app(
    *,
    settings: Settings | None = None,
    hasher: PlateHasher | None = None,
    database: Database | None = None,
) -> FastAPI:
    settings = settings or Settings()
    hasher = hasher or PlateHasher()
    db = database or Database(DbConfig(url=settings.db_url))

    screener = Screener()
    ingestor = Ingestor(hasher, retention=settings.retention, screener=screener)
    anomalies = AnomalyDetector()
    investigator = Investigator(hasher)
    retention = RetentionService(settings.retention)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        db.init()
        yield
        db.dispose()

    app = FastAPI(
        title="TheScanner platform API",
        version="0.1.0",
        summary="Vehicle movement intelligence for Nepal",
        lifespan=lifespan,
    )
    app.state.db = db
    app.state.hasher = hasher
    app.state.settings = settings

    # -- dependencies ---------------------------------------------------

    def get_session() -> Session:
        s = db.session_factory()
        try:
            yield s
            s.commit()
        except Exception:
            s.rollback()
            raise
        finally:
            s.close()

    def current_user(
        request: Request,
        authorization: Annotated[str | None, Header()] = None,
    ) -> Principal:
        if not authorization or not authorization.lower().startswith("bearer "):
            raise HTTPException(401, "missing bearer token")
        try:
            return verify_token(
                settings.tokens,
                authorization[7:],
                client_ip=request.client.host if request.client else "",
            )
        except SecurityError as exc:
            raise HTTPException(401, str(exc)) from exc

    def node_auth(
        authorization: Annotated[str | None, Header()] = None,
        x_node_id: Annotated[str | None, Header()] = None,
    ) -> str:
        if settings.node_token:
            if not authorization or authorization[7:] != settings.node_token:
                raise HTTPException(401, "invalid node token")
        if not x_node_id:
            raise HTTPException(400, "X-Node-Id header is required")
        return x_node_id

    UserDep = Annotated[Principal, Depends(current_user)]
    Db = Annotated[Session, Depends(get_session)]

    def guard(fn):
        """Translate security errors into the right HTTP status.

        403 for "you may not", 400 for "you did not say why". Distinguishing
        them matters: a caller who omitted a reason should be told to supply
        one, not told they lack permission.
        """
        try:
            return fn()
        except PermissionDenied as exc:
            raise HTTPException(403, str(exc)) from exc
        except ReasonRequired as exc:
            raise HTTPException(400, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    # -- health ---------------------------------------------------------

    @app.get("/healthz", tags=["ops"])
    def healthz(session: Db) -> dict:
        overdue = retention.overdue(session)
        return {
            "status": "ok",
            # A growing overdue count means retention has stopped running,
            # which is a compliance failure that is otherwise invisible.
            "retention_overdue": overdue,
            "retention_healthy": sum(overdue.values()) == 0,
        }

    # -- auth -----------------------------------------------------------

    @app.post("/api/v1/auth/token", tags=["auth"])
    def login(body: LoginRequest, session: Db) -> dict:
        from .models import User

        user = session.scalar(select(User).where(User.username == body.username))
        if user is None or not user.active or not verify_password(body.password, user.password_hash):
            # Identical response for unknown user and wrong password: telling
            # them apart is a free username-enumeration oracle.
            raise HTTPException(401, "invalid credentials")
        role = Role(user.role)
        return {
            "token": issue_token(settings.tokens, user.username, role, mfa=user.mfa_enrolled),
            "role": role.value,
            "mfa_enrolled": user.mfa_enrolled,
            "expires_in_minutes": settings.tokens.ttl_minutes,
        }

    # -- node ingest ----------------------------------------------------

    @app.post("/api/v1/nodes/enrol", tags=["ingest"])
    def enrol(body: EnrolRequest, session: Db, node_id: Annotated[str, Depends(node_auth)]) -> dict:
        try:
            node = ingestor.enrol(session, body.model_dump())
        except PermissionError as exc:
            raise HTTPException(409, str(exc)) from exc
        if body.latitude is not None:
            node.latitude = body.latitude
            node.longitude = body.longitude
        # enrolled_at is a Python-side column default, so it is not populated
        # until the row is flushed.
        session.flush()
        return {"node_id": node.node_id, "enrolled_at": node.enrolled_at.isoformat()}

    @app.post("/api/v1/ingest/events", tags=["ingest"])
    def ingest_events(
        body: IngestRequest, session: Db, node_id: Annotated[str, Depends(node_auth)]
    ) -> dict:
        if body.node_id != node_id:
            raise HTTPException(403, "node_id does not match the authenticated node")
        try:
            result = ingestor.ingest(session, body.node_id, body.events)
        except UnknownNode as exc:
            raise HTTPException(403, str(exc)) from exc

        # Anomaly detection runs after the batch is applied, so that
        # impossible-movement checks can see reads from the same batch.
        for seq in result.accepted:
            read = session.scalar(
                select(Read).where(Read.node_id == body.node_id, Read.sequence == seq)
            )
            if read is not None and read.verified and not read.provisional:
                anomalies.run_all(session, read)
        return result.to_dict()

    @app.post("/api/v1/ingest/blobs/check", tags=["ingest"])
    def blobs_check(
        body: BlobCheckRequest, session: Db, node_id: Annotated[str, Depends(node_auth)]
    ) -> dict:
        return {"have": ingestor.have_blobs(session, body.sha256)}

    @app.post("/api/v1/ingest/blobs", tags=["ingest"])
    def blobs_put(
        body: BlobRequest, session: Db, node_id: Annotated[str, Depends(node_auth)]
    ) -> dict:
        try:
            blob = ingestor.store_blob(session, body.sha256, body.data, body.content_type)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        return {"sha256": blob.sha256, "bytes": blob.bytes}

    # -- reads ----------------------------------------------------------

    @app.get("/api/v1/reads/live", tags=["reads"])
    def reads_live(
        user: UserDep, session: Db,
        camera_id: str | None = None, minutes: int = Query(15, ge=1, le=180),
        limit: int = Query(100, ge=1, le=1000),
    ) -> list[dict]:
        return [
            r.to_dict()
            for r in guard(lambda: investigator.camera_feed(
                session, user, camera_id=camera_id, minutes=minutes, limit=limit
            ))
        ]

    @app.get("/api/v1/reads/search", tags=["reads"])
    def reads_search(
        user: UserDep, session: Db,
        plate: str, reason: str,
        since: datetime | None = None, until: datetime | None = None,
        limit: int = Query(200, ge=1, le=1000),
    ) -> list[dict]:
        return [
            r.to_dict()
            for r in guard(lambda: investigator.find_plate(
                session, user, plate=plate, reason=reason,
                since=since, until=until, limit=limit,
            ))
        ]

    @app.get("/api/v1/reads/partial", tags=["reads"])
    def reads_partial(
        user: UserDep, session: Db,
        fragment: str, reason: str, limit: int = Query(200, ge=1, le=1000),
    ) -> list[dict]:
        return [
            r.to_dict()
            for r in guard(lambda: investigator.find_partial(
                session, user, fragment=fragment, reason=reason, limit=limit
            ))
        ]

    @app.get("/api/v1/reads/convoy", tags=["reads"])
    def reads_convoy(
        user: UserDep, session: Db,
        plate: str, reason: str,
        window_seconds: int = Query(120, ge=10, le=900),
        min_shared: int = Query(2, ge=1, le=20),
    ) -> list[dict]:
        return guard(lambda: investigator.convoy(
            session, user, plate=plate, reason=reason,
            window_seconds=window_seconds, min_shared=min_shared,
        ))

    @app.get("/api/v1/blobs/{sha256}", tags=["reads"])
    def get_blob(sha256: str, user: UserDep, session: Db, reason: str) -> Response:
        guard(lambda: authorise(user, "read.image", reason))
        blob = session.get(Blob, sha256)
        if blob is None or blob.data is None:
            raise HTTPException(404, "no such image")
        return Response(content=blob.data, media_type=blob.content_type)

    # -- zones ----------------------------------------------------------

    @app.get("/api/v1/sessions", tags=["zones"])
    def sessions(
        user: UserDep, session: Db, reason: str,
        zone_id: str | None = None, plate: str | None = None,
        open_only: bool = False, limit: int = Query(200, ge=1, le=1000),
    ) -> list[dict]:
        return guard(lambda: investigator.zone_sessions(
            session, user, reason=reason, zone_id=zone_id, plate=plate,
            open_only=open_only, limit=limit,
        ))

    @app.get("/api/v1/occupancy", tags=["zones"])
    def occupancy(user: UserDep, session: Db) -> dict:
        return guard(lambda: investigator.occupancy(session, user))

    @app.get("/api/v1/stats", tags=["zones"])
    def stats(user: UserDep, session: Db, hours: int = Query(24, ge=1, le=720)) -> dict:
        return guard(lambda: investigator.statistics(session, user, hours=hours))

    # -- screening ------------------------------------------------------

    @app.get("/api/v1/watchlist", tags=["screening"])
    def watchlist_list(user: UserDep, session: Db) -> list[dict]:
        guard(lambda: authorise(user, "watchlist.read", None))
        rows = session.scalars(
            select(WatchlistEntry).where(WatchlistEntry.active.is_(True))
            .order_by(WatchlistEntry.added_at.desc()).limit(500)
        ).all()
        return [
            {
                "id": e.id, "plate": e.plate_canonical, "category": e.category,
                "reason": e.reason, "authority_ref": e.authority_ref,
                "added_by": e.added_by, "added_at": e.added_at.isoformat(),
                "expires_at": e.expires_at.isoformat() if e.expires_at else None,
            }
            for e in rows
        ]

    @app.post("/api/v1/watchlist", tags=["screening"])
    def watchlist_add(body: WatchlistRequest, user: UserDep, session: Db) -> dict:
        guard(lambda: authorise(user, "watchlist.write", body.reason))
        expires = (
            utcnow() + timedelta(days=body.expires_in_days)
            if body.expires_in_days else None
        )
        entry = guard(lambda: screener.add_entry(
            session, hasher, plate=body.plate, reason=body.reason,
            added_by=user.username, category=body.category,
            authority_ref=body.authority_ref, expires_at=expires,
        ))
        # Surface where the vehicle has already been, not only where it goes
        # next -- otherwise the weeks already in the database are invisible.
        historical = screener.backfill(session, entry, days=body.backfill_days)
        return {
            "id": entry.id,
            "plate": entry.plate_canonical,
            "expires_at": entry.expires_at.isoformat() if entry.expires_at else None,
            "historical_hits": len(historical),
        }

    @app.get("/api/v1/hits", tags=["screening"])
    def hits(user: UserDep, session: Db, unacknowledged_only: bool = True) -> list[dict]:
        guard(lambda: authorise(user, "watchlist.read", None))
        stmt = select(Hit).order_by(Hit.matched_at.desc()).limit(500)
        if unacknowledged_only:
            stmt = stmt.where(Hit.acknowledged_at.is_(None))
        out = []
        for h in session.scalars(stmt).all():
            read = session.get(Read, h.read_id)
            entry = session.get(WatchlistEntry, h.watchlist_id)
            out.append({
                "id": h.id, "matched_at": h.matched_at.isoformat(),
                "confidence": h.confidence, "actionable": h.actionable,
                "plate": read.plate_canonical if read else None,
                "camera_id": read.camera_id if read else None,
                "captured_at": read.captured_at.isoformat() if read else None,
                "category": entry.category if entry else None,
            })
        return out

    @app.post("/api/v1/hits/{hit_id}/acknowledge", tags=["screening"])
    def acknowledge(hit_id: int, user: UserDep, session: Db) -> dict:
        guard(lambda: authorise(user, "hit.acknowledge", None))
        hit = session.get(Hit, hit_id)
        if hit is None:
            raise HTTPException(404, "no such hit")
        hit.acknowledged_by = user.username
        hit.acknowledged_at = utcnow()
        return {"id": hit.id, "acknowledged_by": hit.acknowledged_by}

    @app.get("/api/v1/anomalies", tags=["screening"])
    def anomaly_list(user: UserDep, session: Db, open_only: bool = True) -> list[dict]:
        guard(lambda: authorise(user, "watchlist.read", None))
        stmt = select(Anomaly).order_by(Anomaly.detected_at.desc()).limit(500)
        if open_only:
            stmt = stmt.where(Anomaly.reviewed_at.is_(None))
        return [
            {
                "id": a.id, "kind": a.kind, "plate": a.plate_canonical,
                "severity": a.severity, "detected_at": a.detected_at.isoformat(),
                "detail": a.detail, "read_ids": a.read_ids,
            }
            for a in session.scalars(stmt).all()
        ]

    # -- review queue (the active-learning loop) ------------------------

    @app.get("/api/v1/review", tags=["review"])
    def review_queue(user: UserDep, session: Db, limit: int = Query(50, ge=1, le=200)) -> list[dict]:
        guard(lambda: authorise(user, "review.read", None))
        rows = session.scalars(
            select(ReviewItem).where(ReviewItem.reviewed_at.is_(None))
            .order_by(ReviewItem.queued_at).limit(limit)
        ).all()
        out = []
        for item in rows:
            read = session.get(Read, item.read_id)
            out.append({
                "id": item.id, "read_id": item.read_id, "reason": item.reason,
                "machine_plate": item.machine_plate,
                "queued_at": item.queued_at.isoformat(),
                "camera_id": read.camera_id if read else None,
                "captured_at": read.captured_at.isoformat() if read else None,
                "plate_image_sha256": read.plate_image_sha256 if read else None,
                "confidence": read.confidence if read else None,
                "repaired_fields": list(read.repaired_fields or []) if read else [],
            })
        return out

    @app.post("/api/v1/review/{item_id}", tags=["review"])
    def review_submit(item_id: int, body: ReviewRequest, user: UserDep, session: Db) -> dict:
        guard(lambda: authorise(user, "review.write", None))
        item = session.get(ReviewItem, item_id)
        if item is None:
            raise HTTPException(404, "no such review item")
        if body.corrected_plate:
            from nepal_plate import parse as parse_plate

            parsed = parse_plate(body.corrected_plate)
            if not parsed.is_valid:
                raise HTTPException(400, f"{body.corrected_plate!r} is not a valid plate")
            item.corrected_plate = parsed.canonical
            item.confirmed = False
        else:
            item.confirmed = body.confirmed
        item.reviewed_by = user.username
        item.reviewed_at = utcnow()
        return {
            "id": item.id,
            "confirmed": item.confirmed,
            "corrected_plate": item.corrected_plate,
        }

    # -- oversight and administration -----------------------------------

    @app.get("/api/v1/audit", tags=["oversight"])
    def audit(
        user: UserDep, session: Db,
        actor: str | None = None, limit: int = Query(200, ge=1, le=1000),
    ) -> list[dict]:
        return guard(lambda: investigator.audit_trail(
            session, user, actor=actor, limit=limit
        ))

    @app.post("/api/v1/erasure", tags=["oversight"])
    def erasure(body: ErasureRequestBody, user: UserDep, session: Db) -> dict:
        req = guard(lambda: retention.request_erasure(
            session, user, plate=body.plate, reason=body.reason,
            hasher=hasher, authority_ref=body.authority_ref,
        ))
        done = retention.execute_erasure(session, req.id)
        return {
            "id": done.id, "status": done.status,
            "reads_deleted": done.reads_deleted,
            "sessions_deleted": done.sessions_deleted,
            "executed_at": done.executed_at.isoformat() if done.executed_at else None,
        }

    @app.post("/api/v1/admin/retention/sweep", tags=["oversight"])
    def sweep(user: UserDep, session: Db) -> dict:
        guard(lambda: authorise(user, "admin.retention", "scheduled retention sweep"))
        return retention.sweep(session).to_dict()

    return app


__all__ = ["create_app", "Settings"]
