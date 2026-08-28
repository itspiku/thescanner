"""Database engine and dialect-conditional features.

PostgreSQL in production, SQLite for development and tests. The ORM models are
identical; what differs is the extensions applied on top.

Running the same schema on SQLite is not a compromise for its own sake -- it is
what lets the whole test suite run with no infrastructure, which in turn is what
makes the tests get run. The Postgres-only features (hypertables, compression,
vector indexes) are all *performance* features. Nothing about correctness
depends on them, so a developer sees the same behaviour, just slower on data
volumes they will never have locally.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from sqlalchemy import Engine, create_engine, event, text
from sqlalchemy.orm import Session, sessionmaker

from .models import Base

DEFAULT_SQLITE = "sqlite:///./scanner.db"


@dataclass(frozen=True)
class DbConfig:
    url: str = os.environ.get("SCANNER_DB_URL", DEFAULT_SQLITE)
    echo: bool = False
    pool_size: int = 10
    max_overflow: int = 20

    @property
    def is_postgres(self) -> bool:
        return self.url.startswith(("postgresql", "postgres://"))


def create_db_engine(cfg: DbConfig | None = None) -> Engine:
    cfg = cfg or DbConfig()
    if cfg.is_postgres:
        engine = create_engine(
            cfg.url, echo=cfg.echo, pool_size=cfg.pool_size,
            max_overflow=cfg.max_overflow, pool_pre_ping=True,
        )
    else:
        engine = create_engine(
            cfg.url, echo=cfg.echo,
            connect_args={"check_same_thread": False},
        )

        @event.listens_for(engine, "connect")
        def _sqlite_pragmas(conn, _record):  # pragma: no cover - trivial
            cur = conn.cursor()
            # WAL for concurrent readers; foreign keys are OFF by default in
            # SQLite, which would silently disable every FK constraint in the
            # schema and make the tests pass for the wrong reason.
            cur.execute("PRAGMA journal_mode=WAL")
            cur.execute("PRAGMA foreign_keys=ON")
            cur.close()

    return engine


def create_session_factory(engine: Engine) -> sessionmaker[Session]:
    """Sessions with autoflush left ON.

    Turning it off looks like a performance win and is a correctness trap: a row
    added but not yet flushed is invisible to a query in the same transaction,
    so ingest would add a read and screening would fail to find it. The flushes
    are cheap; the silent misses are not.

    ``expire_on_commit`` is off so that objects returned to callers stay usable
    after the session commits, which is what the request-scoped session
    dependency needs.
    """
    return sessionmaker(bind=engine, autoflush=True, expire_on_commit=False)


def init_schema(engine: Engine) -> None:
    """Create tables, then apply Postgres-only features if available."""
    Base.metadata.create_all(engine)
    if engine.dialect.name == "postgresql":
        apply_postgres_extensions(engine)


def apply_postgres_extensions(engine: Engine) -> dict[str, bool]:
    """Enable TimescaleDB and pgvector where present.

    Every step is individually optional and reported. A deployment without
    TimescaleDB still works -- ``reads`` is an ordinary table with ordinary
    indexes -- so a missing extension degrades performance rather than blocking
    startup. Silently requiring an extension is how a system becomes
    undeployable on the one server the ministry actually has.
    """
    applied: dict[str, bool] = {}
    statements = [
        ("timescaledb", "CREATE EXTENSION IF NOT EXISTS timescaledb"),
        ("pgvector", "CREATE EXTENSION IF NOT EXISTS vector"),
    ]
    with engine.begin() as conn:
        for name, sql in statements:
            try:
                conn.execute(text(sql))
                applied[name] = True
            except Exception:
                applied[name] = False

    if applied.get("timescaledb"):
        with engine.begin() as conn:
            try:
                # Reads are overwhelmingly written in time order and queried by
                # time range, which is exactly the hypertable access pattern.
                conn.execute(
                    text(
                        "SELECT create_hypertable('reads', 'captured_at', "
                        "if_not_exists => TRUE, migrate_data => TRUE)"
                    )
                )
                conn.execute(
                    text(
                        "ALTER TABLE reads SET ("
                        "  timescaledb.compress,"
                        "  timescaledb.compress_segmentby = 'camera_id',"
                        "  timescaledb.compress_orderby = 'captured_at DESC'"
                        ")"
                    )
                )
                # Reads older than 30 days are investigated, not monitored, so
                # compression costs nothing operationally and saves most of the
                # storage over a 12-month retention.
                conn.execute(
                    text(
                        "SELECT add_compression_policy('reads', INTERVAL '30 days', "
                        "if_not_exists => TRUE)"
                    )
                )
                applied["hypertable"] = True
            except Exception:
                applied["hypertable"] = False

    return applied


def continuous_aggregates_sql() -> str:
    """Zone occupancy rollups, as a TimescaleDB continuous aggregate.

    The console's occupancy dashboard would otherwise scan the whole reads
    table on every refresh. Kept as SQL rather than ORM because continuous
    aggregates have no ORM representation and pretending otherwise would hide
    what is actually being created.
    """
    return """
    CREATE MATERIALIZED VIEW IF NOT EXISTS reads_hourly
    WITH (timescaledb.continuous) AS
    SELECT
        time_bucket('1 hour', captured_at) AS bucket,
        camera_id,
        site_id,
        count(*)                                        AS n_reads,
        count(*) FILTER (WHERE confidence = 'high')     AS n_high,
        count(DISTINCT plate_hmac)                      AS n_distinct_plates
    FROM reads
    WHERE NOT provisional
    GROUP BY bucket, camera_id, site_id;

    SELECT add_continuous_aggregate_policy('reads_hourly',
        start_offset => INTERVAL '3 days',
        end_offset   => INTERVAL '1 hour',
        schedule_interval => INTERVAL '15 minutes',
        if_not_exists => TRUE);
    """


class Database:
    """Engine plus session factory, with a scoped-session helper."""

    def __init__(self, cfg: DbConfig | None = None) -> None:
        self.cfg = cfg or DbConfig()
        self.engine = create_db_engine(self.cfg)
        self.session_factory = create_session_factory(self.engine)

    def init(self) -> None:
        init_schema(self.engine)

    @contextmanager
    def session(self) -> Iterator[Session]:
        s = self.session_factory()
        try:
            yield s
            s.commit()
        except Exception:
            s.rollback()
            raise
        finally:
            s.close()

    def dispose(self) -> None:
        self.engine.dispose()


__all__ = [
    "DbConfig",
    "Database",
    "create_db_engine",
    "create_session_factory",
    "init_schema",
    "apply_postgres_extensions",
    "continuous_aggregates_sql",
]
