"""Signed store-and-forward queue.

Nepal has load-shedding and unreliable rural connectivity. A system that stops
recording when the uplink drops is not deployable, so every read is written to
local durable storage *before* any attempt to send it, and the uplink is a
best-effort drain of that store rather than the primary path.

Design consequences, each of which is a decision that could have gone the other
way:

**SQLite, not a file-per-event or an in-memory ring.** A single WAL-mode SQLite
file gives atomic append, crash-safe ordering and cheap range queries in one
dependency that is already on every Linux image. A directory of JSON files
would be simpler still until the first power cut leaves a half-written record.

**Events are never deleted before acknowledgement, and gaps are never closed.**
The sequence numbers form a hash chain; renumbering to fill a gap would destroy
the evidence that something went missing. A gap is information.

**Chain state is persisted with the events, in the same transaction.** A node
that restarts and resumes at sequence 0 forks its own history and every read
after the restart becomes unverifiable. This is the single most dangerous
failure mode in the design, so the sequence and head hash live in the same
database and are read back on startup rather than being held only in memory.

**Blobs are content-addressed and reference-counted separately.** Several
frames of one track often produce identical crops; storing by SHA-256 collapses
them, and it means an event referencing a blob can be replayed after the blob
has been re-fetched.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Iterator, Sequence

from scanner_evidence import GENESIS, NodeIdentity, SignedEvent

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=FULL;

CREATE TABLE IF NOT EXISTS events (
    sequence     INTEGER PRIMARY KEY,
    node_id      TEXT    NOT NULL,
    captured_at  TEXT    NOT NULL,
    prev_hash    TEXT    NOT NULL,
    payload_hash TEXT    NOT NULL,
    signature    TEXT    NOT NULL,
    payload      TEXT    NOT NULL,
    kind         TEXT    NOT NULL,
    sent_at      TEXT,
    attempts     INTEGER NOT NULL DEFAULT 0,
    queued_at    TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_unsent ON events(sent_at, sequence);

CREATE TABLE IF NOT EXISTS blobs (
    sha256    TEXT PRIMARY KEY,
    path      TEXT NOT NULL,
    bytes     INTEGER NOT NULL,
    refs      INTEGER NOT NULL DEFAULT 0,
    stored_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True, slots=True)
class QueueStats:
    total: int
    unsent: int
    oldest_unsent: str | None
    blob_bytes: int
    disk_bytes: int


class EventQueue:
    """Durable, ordered, signed local queue for one edge node."""

    def __init__(self, root: Path | str, identity: NodeIdentity) -> None:
        self.root = Path(root)
        self.blob_dir = self.root / "blobs"
        self.blob_dir.mkdir(parents=True, exist_ok=True)
        self.identity = identity
        self._lock = threading.Lock()
        self._db = sqlite3.connect(self.root / "queue.db", check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.executescript(SCHEMA)
        self._restore_chain()

    # -- chain state ----------------------------------------------------

    def _restore_chain(self) -> None:
        """Resume the hash chain exactly where it stopped.

        Read from the events table itself rather than a separate counter: the
        counter can be lost or lag a crash, the last row cannot.
        """
        row = self._db.execute(
            "SELECT sequence, prev_hash, payload_hash, signature, node_id, captured_at "
            "FROM events ORDER BY sequence DESC LIMIT 1"
        ).fetchone()
        if row is None:
            self.identity.resume(0, GENESIS)
            return
        last = SignedEvent(
            node_id=row["node_id"], sequence=row["sequence"], captured_at=row["captured_at"],
            prev_hash=row["prev_hash"], payload_hash=row["payload_hash"],
            signature=row["signature"], payload={},
        )
        self.identity.resume(row["sequence"], last.event_hash())

    # -- writing --------------------------------------------------------

    def append(self, kind: str, payload: dict) -> SignedEvent:
        """Sign ``payload`` and durably enqueue it. Returns the signed event."""
        with self._lock:
            event = self.identity.sign({**payload, "kind": kind})
            self._db.execute(
                "INSERT INTO events (sequence, node_id, captured_at, prev_hash, payload_hash,"
                " signature, payload, kind, queued_at) VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    event.sequence, event.node_id, event.captured_at, event.prev_hash,
                    event.payload_hash, event.signature,
                    json.dumps(event.payload, ensure_ascii=False, sort_keys=True),
                    kind, _now(),
                ),
            )
            self._db.commit()
            return event

    def put_blob(self, data: bytes, *, suffix: str = ".jpg") -> str:
        """Store bytes content-addressed; returns the SHA-256 hex digest.

        Frames of one track frequently yield identical crops, so addressing by
        content collapses them for free.
        """
        sha = hashlib.sha256(data).hexdigest()
        with self._lock:
            row = self._db.execute("SELECT refs FROM blobs WHERE sha256=?", (sha,)).fetchone()
            if row is not None:
                self._db.execute("UPDATE blobs SET refs=refs+1 WHERE sha256=?", (sha,))
                self._db.commit()
                return sha
            rel = f"{sha[:2]}/{sha}{suffix}"
            path = self.blob_dir / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            self._db.execute(
                "INSERT INTO blobs (sha256, path, bytes, refs, stored_at) VALUES (?,?,?,1,?)",
                (sha, rel, len(data), _now()),
            )
            self._db.commit()
            return sha

    def blob_path(self, sha: str) -> Path | None:
        row = self._db.execute("SELECT path FROM blobs WHERE sha256=?", (sha,)).fetchone()
        return self.blob_dir / row["path"] if row else None

    # -- reading / draining ---------------------------------------------

    def pending(self, limit: int = 256) -> list[SignedEvent]:
        """Oldest unsent events first.

        Strict sequence order matters: the receiver verifies the chain link by
        link, so an out-of-order delivery looks identical to tampering.
        """
        rows = self._db.execute(
            "SELECT * FROM events WHERE sent_at IS NULL ORDER BY sequence LIMIT ?", (limit,)
        ).fetchall()
        return [self._row_to_event(r) for r in rows]

    def mark_sent(self, sequences: Sequence[int]) -> None:
        """Acknowledge delivery. Called only after the platform confirms."""
        if not sequences:
            return
        with self._lock:
            self._db.executemany(
                "UPDATE events SET sent_at=? WHERE sequence=?",
                [(_now(), s) for s in sequences],
            )
            self._db.commit()

    def mark_attempt(self, sequences: Sequence[int]) -> None:
        """Record a failed delivery attempt.

        Attempt counts are exposed in telemetry: a node whose attempts climb
        without acknowledgements has a reachable-but-broken uplink, which looks
        very different from one that is simply offline and should be alerted on
        differently.
        """
        if not sequences:
            return
        with self._lock:
            self._db.executemany(
                "UPDATE events SET attempts=attempts+1 WHERE sequence=?",
                [(s,) for s in sequences],
            )
            self._db.commit()

    def _row_to_event(self, row: sqlite3.Row) -> SignedEvent:
        return SignedEvent(
            node_id=row["node_id"], sequence=row["sequence"], captured_at=row["captured_at"],
            prev_hash=row["prev_hash"], payload_hash=row["payload_hash"],
            signature=row["signature"], payload=json.loads(row["payload"]),
        )

    def iter_all(self) -> Iterator[SignedEvent]:
        for row in self._db.execute("SELECT * FROM events ORDER BY sequence"):
            yield self._row_to_event(row)

    # -- retention ------------------------------------------------------

    def prune(self, *, keep_days: int = 7, keep_unsent: bool = True) -> tuple[int, int]:
        """Delete acknowledged events and unreferenced blobs past retention.

        Edge retention is deliberately short -- seven days by default, matching
        the tiered policy in ``docs/security-and-privacy.md``. The node is a
        buffer, not an archive; long-term retention is the platform's job, under
        access control an edge box cannot enforce.

        Unsent events are never pruned by default. Discarding a read that was
        never delivered is data loss, and an operator should be told the disk is
        full rather than quietly losing evidence.
        """
        cutoff = (datetime.now(timezone.utc) - timedelta(days=keep_days)).isoformat()
        with self._lock:
            if keep_unsent:
                cur = self._db.execute(
                    "DELETE FROM events WHERE sent_at IS NOT NULL AND queued_at < ?", (cutoff,)
                )
            else:
                cur = self._db.execute("DELETE FROM events WHERE queued_at < ?", (cutoff,))
            n_events = cur.rowcount

            # Blobs still referenced by a surviving event must stay.
            live: set[str] = set()
            for row in self._db.execute("SELECT payload FROM events"):
                live.update(_blob_refs(json.loads(row["payload"])))

            n_blobs = 0
            for row in self._db.execute("SELECT sha256, path FROM blobs").fetchall():
                if row["sha256"] in live:
                    continue
                p = self.blob_dir / row["path"]
                try:
                    p.unlink(missing_ok=True)
                except OSError:
                    continue
                self._db.execute("DELETE FROM blobs WHERE sha256=?", (row["sha256"],))
                n_blobs += 1
            self._db.commit()
        return n_events, n_blobs

    # -- introspection --------------------------------------------------

    def stats(self) -> QueueStats:
        total = self._db.execute("SELECT COUNT(*) c FROM events").fetchone()["c"]
        unsent = self._db.execute(
            "SELECT COUNT(*) c FROM events WHERE sent_at IS NULL"
        ).fetchone()["c"]
        oldest = self._db.execute(
            "SELECT captured_at FROM events WHERE sent_at IS NULL ORDER BY sequence LIMIT 1"
        ).fetchone()
        blob_bytes = self._db.execute(
            "SELECT COALESCE(SUM(bytes),0) b FROM blobs"
        ).fetchone()["b"]
        db_bytes = (self.root / "queue.db").stat().st_size if (self.root / "queue.db").exists() else 0
        return QueueStats(
            total=total,
            unsent=unsent,
            oldest_unsent=oldest["captured_at"] if oldest else None,
            blob_bytes=int(blob_bytes),
            disk_bytes=int(blob_bytes) + db_bytes,
        )

    def close(self) -> None:
        self._db.close()

    def __enter__(self) -> "EventQueue":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


#: Length of a hex SHA-256 digest.
_SHA_LEN = 64


def _blob_refs(payload) -> set[str]:
    """Every blob digest referenced anywhere in an event payload.

    Matches on the key *suffix* rather than an explicit list of names. An
    earlier version listed the names it knew about and silently missed
    ``plate_image_sha256`` and ``context_image_sha256`` -- the two the pipeline
    actually writes -- so retention deleted evidence images that live events
    still pointed at. Anything ending in ``_sha256`` (or named ``sha256``) is
    treated as a reference, and the value is shape-checked so a field that
    merely happens to be named that way cannot make a stray string look like a
    live reference.
    """
    found: set[str] = set()

    def looks_like_digest(v: object) -> bool:
        return (
            isinstance(v, str)
            and len(v) == _SHA_LEN
            and all(c in "0123456789abcdef" for c in v)
        )

    def walk(node) -> None:
        if isinstance(node, dict):
            for k, v in node.items():
                if (k == "sha256" or k.endswith("_sha256")) and looks_like_digest(v):
                    found.add(v)
                else:
                    walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(payload)
    return found


__all__ = ["EventQueue", "QueueStats"]
