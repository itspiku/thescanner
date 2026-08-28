"""Draining the local queue to the platform.

The uplink is deliberately a *best-effort drain of durable local storage*, never
the primary write path. Every read is already safe on disk before this code
runs, so a network failure degrades latency rather than losing evidence.

Rules the implementation follows, each of which exists because the obvious
alternative breaks something:

**Strict sequence order.** The receiver verifies the hash chain link by link,
so an out-of-order batch is indistinguishable from tampering. Batches are sent
in sequence order and a failed batch is retried before any later one is
attempted -- no skipping ahead past a poisoned event, because a gap in the chain
is exactly what the chain exists to detect.

**Acknowledge only on confirmation.** An event is marked sent when the platform
says it stored it, not when the HTTP request left. The cost of re-sending a
duplicate is nothing (the platform deduplicates on ``node_id`` + ``sequence``);
the cost of dropping an unacknowledged read is evidence.

**Backoff that distinguishes offline from broken.** A connection error means the
link is down and the node should back off hard. A 4xx means the platform
rejected the *content* -- a schema mismatch, an unenrolled key -- and retrying
forever will never fix it, so it is surfaced as an alert instead of a silent
retry loop.

**Blobs before events.** An event referencing an image the platform does not
have is not much use in an investigation, so referenced blobs are uploaded
first and the event only afterwards.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Callable, Sequence

from .identity import SignedEvent
from .queue import EventQueue


class UplinkError(Exception):
    """Transport failure. Retryable."""


class UplinkRejected(Exception):
    """The platform refused the content. Retrying will not help."""


@dataclass(frozen=True)
class UplinkConfig:
    base_url: str
    node_id: str
    #: Shared secret issued at commissioning. Authenticates the *node*; the
    #: per-event Ed25519 signatures authenticate the *events*, and the two are
    #: deliberately separate so that a leaked transport token cannot forge reads.
    token: str = ""
    batch_size: int = 128
    timeout: float = 20.0
    min_backoff: float = 2.0
    max_backoff: float = 300.0
    verify_tls: bool = True


class Uplink:
    """Ships queued events to the platform."""

    def __init__(
        self,
        cfg: UplinkConfig,
        queue: EventQueue,
        *,
        transport: Callable[[str, bytes, dict], tuple[int, bytes]] | None = None,
    ) -> None:
        self.cfg = cfg
        self.queue = queue
        #: Injectable so tests and the offline simulator do not need a server.
        self._transport = transport or self._http
        self.backoff = cfg.min_backoff
        self.last_success: float | None = None
        self.consecutive_failures = 0

    # -- transport ------------------------------------------------------

    def _http(self, path: str, body: bytes, headers: dict) -> tuple[int, bytes]:
        req = urllib.request.Request(
            self.cfg.base_url.rstrip("/") + path, data=body, headers=headers, method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=self.cfg.timeout) as r:
                return r.status, r.read()
        except urllib.error.HTTPError as e:
            return e.code, e.read()
        except (urllib.error.URLError, OSError, TimeoutError) as e:
            raise UplinkError(str(e)) from e

    def _post(self, path: str, payload: dict) -> dict:
        headers = {"Content-Type": "application/json", "X-Node-Id": self.cfg.node_id}
        if self.cfg.token:
            headers["Authorization"] = f"Bearer {self.cfg.token}"
        status, body = self._transport(
            path, json.dumps(payload, ensure_ascii=False).encode("utf-8"), headers
        )
        if status in (200, 201, 202):
            try:
                return json.loads(body) if body else {}
            except json.JSONDecodeError:
                return {}
        if 400 <= status < 500 and status not in (408, 429):
            raise UplinkRejected(f"HTTP {status}: {body[:300]!r}")
        raise UplinkError(f"HTTP {status}")

    # -- draining -------------------------------------------------------

    def drain_once(self) -> int:
        """Send at most one batch. Returns the number acknowledged."""
        events = self.queue.pending(self.cfg.batch_size)
        if not events:
            return 0

        try:
            self._upload_blobs(events)
            result = self._post(
                "/api/v1/ingest/events",
                {
                    "node_id": self.cfg.node_id,
                    "events": [e.to_dict() for e in events],
                },
            )
        except UplinkRejected:
            # Content the platform will never accept. Count the attempt so it
            # surfaces in telemetry, then re-raise -- silently retrying forever
            # would hide a schema or enrolment problem indefinitely.
            self.queue.mark_attempt([e.sequence for e in events])
            self.consecutive_failures += 1
            raise
        except UplinkError:
            self.queue.mark_attempt([e.sequence for e in events])
            self.consecutive_failures += 1
            self.backoff = min(self.backoff * 2, self.cfg.max_backoff)
            return 0

        # Trust the platform's explicit acknowledgement list when it sends one:
        # a partial store must not mark the whole batch delivered.
        accepted = result.get("accepted")
        sequences = (
            [int(s) for s in accepted]
            if isinstance(accepted, list)
            else [e.sequence for e in events]
        )
        self.queue.mark_sent(sequences)
        self.last_success = time.time()
        self.consecutive_failures = 0
        self.backoff = self.cfg.min_backoff
        return len(sequences)

    def _upload_blobs(self, events: Sequence[SignedEvent]) -> None:
        """Upload referenced images before the events that reference them."""
        wanted: set[str] = set()
        for e in events:
            for key in ("plate_image_sha256", "context_image_sha256"):
                sha = e.payload.get(key)
                if isinstance(sha, str):
                    wanted.add(sha)
        if not wanted:
            return

        # Ask which are missing rather than pushing all of them: crops repeat
        # heavily across a track, and the link is the scarcest resource here.
        have = set(self._post("/api/v1/ingest/blobs/check", {"sha256": sorted(wanted)}).get("have", []))
        for sha in sorted(wanted - have):
            path = self.queue.blob_path(sha)
            if path is None or not path.is_file():
                continue
            import base64

            self._post(
                "/api/v1/ingest/blobs",
                {
                    "sha256": sha,
                    "content_type": "image/jpeg",
                    "data": base64.b64encode(path.read_bytes()).decode("ascii"),
                },
            )

    def drain(self, *, max_batches: int = 64) -> int:
        """Drain up to ``max_batches``. Returns the total acknowledged."""
        total = 0
        for _ in range(max_batches):
            try:
                n = self.drain_once()
            except UplinkRejected:
                break
            if n == 0:
                break
            total += n
        return total

    def enrol(self) -> dict:
        """Register this node's public key with the platform.

        Idempotent, and run at commissioning. The platform will not accept
        events from a node whose public key it does not hold, which is what
        stops a fabricated node from injecting reads.
        """
        return self._post("/api/v1/nodes/enrol", self.queue.identity.enrolment_record())

    def status(self) -> dict:
        stats = self.queue.stats()
        return {
            "node_id": self.cfg.node_id,
            "unsent": stats.unsent,
            "oldest_unsent": stats.oldest_unsent,
            "disk_bytes": stats.disk_bytes,
            "last_success": self.last_success,
            "consecutive_failures": self.consecutive_failures,
            "backoff_s": self.backoff,
            # An offline node and a node whose uplink is rejecting content look
            # identical from the outside unless this distinction is exposed.
            "healthy": self.consecutive_failures == 0,
        }


__all__ = ["Uplink", "UplinkConfig", "UplinkError", "UplinkRejected"]
