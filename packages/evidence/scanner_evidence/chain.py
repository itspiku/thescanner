"""Node identity and the tamper-evident read chain.

A read produced by this system may end up as evidence in a prosecution. That
imposes requirements ordinary telemetry does not have: it must be possible to
show, later and to someone hostile, that a given read came from a given camera
at a given time and has not been altered since.

Two mechanisms, both cheap:

**Per-node signing.** Every edge node holds an Ed25519 keypair generated on
first run. The private key never leaves the node. Every event is signed at the
moment of capture, so a read's origin is provable and a forged read injected
downstream cannot be made to verify.

**Hash chaining.** Each event carries the hash of its predecessor from the same
node, forming an append-only chain. Altering or deleting a historical read
breaks every link after it, so tampering is not merely detected but *located*.
Chain heads are published periodically to the platform, which pins the history
up to that point -- an attacker who compromises a node afterwards still cannot
rewrite anything already witnessed.

What this does and does not give you
------------------------------------
It proves integrity and origin. It does **not** prove the camera was pointed
where the metadata says, nor that its clock was right. Those are addressed
separately: siting is recorded at commissioning, and clock discipline is part
of node health telemetry. Signing a confidently wrong timestamp is still
signing something wrong, so the chain records the node's clock *source* and
last sync alongside the time.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import stat
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from cryptography.exceptions import InvalidSignature

#: Genesis link for a node's first event.
GENESIS: str = "0" * 64


def _b64(raw: bytes) -> str:
    return base64.b64encode(raw).decode("ascii")


def _unb64(s: str) -> bytes:
    return base64.b64decode(s.encode("ascii"))


def canonical_bytes(payload: Mapping[str, Any]) -> bytes:
    """Deterministic serialisation of an event body.

    Signatures are only verifiable if both sides serialise identically, so this
    is pinned: sorted keys, no whitespace, UTF-8, no ASCII escaping. Any change
    here invalidates every signature ever produced, so it is deliberately
    boring and must never be "improved".
    """
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def digest(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_bytes(payload)).hexdigest()


@dataclass(frozen=True, slots=True)
class SignedEvent:
    """One link in a node's chain."""

    node_id: str
    #: Monotonic per-node sequence. Gaps are themselves evidence -- of a lost
    #: event or a deletion -- so they are never silently renumbered.
    sequence: int
    captured_at: str
    prev_hash: str
    payload_hash: str
    signature: str
    payload: dict[str, Any] = field(default_factory=dict)

    def header(self) -> dict[str, Any]:
        """The part that is signed: everything except the signature itself."""
        return {
            "node_id": self.node_id,
            "sequence": self.sequence,
            "captured_at": self.captured_at,
            "prev_hash": self.prev_hash,
            "payload_hash": self.payload_hash,
        }

    def event_hash(self) -> str:
        """This event's link value, which the next event will point back to."""
        return hashlib.sha256(
            canonical_bytes(self.header()) + _unb64(self.signature)
        ).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "SignedEvent":
        return cls(
            node_id=d["node_id"],
            sequence=int(d["sequence"]),
            captured_at=d["captured_at"],
            prev_hash=d["prev_hash"],
            payload_hash=d["payload_hash"],
            signature=d["signature"],
            payload=dict(d.get("payload", {})),
        )


class NodeIdentity:
    """An edge node's signing key and chain position.

    The key is generated on first run and stored with owner-only permissions.
    It is never transmitted; the platform is enrolled with the *public* key at
    commissioning.
    """

    def __init__(self, node_id: str, private_key: Ed25519PrivateKey, *, sequence: int = 0,
                 head: str = GENESIS) -> None:
        self.node_id = node_id
        self._key = private_key
        self.sequence = sequence
        self.head = head

    # -- construction ---------------------------------------------------

    @classmethod
    def load_or_create(cls, path: Path | str, node_id: str) -> "NodeIdentity":
        path = Path(path)
        if path.is_file():
            key = serialization.load_pem_private_key(path.read_bytes(), password=None)
            if not isinstance(key, Ed25519PrivateKey):
                raise ValueError(f"{path} is not an Ed25519 private key")
            return cls(node_id, key)

        key = Ed25519PrivateKey.generate()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(
            key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption(),
            )
        )
        # Owner-only. A world-readable signing key would let anyone with shell
        # access on the node mint reads attributable to that camera.
        try:
            os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
        except OSError:
            pass  # best effort; Windows ACLs are handled at provisioning
        return cls(node_id, key)

    # -- keys -----------------------------------------------------------

    @property
    def public_key(self) -> Ed25519PublicKey:
        return self._key.public_key()

    def public_key_b64(self) -> str:
        return _b64(
            self.public_key.public_bytes(
                encoding=serialization.Encoding.Raw,
                format=serialization.PublicFormat.Raw,
            )
        )

    def enrolment_record(self) -> dict[str, str]:
        """What the platform needs to verify this node. Public data only."""
        return {
            "node_id": self.node_id,
            "public_key": self.public_key_b64(),
            "algorithm": "Ed25519",
        }

    # -- signing --------------------------------------------------------

    def sign(self, payload: Mapping[str, Any], *, captured_at: datetime | None = None) -> SignedEvent:
        """Append ``payload`` to this node's chain and sign it."""
        ts = (captured_at or datetime.now(timezone.utc)).astimezone(timezone.utc)
        self.sequence += 1
        header = {
            "node_id": self.node_id,
            "sequence": self.sequence,
            "captured_at": ts.isoformat().replace("+00:00", "Z"),
            "prev_hash": self.head,
            "payload_hash": digest(payload),
        }
        signature = self._key.sign(canonical_bytes(header))
        event = SignedEvent(**header, signature=_b64(signature), payload=dict(payload))
        self.head = event.event_hash()
        return event

    def resume(self, sequence: int, head: str) -> None:
        """Restore chain position after a restart.

        Getting this wrong is the one way to break the chain from the inside: a
        node that restarts at sequence 0 produces a second event 1 and the
        history forks. The queue persists both values and calls this on
        startup.
        """
        self.sequence = sequence
        self.head = head


def verify(event: SignedEvent, public_key_b64: str, *, expected_prev: str | None = None) -> bool:
    """Verify one event's signature, payload hash and (optionally) its link.

    All three matter. A valid signature over a header whose ``payload_hash``
    does not match the payload proves only that someone signed a *description*
    of a different event.
    """
    if digest(event.payload) != event.payload_hash:
        return False
    if expected_prev is not None and event.prev_hash != expected_prev:
        return False
    key = Ed25519PublicKey.from_public_bytes(_unb64(public_key_b64))
    try:
        key.verify(_unb64(event.signature), canonical_bytes(event.header()))
    except InvalidSignature:
        return False
    return True


def verify_chain(
    events: list[SignedEvent], public_key_b64: str, *, start_hash: str = GENESIS
) -> tuple[bool, str | None]:
    """Verify a run of events links correctly.

    Returns ``(ok, reason)``. On failure the reason names the sequence number
    where the chain broke, because "the evidence is invalid" is not a useful
    thing to tell a court -- "read 41,207 from node KTM-BAL-02 was altered" is.
    """
    prev = start_hash
    expected_seq: int | None = None
    for e in events:
        if expected_seq is not None and e.sequence != expected_seq:
            return False, f"sequence gap at {e.sequence} (expected {expected_seq})"
        if not verify(e, public_key_b64, expected_prev=prev):
            return False, f"verification failed at sequence {e.sequence}"
        prev = e.event_hash()
        expected_seq = e.sequence + 1
    return True, None


__all__ = [
    "GENESIS",
    "SignedEvent",
    "NodeIdentity",
    "canonical_bytes",
    "digest",
    "verify",
    "verify_chain",
]
