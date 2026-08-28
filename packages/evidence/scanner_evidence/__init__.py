"""``scanner_evidence`` -- tamper-evident event chains.

Extracted into its own package because **both** the edge agent (which signs) and
the platform (which verifies) need exactly the same logic, and a divergence
between a signer and its verifier is a silent failure in the most
security-critical code in the system. One implementation, one canonical
serialisation, one set of tests.

Dependencies: ``cryptography`` only. Neither side should have to pull in the
other's stack -- the platform has no business importing OpenCV, and an edge
node has no business importing a web framework.
"""

from __future__ import annotations

from .chain import (
    GENESIS,
    NodeIdentity,
    SignedEvent,
    canonical_bytes,
    digest,
    verify,
    verify_chain,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "GENESIS",
    "SignedEvent",
    "NodeIdentity",
    "canonical_bytes",
    "digest",
    "verify",
    "verify_chain",
]
