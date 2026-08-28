# scanner-evidence

Tamper-evident, signed event chains for
[TheScanner](https://github.com/itspiku/thescanner).

Reads produced by this system may end up as evidence in a prosecution, which
imposes requirements ordinary telemetry does not have: it must be possible to
show, later and to someone hostile, that a given read came from a given camera
at a given time and has not been altered since.

- **Per-node Ed25519 signing.** The private key never leaves the edge node.
  Every event is signed at capture, so origin is provable and a read injected
  downstream cannot be made to verify.
- **Hash chaining.** Each event carries the hash of its predecessor, so altering
  or deleting a historical read breaks every link after it. Tampering is not
  merely detected but *located* — `verify_chain` names the sequence number where
  the chain broke, because "the evidence is invalid" is not a useful thing to
  tell a court.

## Why a separate package

The edge agent signs and the platform verifies. A divergence between a signer
and its verifier is a silent failure in the most security-critical code in the
system, so there is one implementation, one canonical serialisation, and one set
of tests. Neither side pulls in the other's stack.

## What it does not give you

Integrity and origin, not truth. It does not prove the camera was pointed where
its metadata says, nor that its clock was right — signing a confidently wrong
timestamp is still signing something wrong. Siting is recorded at
commissioning; clock discipline is part of node health telemetry.

Licence: Apache-2.0.
