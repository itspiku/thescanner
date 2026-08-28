"""Platform command line.

    scanner-api genkey                       # generate a secret
    scanner-api initdb                       # create tables and extensions
    scanner-api adduser --username x --role admin
    scanner-api serve                        # run the API
    scanner-api sweep                        # run retention once
    scanner-api verify-node --node-id NODE   # re-verify a node's chain
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import sys

from sqlalchemy import select

from .db import Database, DbConfig, apply_postgres_extensions
from .models import Node, Read, Role, User
from .retention import RetentionService
from .security import PlateHasher, hash_password


def _db() -> Database:
    return Database(DbConfig(url=os.environ.get("SCANNER_DB_URL", "sqlite:///./scanner.db")))


def cmd_genkey(args) -> int:
    print("Generated secrets. Store these in your secrets manager, never in the")
    print("database or a config file committed to git.\n")
    print(f"SCANNER_PLATE_KEY={PlateHasher.generate_key()}")
    print(f"SCANNER_TOKEN_SECRET={PlateHasher.generate_key()}")
    print(f"SCANNER_NODE_TOKEN={PlateHasher.generate_key()}")
    print()
    print("SCANNER_PLATE_KEY in particular must never change once data exists:")
    print("every stored plate is indexed by its HMAC, so a new key orphans the")
    print("entire history.")
    return 0


def cmd_initdb(args) -> int:
    db = _db()
    db.init()
    print(f"schema created on {db.cfg.url}")
    if db.cfg.is_postgres:
        applied = apply_postgres_extensions(db.engine)
        for name, ok in applied.items():
            print(f"  {name:14s} {'enabled' if ok else 'unavailable (degrades to plain SQL)'}")
    else:
        print("  running on SQLite: TimescaleDB and pgvector features are skipped.")
        print("  Correctness is identical; only large-scale performance differs.")
    return 0


def cmd_adduser(args) -> int:
    password = args.password or getpass.getpass("password: ")
    if len(password) < 12:
        print("password must be at least 12 characters", file=sys.stderr)
        return 1
    db = _db()
    db.init()
    with db.session() as s:
        if s.scalar(select(User).where(User.username == args.username)):
            print(f"user {args.username!r} already exists", file=sys.stderr)
            return 1
        s.add(User(
            username=args.username,
            full_name=args.full_name,
            role=Role(args.role).value,
            password_hash=hash_password(password),
            mfa_enrolled=False,
        ))
    print(f"created {args.username} with role {args.role}")
    print("MFA is not yet enrolled. Production deployments must front this API")
    print("with OIDC and require MFA before granting any role above operator.")
    return 0


def cmd_serve(args) -> int:
    import uvicorn

    uvicorn.run(
        "scanner_api.app:create_app",
        factory=True,
        host=args.host,
        port=args.port,
        reload=args.reload,
    )
    return 0


def cmd_sweep(args) -> int:
    db = _db()
    svc = RetentionService()
    with db.session() as s:
        before = svc.overdue(s)
        result = svc.sweep(s)
    print(json.dumps({"deleted": result.to_dict(), "was_overdue": before}, indent=2))
    return 0


def cmd_verify_node(args) -> int:
    """Re-verify a node's stored reads against its enrolled public key.

    The platform verifies on ingest, but an auditor needs to be able to check
    the stored records independently and later -- ingest-time verification is a
    claim about the past that this re-checks against the data as it stands now.
    """
    from scanner_evidence import SignedEvent, verify

    db = _db()
    with db.session() as s:
        node = s.get(Node, args.node_id)
        if node is None:
            print(f"no node {args.node_id!r}", file=sys.stderr)
            return 1
        reads = s.scalars(
            select(Read).where(Read.node_id == args.node_id).order_by(Read.sequence)
        ).all()
        if not reads:
            print(f"{args.node_id}: no reads stored")
            return 0

        bad: list[int] = []
        gaps: list[str] = []
        expected = None
        for r in reads:
            if expected is not None and r.sequence != expected:
                gaps.append(f"{expected}..{r.sequence - 1}")
            expected = r.sequence + 1
            # Rebuild the signed header from stored columns. The payload itself
            # is not reconstructed -- payload_hash is what was signed, and it is
            # stored verbatim.
            ev = SignedEvent(
                node_id=r.node_id, sequence=r.sequence,
                captured_at=r.captured_at.isoformat().replace("+00:00", "Z"),
                prev_hash=r.prev_hash, payload_hash=r.payload_hash,
                signature=r.signature, payload={},
            )
            from scanner_evidence import canonical_bytes
            from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
            from cryptography.exceptions import InvalidSignature
            import base64

            try:
                Ed25519PublicKey.from_public_bytes(
                    base64.b64decode(node.public_key)
                ).verify(base64.b64decode(ev.signature), canonical_bytes(ev.header()))
            except (InvalidSignature, ValueError):
                bad.append(r.sequence)

        print(f"{args.node_id}: {len(reads):,} reads, "
              f"sequence {reads[0].sequence}-{reads[-1].sequence}")
        print(f"  signature failures : {len(bad)}" + (f"  {bad[:10]}" if bad else ""))
        print(f"  sequence gaps      : {len(gaps)}" + (f"  {gaps[:10]}" if gaps else ""))
        if gaps:
            print("  A gap is not necessarily tampering -- it can be a read lost in")
            print("  transit or removed by retention -- but it is never nothing.")
        return 1 if bad else 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="scanner-api", description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("genkey", help="generate deployment secrets").set_defaults(func=cmd_genkey)
    sub.add_parser("initdb", help="create schema and extensions").set_defaults(func=cmd_initdb)

    u = sub.add_parser("adduser", help="create a user")
    u.add_argument("--username", required=True)
    u.add_argument("--role", required=True, choices=[r.value for r in Role])
    u.add_argument("--full-name", default="")
    u.add_argument("--password", default=None, help="omit to be prompted")
    u.set_defaults(func=cmd_adduser)

    s = sub.add_parser("serve", help="run the API")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--reload", action="store_true")
    s.set_defaults(func=cmd_serve)

    sub.add_parser("sweep", help="run retention once").set_defaults(func=cmd_sweep)

    v = sub.add_parser("verify-node", help="re-verify a node's stored chain")
    v.add_argument("--node-id", required=True)
    v.set_defaults(func=cmd_verify_node)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
