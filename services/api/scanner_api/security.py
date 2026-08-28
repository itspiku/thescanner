"""Pseudonymisation, authentication, and mandatory reason-for-access.

Three mechanisms, each aimed at a specific threat from
``docs/security-and-privacy.md``.

**Keyed HMAC pseudonyms** (against database disclosure). Plates are indexed by
``HMAC-SHA256(key, canonical)`` rather than plaintext, so a stolen dump does not
hand the attacker a searchable movement history. This is pseudonymisation, not
encryption: an attacker holding the key, or one able to guess plates and
recompute digests, defeats it. Nepal's plate space is small enough (2.2 x 10^8
legacy plates) that offline enumeration against a leaked key is entirely
feasible, so the key must be held in a KMS or HSM and never in the database.
Stated plainly because a pseudonymisation scheme oversold as encryption is
worse than none -- it produces false confidence.

**Role-based access** (against over-broad authority). Four roles, least
privilege. ``AUDITOR`` cannot read reads at all, only the access log: an
oversight role that can perform the activity it oversees is not oversight.

**Mandatory reason-for-access** (against the insider). The likeliest real abuse
of a national ANPR is not an intrusion but an operator looking up a spouse, a
journalist or a political rival. No technical control stops that outright. What
this does is make every lookup carry an attributable, non-empty purpose, written
to an append-only log before the query runs -- so the abuse leaves a record even
when it succeeds.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterable

from .models import Role

#: Minimum characters for an access reason. Low enough not to obstruct real
#: work, high enough that "x" does not satisfy the requirement.
MIN_REASON_LENGTH = 8


class SecurityError(Exception):
    pass


class PermissionDenied(SecurityError):
    pass


class ReasonRequired(SecurityError):
    pass


# ---------------------------------------------------------------------------
# Pseudonymisation
# ---------------------------------------------------------------------------

class PlateHasher:
    """Keyed HMAC over canonical plate strings.

    The key must come from a secrets manager. Passing one explicitly is
    supported for tests; in production the environment variable is read once at
    startup and the process refuses to run without it, because a default key
    would make every deployment's pseudonyms identical and rainbow-tablable
    across installations.
    """

    ENV_VAR = "SCANNER_PLATE_KEY"

    def __init__(self, key: bytes | str | None = None) -> None:
        if key is None:
            env = os.environ.get(self.ENV_VAR)
            if not env:
                raise SecurityError(
                    f"{self.ENV_VAR} is not set. Generate one with "
                    f"'python -m scanner_api.cli genkey' and supply it from your "
                    f"secrets manager. Refusing to start with a default key."
                )
            key = env
        self._key = key.encode("utf-8") if isinstance(key, str) else key
        if len(self._key) < 32:
            raise SecurityError("plate key must be at least 32 bytes")

    def hash(self, canonical: str) -> str:
        return hmac.new(self._key, canonical.encode("utf-8"), hashlib.sha256).hexdigest()

    def hash_many(self, canonicals: Iterable[str]) -> list[str]:
        return [self.hash(c) for c in canonicals]

    @staticmethod
    def generate_key() -> str:
        return secrets.token_hex(32)


# ---------------------------------------------------------------------------
# Roles and permissions
# ---------------------------------------------------------------------------

#: action -> roles allowed to perform it.
PERMISSIONS: dict[str, frozenset[Role]] = {
    # Reading data
    "read.live": frozenset({Role.OPERATOR, Role.INVESTIGATOR, Role.ADMIN}),
    "read.search": frozenset({Role.INVESTIGATOR, Role.ADMIN}),
    "read.export": frozenset({Role.INVESTIGATOR, Role.ADMIN}),
    "read.image": frozenset({Role.OPERATOR, Role.INVESTIGATOR, Role.ADMIN}),
    "session.search": frozenset({Role.OPERATOR, Role.INVESTIGATOR, Role.ADMIN}),
    # Screening
    "watchlist.read": frozenset({Role.OPERATOR, Role.INVESTIGATOR, Role.ADMIN}),
    "watchlist.write": frozenset({Role.INVESTIGATOR, Role.ADMIN}),
    "hit.acknowledge": frozenset({Role.OPERATOR, Role.INVESTIGATOR, Role.ADMIN}),
    # Review queue -- the active-learning loop
    "review.read": frozenset({Role.OPERATOR, Role.INVESTIGATOR, Role.ADMIN}),
    "review.write": frozenset({Role.OPERATOR, Role.INVESTIGATOR, Role.ADMIN}),
    # Administration
    "node.enrol": frozenset({Role.ADMIN}),
    "admin.users": frozenset({Role.ADMIN}),
    "admin.retention": frozenset({Role.ADMIN}),
    "erasure.request": frozenset({Role.ADMIN, Role.INVESTIGATOR}),
    # Oversight. Deliberately disjoint from everything above.
    "audit.read": frozenset({Role.AUDITOR}),
}

#: Actions that touch personal data and therefore require a stated purpose.
#: Live monitoring is excluded: an operator watching a feed cannot type a reason
#: per vehicle, and requiring one would train everybody to paste boilerplate --
#: which destroys the value of the reasons that do matter.
REASON_REQUIRED = frozenset({
    "read.search", "read.export", "read.image", "session.search",
    "watchlist.write", "erasure.request",
})


@dataclass(frozen=True)
class Principal:
    """An authenticated caller."""

    username: str
    role: Role
    mfa: bool = False
    client_ip: str = ""

    def can(self, action: str) -> bool:
        return self.role in PERMISSIONS.get(action, frozenset())


def authorise(principal: Principal, action: str, reason: str | None) -> str:
    """Check permission and purpose. Returns the validated reason.

    Raises before any data is touched. The ordering matters: permission first,
    then purpose, so a caller who is not allowed to perform an action never
    learns whether their reason would have been acceptable.
    """
    if action not in PERMISSIONS:
        raise PermissionDenied(f"unknown action {action!r}")
    if not principal.can(action):
        raise PermissionDenied(
            f"role {principal.role.value!r} may not perform {action!r}"
        )
    if action in REASON_REQUIRED:
        cleaned = (reason or "").strip()
        if len(cleaned) < MIN_REASON_LENGTH:
            raise ReasonRequired(
                f"{action!r} requires a stated reason of at least "
                f"{MIN_REASON_LENGTH} characters"
            )
        return cleaned
    return (reason or "").strip() or "not required for this action"


# ---------------------------------------------------------------------------
# Tokens
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TokenConfig:
    secret: str
    ttl_minutes: int = 60


def issue_token(cfg: TokenConfig, username: str, role: Role, *, mfa: bool) -> str:
    """Mint a signed, expiring bearer token.

    A deliberately small HMAC token rather than JWT: no algorithm-confusion
    surface, no library, and nothing in it a client could be tempted to trust
    without verifying. For a system whose identity provider is expected to be
    Keycloak or similar in production, this is the local fallback.
    """
    expiry = int((datetime.now(timezone.utc) + timedelta(minutes=cfg.ttl_minutes)).timestamp())
    body = f"{username}|{role.value}|{int(mfa)}|{expiry}"
    sig = hmac.new(cfg.secret.encode(), body.encode(), hashlib.sha256).hexdigest()[:32]
    return f"{body}|{sig}"


def verify_token(cfg: TokenConfig, token: str, *, client_ip: str = "") -> Principal:
    parts = token.split("|")
    if len(parts) != 5:
        raise SecurityError("malformed token")
    username, role_s, mfa_s, expiry_s, sig = parts
    body = "|".join(parts[:4])
    expected = hmac.new(cfg.secret.encode(), body.encode(), hashlib.sha256).hexdigest()[:32]
    if not hmac.compare_digest(sig, expected):
        raise SecurityError("bad token signature")
    if int(expiry_s) < int(datetime.now(timezone.utc).timestamp()):
        raise SecurityError("token expired")
    try:
        role = Role(role_s)
    except ValueError as exc:
        raise SecurityError(f"unknown role {role_s!r}") from exc
    return Principal(username=username, role=role, mfa=mfa_s == "1", client_ip=client_ip)


def hash_password(password: str, *, salt: bytes | None = None) -> str:
    """scrypt. Deliberately not a fast hash."""
    salt = salt or secrets.token_bytes(16)
    dk = hashlib.scrypt(password.encode(), salt=salt, n=2**14, r=8, p=1, dklen=32)
    return f"scrypt${salt.hex()}${dk.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        scheme, salt_hex, dk_hex = stored.split("$")
        if scheme != "scrypt":
            return False
        dk = hashlib.scrypt(
            password.encode(), salt=bytes.fromhex(salt_hex), n=2**14, r=8, p=1, dklen=32
        )
        return hmac.compare_digest(dk.hex(), dk_hex)
    except (ValueError, TypeError):
        return False


__all__ = [
    "MIN_REASON_LENGTH",
    "SecurityError",
    "PermissionDenied",
    "ReasonRequired",
    "PlateHasher",
    "PERMISSIONS",
    "REASON_REQUIRED",
    "Principal",
    "authorise",
    "TokenConfig",
    "issue_token",
    "verify_token",
    "hash_password",
    "verify_password",
]
