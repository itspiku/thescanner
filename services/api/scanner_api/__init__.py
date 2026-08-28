"""``scanner_api`` -- the platform.

The Screen and Exploit stages of the Edge -> Ingest -> Screen -> Exploit
pipeline: accept signed reads from edge nodes, verify them, match against
watch-lists, detect cloned plates, serve investigation queries, and enforce
retention and erasure.

One PostgreSQL instance carries relational entities, the time-series of reads
(TimescaleDB) and appearance vectors (pgvector). The same schema runs on SQLite
for development and tests, which is what lets the suite run with no
infrastructure. Only performance differs.

Three properties the design is built around:

* **The audit trail is unbypassable.** No SQL lives in the HTTP layer. Every
  route delegates to the query layer, which writes an attributable, mandatory
  reason to an append-only log before the query runs.
* **Reads are verified, not trusted.** Every event's Ed25519 signature is checked
  against the node's enrolled public key. Failures are stored and flagged rather
  than discarded -- silently dropping unverifiable events would let anyone who
  can corrupt the link erase reads at will.
* **Retention is a column, not a cron job.** Every row carries its own
  ``expires_at``, so "is retention actually happening?" is an ordinary query
  rather than a matter of trusting a scheduler.

Entry points::

    scanner-api genkey
    scanner-api initdb
    scanner-api adduser --username alice --role investigator
    scanner-api serve
"""

from __future__ import annotations

from .db import Database, DbConfig
from .ingest import Ingestor, IngestResult, RetentionPolicy
from .models import Role
from .retention import RetentionService
from .screening import AnomalyDetector, Screener
from .search import Investigator
from .security import PlateHasher, Principal

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "Database",
    "DbConfig",
    "Ingestor",
    "IngestResult",
    "RetentionPolicy",
    "RetentionService",
    "Screener",
    "AnomalyDetector",
    "Investigator",
    "PlateHasher",
    "Principal",
    "Role",
]
