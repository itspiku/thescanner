# scanner-api

The platform for [TheScanner](https://github.com/itspiku/thescanner) — the Screen
and Exploit stages of **Edge → Ingest → Screen → Exploit**.

Accepts signed reads from edge nodes, verifies them, matches against watch-lists,
detects cloned plates, serves investigation queries, and enforces retention and
erasure under Nepal's Privacy Act 2075.

## One database

PostgreSQL carries everything: relational entities, the time-series of reads
(TimescaleDB hypertable with compression) and vehicle appearance vectors
(pgvector). The obvious alternative — Postgres plus a time-series store plus a
vector store — is three systems to operate, back up, secure and staff, which for
a government deployment in a country with a small pool of specialist operators
costs more than the marginal performance it buys.

The same schema runs on **SQLite** for development and tests. Timescale and
pgvector features are applied conditionally, so the suite runs with no
infrastructure and production loses nothing. Every Postgres-only feature is a
*performance* feature; correctness is identical either way.

## Three properties the design is built around

**The audit trail is unbypassable.** No SQL lives in the HTTP layer. Every route
delegates to `search.Investigator`, which writes an attributable, mandatory
reason to an append-only log *before* the query runs — so a query that crashes
or times out still leaves a record. An endpoint cannot accidentally read personal
data without being audited, because it has no way to query at all.

**Reads are verified, not trusted.** Every event's Ed25519 signature and payload
hash are checked against the node's enrolled public key. Failures are stored with
`verified=False` rather than discarded — silently dropping unverifiable events
would let anyone who can corrupt the link erase reads at will. Only verified
reads can raise a watch-list alert.

**Retention is a column, not a cron job.** Every row carries its own
`expires_at`, so "is retention actually happening?" is an ordinary query rather
than a matter of trusting a scheduler. `/healthz` reports the overdue count.

## Honest note on pseudonymisation

Plates are indexed by `HMAC-SHA256(key, canonical)` so a stolen database dump
does not hand an attacker a searchable movement history. This is
**pseudonymisation, not encryption**: Nepal's plate space is small enough
(2.2 × 10⁸ legacy plates) that anyone holding the key can enumerate it offline in
minutes. The key must live in a KMS or HSM and never in the database. Overselling
this as encryption would be worse than not having it, because it produces false
confidence.

## Usage

```bash
pip install -e "services/api[dev]"
```

```bash
scanner-api genkey
```

```bash
scanner-api initdb
```

```bash
scanner-api adduser --username alice --role investigator
```

```bash
scanner-api serve --port 8000
```

Re-verify a node's stored chain independently of ingest-time checks — what an
auditor runs:

```bash
scanner-api verify-node --node-id KTM-BAL-01:KTM-BAL-01-N
```

## Roles

| Role | Can |
|---|---|
| `operator` | Live feed, alerts, review queue. No bulk search |
| `investigator` | Search, export, watch-lists — always with a stated reason |
| `admin` | Nodes, users, retention, erasure |
| `auditor` | The access log, **and nothing else** |

`auditor` is deliberately disjoint from every other permission. An oversight role
that can also perform the activity it oversees is not oversight.

Licence: Apache-2.0.
