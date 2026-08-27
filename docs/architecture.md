# Architecture

Four stages — **Edge → Ingest → Screen → Exploit** — borrowed in shape from the
UK's national ANPR platform and scaled down by two orders of magnitude to fit
Nepal's realistic volume and budget.

```
   ┌─── SITE ────────────────────────────────────────┐
   │  camera(s) ──RTSP──▶ edge node                  │
   │                       detect → track → select   │
   │                       recognise → fuse          │
   │                       zone sessions             │
   │                       face blur                 │
   │                       signed store-and-forward  │
   └───────────────────────────┬─────────────────────┘
                               │  intermittent link
                    ┌──────────▼──────────┐
                    │  INGEST  NATS JS    │  at-least-once, idempotent dedup
                    └──────────┬──────────┘
                    ┌──────────▼──────────┐
                    │  SCREEN             │  watch-list, cloned-plate checks
                    └──────────┬──────────┘
                    ┌──────────▼──────────┐
                    │  EXPLOIT            │  Postgres + TimescaleDB + pgvector
                    │  API · console      │  MinIO for imagery
                    └─────────────────────┘
```

---

## Why these choices

### Edge-first, not cloud-first

Nepal has load-shedding and unreliable rural connectivity. A system that stops
recording when the link drops is not deployable. Each edge node does the full
recognition pipeline locally and holds results in a **signed, ordered,
store-and-forward queue** — target 72 hours offline with zero loss.

This also collapses bandwidth by ~1000×: a site uploads structured read events
and a handful of selected crops, not continuous video.

### One database, not three

The obvious design uses Postgres for entities, a time-series store for reads,
and a vector database for appearance search. Instead:

- **PostgreSQL** — entities, users, zones, watch-lists
- **+ TimescaleDB** — reads as a hypertable, with native compression and
  continuous aggregates for zone occupancy
- **+ pgvector** — vehicle appearance embeddings for cross-camera search

One engine to operate, back up, secure and staff. For a government deployment in
a country with a small pool of specialist operators, this is worth more than the
marginal performance of purpose-built stores. Revisit only if measurement says
so.

### NATS JetStream, not Kafka

Kafka is the reflex choice and the wrong one here. NATS JetStream is a single
small binary, runs comfortably at the edge, and its leaf-node model is built for
exactly the intermittent-connectivity topology we have. Kafka's throughput
advantage is irrelevant at Nepal's volume — single-digit millions of reads/day
nationally, against Kafka's design point of millions per second.

### Permissive licences only

Ultralytics YOLO is AGPL-3.0. Deploying it inside a government system creates a
copyleft obligation over derivative work — a procurement blocker. Primary
detector is **D-FINE** or **RT-DETRv2**, both Apache-2.0. An Ultralytics adapter
sits behind a flag for prototyping and is excluded from release builds, enforced
in CI.

### No third-party map or cloud services

MapLibre with self-hosted tiles, MinIO for object storage, self-hosted identity.
Vehicle movement data for a whole country is a national security asset; it does
not leave the country's infrastructure.

---

## Edge node pipeline

```
RTSP ──▶ HW decode ──▶ detect (vehicle + plate)
                            │
                            ▼
                       ByteTrack ──▶ track buffer
                            │
                            ▼
                   crop selection ── the N most *informative* crops,
                            │        not the N most recent
                            ▼
              ┌─────────────┴─────────────┐
              ▼                           ▼
     colour / script classifier    quality estimator
              │                           │
              └─────────────┬─────────────┘
                            ▼
                 CTC recogniser (71-token unified vocab)
                            │
                            ▼
        grammar-constrained decode + colour prior   ← nepal_plate.decode
                            │
                            ▼
              multi-frame fusion (per-field)        ← nepal_plate.fuse
                            │
                            ▼
                    one read per vehicle passage
```

Two details that matter:

**Crop selection.** Naively feeding every frame to the recogniser wastes most of
the compute budget on redundant near-duplicates. The quality estimator ranks
crops and the pipeline recognises a diverse, informative subset — which is what
makes 4 streams fit on an Orin Nano.

**One read per passage.** A vehicle crossing the frame produces *one* read
event, not forty. This is what keeps national volume in the millions rather than
the hundreds of millions, and it is why the fusion stage belongs at the edge
rather than centrally.

### Zone sessions

Zones are polygons over the camera's ground plane. A vehicle entering opens a
session; leaving closes it, yielding a dwell time. Sessions that never close are
aged out and flagged — an unclosed session is itself information (the vehicle
is still inside, or it left by an unmonitored exit, or a read was missed).

---

## Recognition model

A single recogniser handles both scripts: shared visual encoder, one CTC head
over a **71-token unified vocabulary** (blank + 34 Devanagari + 36 Latin). The
colour/script classifier routes to the appropriate grammar at decode time
rather than switching models, so there is one model to train, quantise, deploy
and version.

Devanagari tokens are **atomic visual units**, not Unicode codepoints — `बा` is
one class, not `ब` + `ा`. This matters: zone codes are a closed set of 14 and
are rendered as single visual units on the plate, so modelling them atomically
matches both the typography and the grammar.

---

## Security architecture

Detail in [security-and-privacy.md](security-and-privacy.md). The load-bearing
pieces:

- **Per-node identity.** Every edge node holds an Ed25519 keypair; every read
  event is signed at capture. A read's origin is provable.
- **Hash-chained read log.** Each event carries the hash of its predecessor.
  Tampering with historical reads is detectable and provable — which is what
  makes a read defensible as evidence.
- **Mandatory reason-for-access.** No read is readable without a logged,
  attributable purpose. The access log is itself append-only.
- **Tiered retention.** Short at the edge, longer centrally, automatic expiry.
- **Privacy at capture.** Faces blurred on the edge node before any image
  leaves it.

---

## Deployment tiers

| Tier | Hardware | Scope |
|---|---|---|
| **Single site** | 1 edge node + Docker Compose | One junction; fully functional standalone |
| **Municipal** | N edge nodes + one server | Kathmandu Valley scale |
| **National** | N edge nodes + HA Postgres + Helm/K8s | All provinces |

The edge tier is identical across all three. Only the platform tier changes,
which means a site deployed on day one keeps working unchanged as the system
scales.

---

## Open architectural questions

Tracked in [PLAN.md §4](PLAN.md#4-open-questions): deployment scope, DoTM
registry access, whether to integrate with the existing 297-camera Valley
estate, edge hardware choice, and operating authority.
