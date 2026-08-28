# TheScanner — delivery plan

A production ANPR and vehicle-movement system for Nepal: reads both Nepali plate
systems from road cameras, records when a vehicle entered and left a zone, and
does it on hardware a Nepali municipality can afford, under Nepali privacy law,
with data that stays in the country.

**Status legend:** ✅ done · 🚧 in progress · ⬜ not started · ⏸ blocked

---

## 1. What this system is

Cameras watch a road. For every vehicle that passes:

- detect the vehicle and its plate
- track it across frames
- read the plate — legacy Devanagari **or** embossed — fusing evidence from
  every frame in the track
- classify the vehicle (type, colour, and where possible make/model)
- record an **entry** event when it enters a defined zone and an **exit** event
  when it leaves, with dwell time
- screen every read against a watch-list and raise alerts
- keep all of it queryable, auditable, and defensible in court

### Success criteria

These are the numbers the project is judged on. They are deliberately
conservative — the best result on the ICPR 2026 low-resolution plate benchmark
was 82.13%, so any target near 99% on degraded imagery would be dishonest.

| Metric | Target | Measured on |
|---|---|---|
| Plate detection recall | ≥ 98% | NepalPlate-Bench, plates ≥ 40 px wide |
| Full-plate read accuracy, good conditions | ≥ 96% | Bench, daylight, ≥ 80 px plate width — *synthetic proxy: 98.5% at ≥130 px, 92.4% at 90–130 px* |
| Full-plate read accuracy, degraded | ≥ 80% | Bench, night/rain/blur subset, track-level — *synthetic proxy: 51.4% single-frame; track fusion not yet measured* |
| False-positive rate at HIGH confidence | ≤ 0.5% | Bench — a wrong plate asserted confidently is the worst failure mode. **Currently 2.2% on synthetic; not met** |
| Zone entry/exit pairing accuracy | ≥ 97% | Bench video tracks |
| Edge throughput | ≥ 4 camera streams @ 15 fps per edge node | Jetson Orin Nano 8 GB |
| End-to-end latency, capture → alert | ≤ 2 s p95 | Integration test |
| Offline tolerance | ≥ 72 h with no network, zero read loss | Chaos test |

**Non-goals.** Face recognition (deliberately excluded — see
[security-and-privacy.md](security-and-privacy.md)). Automated fine issuance
without human review. Vehicle tracking outside camera zones. Integration with
the RFID chips in embossed plates (a separate procurement).

---

## 2. Architecture in one paragraph

Four stages, borrowed in shape from the UK's national ANPR platform and scaled
down by two orders of magnitude to fit Nepal's realistic volume:
**Edge → Ingest → Screen → Exploit**. Edge nodes at each site do detection,
tracking, recognition and fusion locally, and hold a signed store-and-forward
queue so a site survives power and network loss. Ingest is NATS JetStream.
Screen matches against watch-lists. Exploit is a single PostgreSQL instance with
TimescaleDB (time-series reads) and pgvector (appearance search) — one database
rather than three, which is the main reason the whole system fits on modest
hardware. Full detail in [architecture.md](architecture.md).

---

## Phase 0 — Domain core ✅

The foundation everything else depends on. Complete.

| Deliverable | Status |
|---|---|
| Machine-readable spec of both plate systems | ✅ `packages/nepal_plate/nepal_plate/spec.py` |
| 34-token Devanagari + 36-token Latin unified vocabulary | ✅ verified against published charsets |
| Finite-state grammars for both layouts | ✅ `grammar.py` |
| Grammar-constrained CTC beam search | ✅ `decode.py` |
| Plate-colour decoding prior | ✅ `decode.colour_slot_bonus` |
| Multi-frame track fusion with per-field consensus | ✅ `fuse.py` |
| Tolerant parser + canonicalisation | ✅ `parse.py` |
| Test suite | ✅ 38 tests passing |

**Acceptance evidence.** Tests directly demonstrate the two core mechanisms:
argmax decoding picks an illegal glyph on a blurred plate and produces a
non-plate; the grammar eliminates it; the colour prior then recovers the correct
class letter *despite it holding the least raw probability mass of the three
candidates*. Grammar constraint measured at 13.0 bits of search-space reduction
for legacy plates, 16.4 for embossed.

---

## Phase 1 — Data 🚧

The synthetic generator (1.3–1.5) is built and tested; real-data acquisition and
the evaluation benchmark are not started.

The binding constraint on the whole project. See
[research/datasets.md](research/datasets.md).

| # | Deliverable | Acceptance criteria | Status |
|---|---|---|---|
| 1.1 | Download, verify and licence-check every public Nepali plate dataset | Written record of size, licence, annotation quality per dataset; anything licence-incompatible excluded and documented | ⬜ |
| 1.2 | Unified annotation schema + converters | All acquired data converted to one schema, round-trip tested | ⬜ |
| 1.3 | **Synthetic plate renderer** (`packages/synthplate`) | Renders every legal plate in both systems, all six legacy colour schemes, correct dimensions, per-character boxes | ✅ |
| 1.4 | Physically-grounded degradation pipeline | n-stage random composition ordered by physics; single difficulty knob; calibrated ground-truth quality score | ✅ |
| 1.5 | Track synthesis | Coherent N-frame sequences: shared scene, smooth 1/distance width trajectory, mid-pass quality peak | ✅ |
| 1.6 | 500k-crop synthetic corpus | Balanced across zone × class × ownership × system; reproducible from a seed. CLI done; corpus not yet generated at scale, and blocked on 1.9 | 🚧 |
| 1.7 | **NepalPlate-Bench** | 3–5k real plates + ≥500 video tracks, stratified; never used for training; inter-annotator agreement ≥ 98% | ⬜ |
| 1.8 | Verify the plate spec against DoTM primary sources | Every table in `spec.py` confirmed or corrected | ⬜ |
| 1.9 | **Obtain authentic plate typefaces** | FE-Schrift for embossed, a Devanagari plate face for legacy. The generator currently falls back to Arial Narrow Bold / Nirmala UI and says so in every manifest. Training a deployed model on fallback-rendered glyphs would fail silently | ⬜ |

**Risk.** 1.7 requires real-world collection with legal authorisation and is the
long pole of the entire project. Start the authorisation conversation in
parallel with Phase 1, not after it.

---

## Phase 2 — Models 🚧

**Trained and evaluated.** A 1.93 M-parameter multi-task recogniser on 50,000
synthetic plates: **83.6% full-plate exact match**, 97.9% colour, 0.070 quality
MAE on the held-out split.

> **The ablation disproved this project's headline claim.**
> Grammar-constrained decoding is worth **+0.001** over greedy, and the colour
> prior **0.000**, because a model trained only on legal plates has already
> internalised the grammar. Full account, and what the mechanisms *are* worth,
> in [research/findings-phase2.md](research/findings-phase2.md). Two rework
> items below follow directly from it.


**Licensing policy: permissive only.** Ultralytics YOLO is AGPL-3.0, which is a
procurement landmine for a government deployment — it would oblige the
government to publish derivative source. Primary detector is therefore
**D-FINE** or **RT-DETRv2** (both Apache-2.0). An Ultralytics adapter exists
behind a flag for rapid prototyping only, and is excluded from release builds.

| # | Deliverable | Acceptance criteria |
|---|---|---|
| 2.1 | Vehicle + plate detector | ≥98% plate recall at ≥40 px on Bench; Apache-2.0 lineage |
| 2.2 | Plate colour + script classifier | ≥99% on the 7-way colour task; this routes the grammar so errors are expensive |
| 2.3 | Dual-script CTC recogniser | Shared visual encoder, single 71-token head; meets the accuracy table in §1 |
| 2.4 | Crop quality estimator | Predicts per-frame read reliability; drives fusion weights. Rank correlation ≥0.8 with actual correctness |
| 2.5 | Vehicle attribute model | Type ≥95%, colour ≥90%; enables plate–vehicle consistency checks |
| 2.6 | Appearance embedding | For cross-camera re-identification and pgvector search |
| 2.7 | Optional plate super-resolution | Only ships if it beats fusion-alone on the Bench degraded subset |
| 2.8 | Export + quantisation | ONNX + INT8; TensorRT for Jetson, OpenVINO for x86. Accuracy loss ≤1% |
| 2.9 | Evaluation harness | Reports constrained-vs-greedy delta, per-stratum accuracy, calibration curves | ✅ |

### Status

| # | Deliverable | Status |
|---|---|---|
| 2.1 | Vehicle + plate detector | ⬜ blocked on annotated road scenes (1.1/1.7); ONNX adapter and motion-detection bootstrap shipped in `services/edge` |
| 2.2 | Plate colour + script classifier | ✅ 97.9% (target ≥99%; short, but it feeds a prior now known to contribute ~0) |
| 2.3 | Dual-script CTC recogniser | ✅ 83.6% exact match on synthetic held-out |
| 2.4 | Crop quality estimator | ✅ MAE 0.070 |
| 2.5 | Vehicle attribute model | ⬜ |
| 2.6 | Appearance embedding | ⬜ |
| 2.7 | Plate super-resolution | ⬜ — deferred; fusion is the stronger lever and enhancement must beat it to ship |
| 2.8 | Export + quantisation | ✅ ONNX export, numerically verified against PyTorch; INT8 ⬜ |
| 2.9 | Evaluation harness | ✅ per-stratum ablation + calibration table |

### Rework arising from the evaluation

| # | Deliverable | Why |
|---|---|---|
| 2.10 | **Isolate and close the single-row gap** | Single-row plates score 70.2% against two-row plates' 94.9%. Per-glyph geometry (13.6 px vs 20.4 px per glyph) explains part of it, but the gap persists at matched per-glyph resolution, so the cause is not established. Diagnose before changing the architecture — shipping on an unverified hypothesis is exactly how the grammar claim went wrong |
| 2.11 | **Tighten the HIGH confidence band** | HIGH-confidence false-positive rate is **2.2%** against the ≤0.5% criterion in §1. This is the most operationally consequential number in the system and it currently fails |

**Hardware note.** Training happens on a 6 GB RTX 4050. That fits the detector
and recogniser at the sizes we need with gradient accumulation, and it is a
useful forcing function: a model that trains on 6 GB will run at the edge.

---

## Phase 3 — Edge pipeline 🚧

Built and tested (`services/edge`), against synthetic and replayed input. Not
yet run against a live RTSP camera.

| # | Deliverable | Status |
|---|---|---|
| 3.1 | RTSP ingest with reconnection and backlog dropping | ✅ code; ⬜ throughput unverified on Jetson |
| 3.2 | Detect → ByteTrack → crop selection | ✅ BYTE two-round association; diversity-aware selection with a hard legibility gate |
| 3.3 | Recognise + fuse per track | ✅ one read per passage, plus a provisional read mid-passage so alerts fire in time |
| 3.4 | Zone entry/exit session engine | ✅ hysteresis, bottom-centre reference point, and `exited` / `track_lost` / `timed_out` distinguished |
| 3.5 | Signed store-and-forward queue | ✅ Ed25519 + hash chain, SQLite WAL, chain resumed across restart; ⬜ 72 h soak untested |
| 3.6 | On-device privacy | ✅ face + cabin redaction before anything reaches durable storage |
| 3.7 | Health + telemetry | 🚧 queue/uplink status shipped; camera tamper and thermal ⬜ |
| 3.8 | OTA model update | ⬜ |

---

## Phase 4 — Platform 🚧

Built and tested (`services/api`), including an end-to-end test that drains a
real signed edge queue through the real uplink into the real API. Not yet run
against PostgreSQL with TimescaleDB — the schema is dialect-conditional and the
suite runs on SQLite.

| # | Deliverable | Status |
|---|---|---|
| 4.1 | Postgres + TimescaleDB + pgvector schema | ✅ schema, hypertable, compression policy and continuous aggregate defined and applied conditionally; ⬜ unverified against a live Postgres |
| 4.2 | Ingest | ✅ HTTP ingest, idempotent on `(node_id, sequence)`, chain gaps recorded not closed. NATS JetStream deferred: HTTP is sufficient at Nepal's volume and one fewer component to operate |
| 4.3 | Screening service | ✅ HMAC-indexed match, historical backfill on new entries, non-actionable below HIGH; ⬜ latency unmeasured |
| 4.4 | **Evidence chain** | ✅ verified on ingest; failures stored and flagged rather than dropped; `scanner-api verify-node` re-checks stored reads independently |
| 4.5 | Search & investigation API | ✅ plate, partial, zone sessions, convoy, occupancy, statistics; ⬜ appearance-similarity (needs 2.6) |
| 4.6 | Cloned-plate detection | ✅ all three signals, plus a registry cross-check. Impossible-movement uses HIGH reads only — a misread digit produces exactly that signal |
| 4.7 | RBAC + mandatory reason-for-access | ✅ four roles, auditor disjoint from all data access; reason logged before the query runs so a crashed query still leaves a record |
| 4.8 | Retention & erasure jobs | ✅ per-row `expires_at` so overdue rows are queryable, not just cron-logged; erasure keeps the access log; legal holds are explicit and time-bounded |
| 4.9 | Registry integration adapter | 🚧 cache table and cross-check implemented; ⬜ no DoTM source to plug in (open question §4.2) |

---

## Phase 5 — Applications 🚧

Built (`services/web`), typechecked and built clean, and verified end to end
against a live API seeded with `scripts/seed_demo.py`: login, dashboard, live
feed rendering both plate systems, reason-gated search matching a romanised
query against a Devanagari read, and the access controls refusing what they
should.

| # | Deliverable | Status |
|---|---|---|
| 5.1 | Web console — React + TypeScript | ✅ builds to ~165 kB JS; zero external requests (no CDN, no font host, no map provider). MapLibre deferred: it needs self-hosted tiles, which is a procurement item, not a code one |
| 5.2 | Live feed, alert queue, zone occupancy | ✅ |
| 5.3 | Investigation workspace | 🚧 plate / partial / convoy search and zone sessions done; ⬜ timeline view and evidence export |
| 5.4 | **Human review queue** | ✅ queue, crop shown unsmoothed, confirm-or-correct; ⬜ corrections are stored but not yet fed back into a training set |
| 5.5 | Nepali (नेपाली) and English UI | ✅ Nepali is the default, Devanagari numerals for counts, plate text never translated |
| 5.6 | Admin: cameras, zones, watch-lists, users, retention | 🚧 all available via the API and CLI; ⬜ no admin UI |

---

## Phase 6 — Hardening and deployment ⬜

| # | Deliverable | Status |
|---|---|---|
| 6.1 | Data Protection Impact Assessment under Privacy Act 2075 |
| 6.2 | Threat model + independent security review + penetration test |
| 6.3 | Adversarial testing — obscured, altered, cloned and absent plates |
| 6.4 | Load and chaos testing — network partition, power loss, clock skew |
| 6.5 | Docker Compose (single site) + Helm (national) | 🚧 Compose, three Dockerfiles, nginx CSP and a documented secrets flow shipped in `deploy/`; ⬜ Helm chart, and nothing has been run against a live PostgreSQL |
| 6.6 | Operator runbooks and training material, in Nepali |
| 6.7 | Bias and error audit across zones, vehicle classes and ownership types |

---

## 3. Risks

| Risk | Impact | Mitigation |
|---|---|---|
| **No real evaluation data** | Cannot know if anything works | NepalPlate-Bench is Phase 1's highest priority; start collection authorisation immediately |
| Synthetic-to-real gap | Model learns the generator | Never evaluate on synthetic; n-stage random degradation composition; real fine-tune |
| Spec inaccuracies | Grammar rejects valid plates | Phase 1.8 primary-source verification; grammar failures logged and reviewed, never silently dropped |
| Grammar snaps damaged plates to wrong-but-legal reads | Wrongful enforcement | `repaired` flags, confidence bands, adversarial testing, human review of anything below HIGH |
| AGPL contamination | Government cannot deploy | Permissive-only policy, enforced in CI |
| Privacy Act non-compliance | Legal challenge; the 2018 plate rollout was already halted once by the Supreme Court | DPIA before deployment; privacy-by-design; no face recognition |
| Scope — this is a large system | Never ships | Phases are independently useful; Phase 0–3 alone is a working single-site ANPR |

---

## 4. Open questions

Answers change the work; recorded here rather than guessed at.

1. **Deployment target** — municipal (Kathmandu Valley) or national from the
   start? Changes the platform tier significantly, not the edge tier.
2. **Registry access** — will DoTM registration data be available? Without it,
   plate–vehicle consistency checking is limited to what the vision model can
   infer.
3. **Existing camera estate** — integrate with the 297-camera Valley control
   room, or deploy independent nodes?
4. **Edge hardware budget** — Jetson Orin Nano (~$250/site) vs. x86 mini-PC with
   OpenVINO (~$180/site) vs. reuse of existing DVR infrastructure.
5. **Who operates it** — Traffic Police, DoTM, or municipality? Determines the
   RBAC model and the retention policy.

---

## 5. Sequencing

Phases 1 and 2 interleave: the synthetic generator (1.3–1.6) unblocks model
training (2.3) long before real data collection (1.7) completes. Phase 3 can
start against synthetic-trained models. The critical path runs
**1.3 → 1.6 → 2.3 → 3.3**, with 1.7 as the parallel long pole that gates any
claim about accuracy.

A useful intermediate milestone: **Phases 0–3 constitute a working single-site
ANPR** that can be demonstrated end-to-end on one camera, independent of the
national platform.
