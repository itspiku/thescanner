# Security and privacy

This system watches public roads and records the movements of identifiable
vehicles on behalf of the state. That is a serious thing to build, and the
design has to earn it. This document states the threat model, the legal
obligations, and the deliberate limits on what the system will do.

---

## 1. Legal basis — Nepal's Privacy Act 2075 (2018)

The [Individual Privacy Act, 2075 (2018)](https://lpr.adb.org/resource/privacy-act-2075-2018-nepal),
in force since 18 September 2018, governs collection and processing of personal
data in Nepal. The relevant obligations:

- **Consent or lawful authority** is required for collection, recording,
  disclosure or processing of personal information.
- **Data subject rights**: to be informed, to access, to rectification, to
  erasure, to object to processing of sensitive personal data, and to complain
  and seek compensation.
- **Penalties**: up to 3 years' imprisonment and NPR 30,000 for unlawful
  collection, disclosure or processing.

A registration plate is not inherently personal data, but a plate *linked to a
registry record* identifies a person, and a *time series of a plate's locations*
reveals their movements. The system therefore treats reads as personal data by
default.

### This is contested ground in Nepal already

The 2018 embossed-plate rollout was **halted by a Supreme Court order**, partly
over the privacy implications of the embedded RFID chips. The mandate was
rescinded in December 2019. Vehicle surveillance in Nepal has already been to
court once. A deployment that cannot answer privacy questions convincingly
should expect to go there again.

### Consequences for the design

| Obligation | Implementation |
|---|---|
| Purpose limitation | Every access requires a logged, attributable reason. Reads cannot be browsed casually |
| Data minimisation | One read per vehicle passage, not per frame. Faces blurred at capture. Full video never leaves the edge node |
| Storage limitation | Tiered automatic retention with hard expiry; no indefinite storage |
| Right of erasure | Erasure requests executed across reads, imagery and derived aggregates, and the execution itself audited |
| Right of access | A data subject can be given the reads pertaining to their vehicle |
| Accountability | Append-only audit log; DPIA before deployment |

---

## 2. Deliberate limits

These are design decisions, not missing features. They exist so the system's
capability stays inside its justification.

- **No face recognition.** The system identifies *vehicles*, not people. Faces
  are detected only in order to blur them. This is the single most important
  limit here: the technical distance from a national ANPR to a national face
  surveillance network is short, and refusing to take that step has to be
  architectural, not a policy note. (The Valley expansion plan does include
  face-recognition cameras at 10 locations — that is a separate system, and
  this one does not become it.)
- **No automated enforcement.** The system produces evidence; a human decides.
  No fine is issued on a machine read alone.
- **No behavioural inference.** No scoring, prediction, or profiling of drivers.
- **No cross-linking to unrelated databases** beyond the vehicle registry, and
  that link is a pluggable adapter that can be disabled.
- **No data export outside Nepal.** Self-hosted throughout.

---

## 3. Threat model

### Assets

1. The read database — a national record of vehicle movements
2. Watch-lists — reveal who is under investigation
3. Captured imagery
4. Edge node signing keys — forging these forges evidence
5. Model artefacts

### Adversaries and mitigations

| Adversary | Goal | Mitigation |
|---|---|---|
| **Insider** with legitimate access | Look up a spouse, a journalist, a political rival | Mandatory reason-for-access; append-only access log; anomaly detection on query patterns; least-privilege RBAC. *This is the most likely real attack and the hardest to stop* |
| **External attacker** | Exfiltrate movement data | Network segmentation; encryption in transit and at rest; no internet-exposed edge nodes; short-lived credentials |
| **Evidence tamperer** (possibly insider) | Alter or delete a read to protect someone | Hash-chained append-only log; per-node Ed25519 signatures; independent chain-head witnessing |
| **Plate spoofer** | Evade or frame | Plate–vehicle class consistency; colour–class consistency; physically-impossible-movement detection between sites |
| **Physical attacker** | Disable or feed false input to a camera | Tamper and defocus detection; signed node identity; injected frames cannot be signed by a legitimate node |
| **Supply-chain attacker** | Backdoor a model or dependency | Pinned hashes; signed model bundles; SBOM; reproducible builds |

### Explicitly out of scope

Compromise of the vehicle registry itself, and lawful-but-unethical use
authorised at a policy level. The second is a governance problem, and no
technical control substitutes for the oversight that should accompany it — but
the reason-for-access log at least makes such use *visible*.

---

## 4. Security controls

### Identity and access

- OIDC with mandatory MFA; WebAuthn preferred
- Role-based access: operator, investigator, administrator, auditor
- Reason-for-access prompt on every query, stored with the query
- Auditors can read the audit log and nothing else
- Sessions time-limited; privileged actions re-authenticated

### Evidence integrity

Each read event carries: the capturing node's Ed25519 signature, a hash of the
associated imagery, and the hash of the preceding event in that node's chain.
Chain heads are periodically published to an append-only witness store.

This makes three claims defensible in court: the read came from *this* camera,
at *this* time, and has not been altered since.

### Data protection

- TLS 1.3 everywhere; mutual TLS between edge nodes and ingest
- Encryption at rest for database and object storage
- Plate values additionally stored as a keyed HMAC pseudonym, so a database
  disclosure does not immediately yield a plaintext movement history
- Edge node disks encrypted — a stolen node must not yield its queue

### Retention

| Data | Edge | Central |
|---|---|---|
| Raw video | not persisted | never |
| Plate crops | 7 days | per policy, default 90 days |
| Context images | 7 days | per policy, default 90 days |
| Read events | 7 days | per policy, default 12 months |
| Watch-list hits | 7 days | per case, subject to review |
| Audit log | — | 7 years |

Defaults follow the UK NAS precedent (12 months central, 7 days edge) and must
be confirmed against Nepali policy before deployment. Expiry is enforced by a
job that runs regardless of operator action.

---

## 5. Fairness and error

An ANPR's errors are not evenly distributed, and uneven errors in an enforcement
system are a harm.

- **Accuracy must be reported per stratum** — per zone code, ownership class,
  vehicle type, time of day, weather — not as a single headline number. A system
  that is 96% accurate overall but 70% on motorcycles disproportionately
  penalises the vehicles poorer people drive.
- **Confidence bands are operational.** Only HIGH-confidence reads may trigger
  automated alerts. Everything else goes to human review.
- **Grammar repair is disclosed.** When the grammar overrides what the pixels
  said, the read is flagged `repaired` and cannot be HIGH confidence. An officer
  must be able to see that the system inferred rather than observed.
- **Absence of a read is not evidence.** A vehicle not seen may have been
  missed. The system must never be used to assert that a vehicle was *not*
  somewhere.

---

## 6. Before deployment

Mandatory, tracked as Phase 6 in [PLAN.md](PLAN.md):

1. Data Protection Impact Assessment under Privacy Act 2075
2. Independent security review and penetration test
3. Adversarial testing — obscured, damaged, altered and cloned plates
4. Bias and error audit across all strata above
5. Published retention and access policy
6. Operator training including the legal limits, in Nepali
7. A named accountable authority and a complaints route for data subjects

---

## Sources

- [The Privacy Act, 2075 (2018) — ADB Law and Policy Reform](https://lpr.adb.org/resource/privacy-act-2075-2018-nepal)
- [Data protection laws in Nepal — DLA Piper](https://www.dlapiperdataprotection.com/index.html?t=law&c=NP)
- [Individual Privacy Act 2075 (2018) — Sherpa Law Associates](https://www.sherpalawassociates.com/resources/articles/cmkqogl24000iktrvmub1a78g)
- [UK National ANPR Service DPIA](https://www.statewatch.org/media/1893/uk-home-office-anpr-network-dpia-2-21.pdf)
- [National ANPR Service: data subject rights — GOV.UK](https://www.gov.uk/government/publications/national-anpr-service-nas-data-subject-rights)
