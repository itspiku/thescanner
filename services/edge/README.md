# scanner-edge

The edge agent for [TheScanner](https://github.com/itspiku/thescanner). Runs on a
box at the roadside.

```
frames --> detect --> track --> zone sessions --> events
                        |
                    accumulate crops
                        |
                 (passage ends)
                        |
                select informative crops
                        |
                 recognise + fuse  --> one read per passage
                        |
                redact --> sign --> durable queue --> uplink
```

## Why it is shaped like this

Two facts about the deployment drive nearly every decision.

**Nepal has load-shedding and unreliable connectivity.** A node must keep working
with no uplink for days, so the durable local queue is the *primary* write path
and the uplink is a best-effort drain of it. Every read is safe on disk before
any network call happens; a link failure costs latency, never evidence.

**Reads may become evidence.** Origin and integrity have to be provable to
someone hostile, later. So every event is signed with the node's Ed25519 key and
hash-chained to its predecessor, *before* it reaches durable storage. Altering or
deleting a historical read breaks every link after it, so tampering is not merely
detected but located:

```bash
scanner-edge verify --config node.yaml
```

```
KTM-BAL-01-N: 41,207 events, sequence 1-41207: BROKEN -- verification failed at sequence 39114
```

"The evidence is invalid" is not a useful thing to tell a court. "Read 39,114
from this camera was altered" is.

## Design notes

**One read per passage, not per frame.** A vehicle crossing the frame is
recognised once, from a fused set of crops. This is what keeps national volume in
the millions rather than the hundreds of millions, and it is why fusion lives
here rather than centrally — fusing after transmission would mean shipping forty
crops per vehicle over the scarcest resource in the system.

**Crop selection is diversity-aware.** The last N frames are the worst ones (the
vehicle is sweeping past fastest, so blur and yaw are at their peak), and the top
N by quality are all adjacent and share their blur. Selection is greedy over
quality with a temporal-diversity penalty, so fusion gets genuinely independent
looks.

**Tracking uses BYTE association.** Low-confidence detections are usually a real
vehicle that got harder to see, and dropping them is what makes a track fragment.
A vehicle that fragments into three tracks produces three reads of one passage,
three zone sessions and three watch-list alerts.

**Zone sessions distinguish "left" from "lost".** A session that never closes
means the vehicle is still inside, or left by an unmonitored exit, or the exit
read was missed — operationally different things. "Went in and never came out" is
precisely the event worth alerting on at a car park or a border post, so it is
never quietly reported as a clean exit.

**Faces are blurred on the node**, before anything reaches the queue. Recognition
reads the unredacted frame; only stored imagery is redacted. And this is
explicitly *not* face recognition — faces are detected in order to destroy them,
and no descriptor is ever computed. See
[`docs/security-and-privacy.md`](../../docs/security-and-privacy.md).

## Usage

```bash
pip install -e "services/edge[dev]"
```

```bash
scanner-edge init --out node.yaml
```

```bash
scanner-edge enrol --config node.yaml
```

```bash
scanner-edge run --config node.yaml
```

```bash
scanner-edge verify --config node.yaml
```

## Detector status

A trained Nepali vehicle/plate detector is Phase 2.1 and **does not exist yet** —
training one needs annotated road scenes, and the public Nepali data is plate
crops rather than scenes. Until it lands, the node falls back to motion detection
(background subtraction), which on a fixed pole-mounted camera finds moving
vehicles perfectly adequately and makes the whole pipeline runnable end to end
today.

That is a bootstrap, and the agent says so on startup. The production path is
`OnnxDetector`, which consumes any exported ONNX detector — D-FINE, RT-DETRv2,
DEIM, all Apache-2.0 — so the choice of weights stays a deployment decision
rather than something baked into the source tree. Ultralytics YOLO is
deliberately excluded: AGPL-3.0 is a procurement blocker for a system a
government operates.

Motion detection also has a genuine production use as a **gate** in front of a
neural detector: skipping frames with no motion cuts inference cost dramatically
on a road that is empty most of the night.

Licence: Apache-2.0.
