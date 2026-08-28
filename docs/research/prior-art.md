# Prior art — what exists, and where the gap is

Survey conducted August 2026. Purpose: avoid rebuilding what works, and locate
the places where a Nepal-specific system can be genuinely better rather than
merely different.

---

## 1. What already exists in Nepal

### Deployed systems

Kathmandu Valley Traffic Police already operate ANPR. As of mid-2026:

- **6 ANPR cameras** capable of reading both Devanagari and embossed plates and
  of measuring speed, feeding a control room with **297 cameras** total.
- Sited at Munibhairabh (4), New Bus Park (2), Maharajgunj (4), with further
  installations at Balkumari, Mahalaxmisthan and Dhobighat on the Ring Road.
- **15,954 drivers prosecuted** via ANPR between Baishakh 2082 and Asar 20, 2083.
- Planned expansion to ~170 cameras, including face recognition at 10 locations
  and AI-based cameras at 150.

**Implication for this project.** This is not a greenfield problem — it is an
*integration and scale* problem. The existing deployment is a small number of
expensive proprietary cameras at fixed sites. The gap is a system that runs on
commodity hardware, can be deployed at hundreds of sites on a Nepali public
budget, and keeps its data inside the country. That framing drives most of the
architecture decisions in [`../architecture.md`](../architecture.md).

### Published Nepali ANPR research and projects

| Work | Approach | Notes |
|---|---|---|
| [Pant et al., 2015 — ANPR with SVM](http://ashokpant.github.io/publications/ashok_2015_automatic.pdf) | Classical features + SVM | The original reference point for Nepali plates |
| [Prasanna1991/LPR](https://github.com/Prasanna1991/LPR) | Dataset | 2,033 cropped character images, 12 classes, Bagmati motorbikes |
| [JuJu2181/annpr](https://github.com/JuJu2181/annpr) | YOLOv4 detect → char segment → CNN classify | Segmentation-based |
| TRaiFIC | Multi-model pipeline | Detect → segment → classify |
| [Sanjaya Subedi — Nepali LPR with deep learning](https://sanjayasubedi.com.np/deeplearning/nepali-license-plate-recognition-with-deep-learning/) | CNN pipeline | Good write-up of the practical difficulties |
| [arXiv 2606.28946 — Character Recognition of Nepali Number Plate](https://arxiv.org/abs/2606.28946) | YOLO detect + CNN over 34 Devanagari classes | Reports up to **93%**; the current published bar |

**Common limitations across all of them:**

1. **Character segmentation.** Most segment the plate into individual characters
   and classify each independently. Segmentation is the most fragile stage of a
   classical OCR pipeline and it fails first under blur — exactly the condition
   that matters.
2. **Single script.** Devanagari *or* embossed, essentially never both, despite
   both being on the road today.
3. **Single frame.** Evaluated on stills, not on video tracks.
4. **No structural prior.** The plate grammar is used, if at all, as a
   post-hoc regex — after the uncertainty has been discarded.
5. **Colour discarded.** Several pipelines convert to greyscale as step one,
   throwing away the ownership signal entirely.
6. **Tiny evaluation sets**, usually drawn from one zone (Bagmati) and one
   vehicle type (motorcycles), in daylight.

---

## 2. What exists globally

### National-scale architecture — the UK reference

The UK's National ANPR Service / National Strategic ANPR Platform is the most
thoroughly documented national deployment, and its shape is worth copying even
at a hundredth of the scale:

- Four-stage pipeline: **Gateway → Ingest → Screen → Exploit**
- Ingest sustains up to **12,000 events/second**; ~**350 million reads/day**
- Screen matches every read against a live vehicle-of-interest state store of
  **45+ million records**
- Exploit serves sub-millisecond alerts, over **36 billion reads** and **70
  billion images** for investigation
- Retention: **12 months** centrally, **7 days** on local management servers

Sources: [NAS DPIA](https://www.statewatch.org/media/1893/uk-home-office-anpr-network-dpia-2-21.pdf),
[Aker Systems on NSAP](https://www.akersystems.com/insights/aker-systems-supports-uk-police-and-law-enforcement-with-next-generation-national-strategic-anpr-platform-nsap),
[NPCC ANPR Strategy 2020–2024](https://npcc.police.uk/ANPR%20Strategy%202020%20Final.pdf).

**What we take:** the four-stage separation, the read/hit distinction, tiered
retention (short at the edge, longer centrally), and the discipline of treating
a "read" as an *observation* rather than a fact.

**What we change:** Nepal's realistic national volume is single-digit millions
of reads per day, not 350 million. Designing for UK throughput would produce an
unaffordable, unoperable system. We target the same *shape* at 1/100th the
scale, on hardware a municipality can buy.

### Open-source ANPR stacks

| Project | Detection | OCR | Licence |
|---|---|---|---|
| [fast-alpr / fast-plate-ocr](https://github.com/mftnakrsu/Automatic_Number_Plate_Recognition_YOLO_OCR) | YOLOv9-t ONNX | Custom CTC | MIT |
| OpenALPR | LBP cascade | Tesseract-derived | AGPL / commercial |
| [YOLO + PaddleOCR pipelines](https://github.com/xavierkoo/computer_vision_anpr_alpr) | YOLOv4/v7 | PaddleOCR | mixed |
| Plate Recognizer | proprietary | proprietary | commercial |

None ship a Devanagari model. The detection halves are reusable; the
recognition halves are not.

### Low-resolution and blurred plates

This is the best-studied part of the problem and the literature is clear about
what works.

**[ICPR 2026 Competition on Low-Resolution License Plate Recognition](https://arxiv.org/abs/2604.22506)**
is the definitive recent benchmark:

- Dataset **LRLPR-26**: 20,000 training tracks, 3,000 test tracks — **each track
  is five low-resolution and five high-resolution images of the same plate**
- 269 teams from 41 countries registered; 99 submitted valid blind-test entries
- **Winner: 82.13%** recognition rate; four teams above 80%

Two things to read off this. First, low-resolution plate recognition is *hard* —
the world's best effort on a curated benchmark is ~82%, so any claim of 99%
accuracy on degraded Nepali plates should be treated as a measurement error.
Second, and more usefully: **the benchmark is built around tracks, not images.**
The field has converged on multi-frame evidence as the primary lever.

Single-image super-resolution is the secondary lever, and it is well developed —
[LPSRGAN](https://wrap.warwick.ac.uk/id/eprint/183557/1/WRAP-LPSRGAN-Generative-adversarial-networks-super-resolution-license-plate-image-24.pdf)
with its n-stage random combination degradation model,
[D_GAN_ESR](https://pmc.ncbi.nlm.nih.gov/articles/PMC8605205/) (denoise/deblur
then 4× upscale), AFA-Net's pixel- and feature-level deblurring, and
[attention + sub-pixel convolution approaches](https://arxiv.org/abs/2305.17313).

**What we take:** fusion first, enhancement second. `nepal_plate.fuse` implements
track-level fusion as the primary path; super-resolution sits in the edge
pipeline as a preprocessing option, not as the core answer.

### Synthetic data

[Advancing Multinational License Plate Recognition Through Synthetic and Real
Data Fusion](https://arxiv.org/abs/2601.07671) is directly applicable: for
countries with little annotated data, synthetic plates plus a realistic
degradation model, fused with a small real set, is the established path. Given
that all public Nepali plate data amounts to roughly five thousand images
(see [datasets.md](datasets.md)), this is not optional for us.

---

## 3. The gap, and what this project does about it

| Gap in existing work | What TheScanner does |
|---|---|
| Devanagari *or* embossed | One recogniser, 71-token unified vocabulary, both grammars, automatic routing by plate colour |
| Regex applied after argmax | **Grammar-constrained CTC beam search** — the decoder emits a well-formed, field-decomposed plate or an explicit refusal, never a bare string. Measured: this does *not* improve accuracy (+0.001), but 10.0% of greedy reads are not legal plates at all and cannot serve as a watch-list key. See [findings-phase2.md](findings-phase2.md) |
| Colour discarded in preprocessing | **Colour predicted at 97.9% and used as a decoding prior.** Measured: it changes 0.5% of reads and nets nothing, because the trained recogniser is already near-certain per glyph. Retained because ownership class is useful in itself and colour–class disagreement flags an altered plate |
| Character segmentation | Segmentation-free CTC over the whole plate |
| Single-frame evaluation | Track-level fusion with per-field consensus, so a plate whose zone is legible in frame 3 and serial in frame 11 still reads correctly |
| Whole-string voting | **Per-field** voting, re-validated against the grammar so consensus can never assemble an impossible plate |
| Confidence as a single float | Operational confidence bands, capped below HIGH for uncorroborated single-frame reads, with per-field support exposed |
| No fraud detection | Plate–vehicle consistency: a plate whose class says *motorcycle* on a vehicle the detector says is a truck is a cloned-plate signal. Colour–class disagreement likewise |
| Cloud-dependent | Edge-first with store-and-forward; a site keeps working through a power cut or a severed link |
| No evidentiary chain | Hash-chained append-only read log with per-node signatures, so a read can be defended in court |
| Privacy as an afterthought | Built to Nepal's Privacy Act 2075 — see [`../security-and-privacy.md`](../security-and-privacy.md) |

### Honest caveats

The first two entries in that table were the project's headline claims, and
**measurement did not support them**. Phase 2 ran the ablation on a trained
model and found the grammar constraint worth +0.001 and the colour prior worth
0.000, because a recogniser trained only on legal plates has already internalised
the grammar (3.1 × 10⁻³ of its mass sits on illegal tokens) and is already
near-certain per glyph (top-1 minus top-2 margin of 0.927). The full account,
including why the reasoning was wrong and what the mechanisms genuinely buy, is
in [findings-phase2.md](findings-phase2.md).

The rest of the differentiators stand, but should be read with that correction
in mind: several of them are *architecturally* sound without being empirically
decisive, and the difference is only visible once someone measures.

Further caveats:

- The colour prior applies **only to legacy plates** — embossed plates are
  uniformly black-on-white — so even its residual value decays as the fleet
  transitions.
- Grammar repair converts self-evident garbage into plausible-but-wrong plates,
  which in a law-enforcement context is the more dangerous failure. The
  `repaired` flags and confidence bands mitigate this, but the HIGH-band
  false-positive rate is currently 2.2% against a 0.5% target.
- No claim here has been validated on real Nepali imagery. All numbers are from
  synthetic data produced by the same generator the model trained on, and should
  be read as an upper bound. NepalPlate-Bench (Phase 1.7) is the only thing that
  can change that.

---

## Sources

- [ICPR 2026 Competition on Low-Resolution License Plate Recognition](https://arxiv.org/abs/2604.22506)
- [Advancing Multinational LPR Through Synthetic and Real Data Fusion](https://arxiv.org/abs/2601.07671)
- [Character Recognition of Nepali Number Plate](https://arxiv.org/abs/2606.28946)
- [LPSRGAN](https://www.sciencedirect.com/science/article/abs/pii/S0925231224001978)
- [Super-Resolution of License Plate Images Using Attention Modules and Sub-Pixel Convolution](https://arxiv.org/abs/2305.17313)
- [UK National ANPR Service DPIA](https://www.statewatch.org/media/1893/uk-home-office-anpr-network-dpia-2-21.pdf)
- [Kathmandu Valley steps up CCTV-based traffic enforcement](https://kathmandupost.com/valley/2026/07/08/kathmandu-valley-steps-up-cctv-based-traffic-enforcement)
- [Number plate reader cameras introduced in Valley](https://kathmandupost.com/valley/2023/07/07/number-plate-reader-cameras-introduced-in-valley)
- [Valley Traffic Police Start Using AI and ANPR Technologies](https://newbusinessage.com/news/19965/valley-traffic-police-start-using-ai-and-anprc-technologies-to-monitor-hit-and-run-cases/)
