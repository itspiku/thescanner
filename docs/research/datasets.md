# Dataset survey and data strategy

**Bottom line: there is not enough public Nepali plate data to train a
production system, and no amount of searching will change that.** Everything
public amounts to roughly five thousand images, heavily skewed toward
motorcycles in one zone, in daylight, almost entirely legacy plates. So the
data strategy has to be *generation plus targeted collection*, not acquisition.

---

## 1. What exists

### Nepali plate datasets

| Dataset | Size | Content | Access | Use here |
|---|---|---|---|---|
| [Prasanna1991/LPR](https://github.com/Prasanna1991/LPR) | 2,033 images | Cropped **characters**, 12 classes (०–९, बा, प), private motorbikes, Bagmati | GitHub, open | Character-level pretraining; too few classes for the full 34 |
| [Roboflow — Nepali License Plate Images](https://universe.roboflow.com/project-uv02l/nepali-license-plate-images-pegbh) | 565 images | Plate bounding boxes | Roboflow Universe | Detector training |
| [Roboflow — Nepali License Plate Images 2](https://universe.roboflow.com/project-uv02l/nepali-license-plate-images-2) | 66 images | Plate bounding boxes | Roboflow Universe | Detector training |
| [Roboflow — nepali ocr plate detection](https://universe.roboflow.com/nepali-ocr-plate-detection) | 352 images | Detection | Roboflow Universe | Detector training |
| [Roboflow — Nepali Number Plate Detection](https://universe.roboflow.com/number-plate-detection-nepal/nepali-number-plate-detection) | — | Detection + pretrained model | Roboflow Universe | Baseline comparison |
| [Kaggle — ALPR V2 (ishworsubedii)](https://www.kaggle.com/datasets/ishworsubedii/alpr-v2) | 2,000+ | Plate images | Kaggle | Recogniser training |
| [Kaggle — Nepali Motorbike Backplate Labeled](https://www.kaggle.com/datasets/saugat111/nepali-moterbike-backplate-lbled) | 1,500+ | Motorcycle plates with boxes | Kaggle | Detector + recogniser |
| [Kaggle — Nepali Vehicles Number Plate Dataset](https://www.kaggle.com/datasets/inspiring-lab/nepali-vehicles-number-plate-dataset) | — | Plates | Kaggle | TBC |
| [Kaggle — Vehicle Number Plate Dataset (Nepal)](https://www.kaggle.com/datasets/ishworsubedii/vehicle-number-plate-datasetnepal) | — | Plates | Kaggle | TBC |
| [Kaggle — Vehicles Dataset Nepal](https://www.kaggle.com/datasets/ishworsubedii/vehicles-dataset-nepal) | — | Whole vehicles | Kaggle | Vehicle attribute model |
| [GTS — Nepali Number Plate OCR, 34 characters](https://gts.ai/dataset-download/nepali-number-plate-characters-dataset/) | — | Character crops | **Commercial** | Not used — licensing |

### Adjacent Devanagari resources

| Dataset | Size | Use |
|---|---|---|
| [DHCD — Devanagari Handwritten Character Dataset](https://github.com/Prasanna1991/DHCD_Dataset) | 92,000 images, 46 classes (36 letters + 10 digits) | Weak pretraining signal only. Handwritten glyphs differ substantially from plate typography — useful for early feature learning, actively harmful if trained on to convergence |
| [pemagrg1/Nepali-Datasets](https://github.com/pemagrg1/Nepali-Datasets) | index | Aggregator |
| [IOST-ASCOL/nepali-datasets](https://github.com/IOST-ASCOL/nepali-datasets) | index | Aggregator |

> **Verification status.** These were located by search and their existence and
> approximate sizes confirmed from listings. Contents, licences and annotation
> quality have **not** yet been verified by download. Doing so is the first task
> of Phase 1 — see [`../PLAN.md`](../PLAN.md). Licence terms in particular must
> be checked before anything enters a government-deployed model.

---

## 2. The gaps

Aggregate the whole public corpus and you get roughly **4,500–5,000 annotated
plate images**. Against that, note what a production Nepali ANPR must handle:

| Dimension | Public data | Needed |
|---|---|---|
| Plate system | Almost entirely legacy Devanagari | Both, roughly balanced |
| Vehicle type | Dominated by motorcycles | Motorcycles, cars, tempos, minibuses, trucks, tractors, heavy equipment |
| Geography | Overwhelmingly Bagmati | All 14 zone codes, all 7 provinces |
| Ownership class | Mostly private (red) | All six colour schemes — government, tourist, diplomatic and corporation plates are rare in the wild and essentially absent from public data |
| Lighting | Daylight | Night, IR illumination, headlight glare, sodium/LED street lighting |
| Weather | Clear | Monsoon rain, fog, dust — Kathmandu's dry-season dust is a first-order image degradation |
| Condition | Clean | Mud-obscured, bent, faded, partially occluded, non-standard fonts (Nepal has widespread informal plate painting) |
| Motion | Stills | Video tracks — required for the fusion path |
| Angle | Mostly frontal | ±45° yaw, significant pitch from overhead poles |

Two gaps matter most:

1. **No video tracks at all.** The fusion architecture needs them, and the
   evaluation needs them. This cannot be synthesised convincingly — it must be
   collected.
2. **Rare classes are missing.** Government, diplomatic, tourist and corporation
   plates are exactly the ones where a misread has the highest consequence, and
   they have essentially no public examples. Synthetic generation is the only
   practical route to balanced coverage.

---

## 3. Strategy

### 3.1 Synthetic generation — the primary source

Render plates from the specification rather than collecting them. Because
`nepal_plate.spec` is a complete, machine-readable model of both plate systems,
we can generate a perfectly balanced, exhaustively labelled corpus.

**Target: 500,000 plate crops**, uniformly covering
zone × class × ownership × colour × serial, for both systems.

Pipeline:

1. **Layout** — render the plate from the grammar: correct fonts (FE-Schrift for
   embossed, a Devanagari plate face for legacy), correct dimensions per vehicle
   class, correct colour scheme, flag strip and `NEP` for embossed.
2. **Physical** — emboss/deboss relief, specular response, reflective sheeting,
   plate mounting geometry, dirt and rust masks, bent-plate warping.
3. **Optical** — perspective warp over a realistic pose distribution, defocus,
   directional motion blur matched to plausible vehicle speeds and shutter
   times, rolling-shutter shear, atmospheric haze, dust.
4. **Sensor** — Bayer/demosaic artefacts, sensor noise at realistic ISO, IR-cut
   behaviour for night, headlight and retroreflective blowout, JPEG at the
   bitrates real RTSP streams actually use.
5. **Track synthesis** — generate *sequences*, not stills: the same plate across
   N frames with coherently evolving pose, scale and blur, so the fusion path
   has training and validation data.

The degradation model follows LPSRGAN's n-stage random combination approach —
randomly composed chains of degradations rather than a fixed pipeline, which is
what makes synthetic data generalise instead of teaching the model one
artificial artefact.

Implementation: `packages/synthplate/`.

### 3.2 Real data — targeted collection

Synthetic data cannot supply the real-world prior. Real data is spent where it
counts most:

- **Fine-tuning set** — aggregate every public dataset above (licences
  permitting), re-annotated to a single schema.
- **NepalPlate-Bench** — a held-out evaluation set, collected deliberately and
  **never** trained on. Stratified across zone, class, ownership colour, vehicle
  type, time of day and weather. Target 3,000–5,000 plates including at least
  500 video tracks. This is the project's most valuable asset: without a
  trustworthy benchmark there is no way to know whether anything works.
- **Active learning loop** — once deployed, every low-confidence read is queued
  for human review. Corrections flow back into training. This turns operation
  into data collection and is how the system escapes the cold-start problem
  permanently.

### 3.3 Sequencing

```
synthetic pretrain (500k)  →  real fine-tune (~5k)  →  active-learning refresh
        balanced,                domain prior            distribution tracking
     exhaustive coverage
```

Evaluation is **always** on NepalPlate-Bench, never on synthetic data. A model
that scores well on its own generator has learned the generator.

---

## 4. Ethical and legal constraints on collection

Collecting road imagery in Nepal engages the Privacy Act 2075. Before any real
collection:

- Vehicles are not people, but drivers, passengers and pedestrians are captured
  incidentally. Faces must be blurred at the point of capture in any dataset
  that leaves the collection device.
- Collection on public roads for a government-authorised purpose needs that
  authorisation documented before, not after.
- Any published dataset must be plate-only crops with surrounding context
  removed, and must not be linkable to registration records.

Details in [`../security-and-privacy.md`](../security-and-privacy.md).
