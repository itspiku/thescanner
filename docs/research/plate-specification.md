# Nepali vehicle registration plates — reference specification

The authoritative machine-readable version of everything below lives in
[`packages/nepal_plate/nepal_plate/spec.py`](../../packages/nepal_plate/nepal_plate/spec.py).
This document records the sources and the reasoning; the code records the facts.

## The central fact: Nepal runs two plate systems at once

Nepal began rolling out embossed plates in 2020. The legacy Devanagari zonal
plates were **not** recalled — they remain legal and remain the majority of
vehicles on the road. Any ANPR intended for real deployment in Nepal must read
both systems, and must never apply one system's grammar to the other's plate.

This is the single most consequential design constraint in the project, and it
is the thing most existing Nepali ANPR work gets wrong — nearly all published
projects handle Devanagari plates *or* embossed plates, not both.

---

## System 1 — Legacy zonal plates (Devanagari)

### Layout

```
<zone> <lot?> <class> <serial>
  बा     १      च      १२३४
  Ba     1     Cha     1234
```

- **zone** — one of 14 codes, closed set
- **lot** — 0–2 Devanagari digits. Appears only once a zone's four-digit series
  is exhausted, so most plates have none.
- **class** — one of 13 codes, closed set
- **serial** — up to 4 Devanagari digits

Physically these are usually laid out on two rows (zone/lot/class on top,
serial below), especially on motorcycles. Cars are sometimes single-row.

### Zone codes

The 14 zones were abolished as administrative units in 2015 when Nepal moved to
7 provinces, but the plate codes outlived them.

| Devanagari | Roman | Zone | Now mostly in province |
|---|---|---|---|
| मे | ME | Mechi | 1 |
| को | KO | Koshi | 1 |
| स | SA | Sagarmatha | 1 |
| ज | JA | Janakpur | 2 |
| बा | BA | Bagmati | 3 |
| ना | NA | Narayani | 3 |
| ग | GA | Gandaki | 4 |
| लु | LU | Lumbini | 5 |
| ध | DHA | Dhawalagiri | 4 |
| रा | RA | Rapti | 5 |
| भे | BHE | Bheri | 6 |
| क | KA | Karnali | 6 |
| से | SE | Seti | 7 |
| म | MA | Mahakali | 7 |

Province mapping is approximate — several zones straddle the new provincial
boundaries. It is used only for coarse geographic analytics, never validation.

### Class letters — ownership × size

| | Heavy | Light | Motorcycle |
|---|---|---|---|
| **Private** | क KA | च CHA | प PA |
| **Public / commercial** | ख KHA | ज JA | फ PHA |
| **Government** | ग GA | झ JHA | ब BA |
| **National corporation** | घ GHA | ञ NYA | — |
| **Tourist** | य YA (any size) | | |
| **Diplomatic** | सी.डी. CD (any size) | | |

### Colour schemes — and why they matter

| Background | Text | Ownership |
|---|---|---|
| Red | White | Private |
| Black | White | Public / commercial |
| White | Red | Government |
| Yellow | Black | National corporation |
| Green | White | Tourist |
| Blue | White | Diplomatic |

**This is the exploitable redundancy.** Ownership is encoded *twice* on every
legacy plate — once in the colour and once in the class letter. Colour is a
low-frequency, large-area cue that survives motion blur and low resolution far
better than glyph shape does. So when the class glyph is unreadable, the colour
still tells you which third of the class alphabet it must come from.

`nepal_plate.decode.colour_slot_bonus` turns this into a per-path prior in the
CTC beam search. See [prior-art.md](prior-art.md) — no existing ANPR appears to
use plate colour this way.

### Glyph collisions

Exactly three glyphs serve as both a zone code and a class letter: **क**, **ग**,
**ज**. Position in the grammar disambiguates them — which is another reason to
decode against a positional grammar rather than a flat character classifier.

### The 34-token charset

10 Devanagari numerals + 14 zone codes + 13 class letters − 3 shared glyphs =
**34 tokens**. This independently reproduces the 34-class charset used by
published Nepali plate-OCR datasets, which is a useful check that the
derivation is right. Pinned by a test.

---

## System 2 — Embossed plates (2020–)

### Layout

```
<province> <class> <series> <serial>
    3         B       PA      1234
```

- **province** — 1–7
- **class** — one letter A–K, some with a numeric subclass
- **series** — two Latin letters
- **serial** — up to 4 digits

### Physical characteristics

- Embossed aluminium, raised characters, reflective finish
- **FE-Schrift** typeface
- Uniformly **black on white** for every category — colour no longer encodes
  ownership
- Left-hand strip carrying the Nepal flag and a blue `NEP`
- Embedded RFID chip linking the plate to a central registry

Plate dimensions (cm), front × rear:

| Vehicle | Front | Rear |
|---|---|---|
| 3-wheeler | 24 × 13 | 24 × 13 |
| Car / light | 45 × 11 | 30 × 18.5 |
| Heavy | 52 × 11 | 36 × 21 |

### Class letters

| Code | Vehicle |
|---|---|
| A | Motorcycle, Scooter, Moped |
| B | Car, Jeep, Cargo/Delivery Van |
| C | Tempo, Auto Rickshaw |
| C1 | E-Rickshaw |
| D | Power Tiller |
| E | Tractor |
| F | Minibus, Mini Truck |
| G | Truck, Bus, Lorry |
| H | Road Roller, Dozer |
| H1 / H2 | Dozer / Road Roller |
| I | Crane, Fire Brigade, Loader |
| I1 / I2 / I3 | Crane / Fire Brigade / Loader |
| J1–J5 | Excavator / Backhoe Loader / Grader / Forklift / Other heavy equipment |
| K | Scooter, Moped |

Only C, H, I and J admit a numeric subclass. This is enforced as a *guard* in
the grammar so the decoder never spends beam width on `A1` or `B3`.

### FE-Schrift is a gift

FE-Schrift (*fälschungserschwerende Schrift*, "forgery-impeding script") was
engineered so that no character can be physically altered into another. The
same property makes it unusually robust to blur: it has markedly fewer
confusable pairs than a normal typeface. Embossed plates are the **easier half**
of this problem, and the difficulty is concentrated in the legacy Devanagari
plates — which is where the grammar and colour priors are aimed.

### Special cases

- The **President's vehicle** carries the Coat of Arms of Nepal and no
  registration number. The system must not treat this as a failed read.
- The 2018 rollout was halted by a Supreme Court order over the removal of
  Devanagari script and over RFID privacy implications; the injunction was
  rescinded in December 2019. That history is a live reminder that plate
  surveillance in Nepal is politically and legally contested — see
  [`docs/security-and-privacy.md`](../security-and-privacy.md).

---

## Sources

- [Vehicle registration plates of Nepal — Wikipedia](https://en.wikipedia.org/wiki/Vehicle_registration_plates_of_Nepal)
- [Embossed Number Plate in Nepal 2026: Rules & Process](https://omodajaecoonepal.com/blog/embossed-number-plate-in-nepal)
- [Vehicle Number Plate System in Nepal: Registration, Cost & Format](https://thirdwheel.com.np/blog/1665957614)
- [Understanding Vehicle Number Plates in Nepal: Types and Categories](https://www.nepaldatabase.com/understanding-vehicle-number-plates-in-nepal-types-and-cate)
- [Vehicle Registration in Nepal: Bluebook Process & Fees](https://courtmarriageinnepal.com/blog/vehicle-registration-nepal)

> **Verification status.** These are secondary sources. Before production
> deployment the tables must be confirmed against the Department of Transport
> Management's own published rules, and against a physical sample of plates from
> each category. Tracked as an open item in [`../PLAN.md`](../PLAN.md).
