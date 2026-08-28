# scanner-web

The operator console for [TheScanner](https://github.com/itspiku/thescanner).

React + TypeScript + Vite. **Nepali by default**, English on a toggle.

## Design positions

**Nepali first.** The people operating this are Nepali traffic police and
municipal staff. An English-only console makes the software harder to use for
the people it is for, so `ne` is the default and counts render in Devanagari
numerals. Plate text is never translated — a plate is a plate.

**No external requests, at all.** System fonts, no CDN, no Google Fonts, no
third-party map provider. Vehicle movement data for a whole country is a
national security asset, and a console that phones out to load a typeface leaks
who is using it and when.

**The reason-for-access field is the most prominent control on the search
screen.** It carries a legal obligation under the Privacy Act 2075, it is
permanently attributed to the person typing it, and the submit button stays
disabled until it is filled. That friction is deliberate: it is the moment
someone decides they have a reason to look at another person's movements, and it
should feel like one. Burying it as a small optional-looking input is exactly how
it degrades into boilerplate.

**Uncertainty is shown, not hidden.** Two markers appear inline on every read
rather than behind a detail view:

- **Inferred, not observed** — the plate grammar overrode what the pixels showed
  for those fields. An officer acting on a read needs to know the system
  inferred rather than saw.
- **Signature unverified** — the read did not check out against its camera's
  key. The row is tinted, and the dashboard tile for unverified reads turns red
  on any non-zero value, because that is a security event rather than a
  statistic.

**Role tabs are a courtesy, not a control.** The nav hides what a role cannot
use, but the server re-checks every route. An auditor sees only the access log,
because an oversight role that can also perform the activity it oversees is not
oversight.

## Screens

| Screen | Who | Reason required |
|---|---|---|
| Live feed | operator, investigator, admin | no — an operator cannot type a purpose per vehicle |
| Alerts | operator, investigator, admin | no |
| Zones | operator, investigator, admin | yes, to list sessions |
| Search (plate / partial / convoy) | investigator, admin | **yes** |
| Review queue | operator, investigator, admin | no |
| Anomalies | investigator, admin | no |
| Audit log | auditor only | n/a |

## Running it

```bash
npm --prefix services/web install
```

```bash
npm --prefix services/web run dev
```

Dev proxies `/api` to `http://127.0.0.1:8000`. Point that at a running
`scanner-api`, or seed a demo one:

```bash
python scripts/seed_demo.py --out demo --reads 400
```

```bash
npm --prefix services/web run build
```

Builds to `dist/` — about 165 kB of JS, 6 kB of CSS, no runtime dependencies
beyond React.

Licence: Apache-2.0.
