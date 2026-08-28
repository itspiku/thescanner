/** The operator-facing panels. */
import { useState } from "react";
import type { Api, Read, Role } from "../api";
import { can } from "../api";
import { type Lang, num, relativeTime, t } from "../i18n";
import { Empty, ErrorNote, Panel, ReadRow, ReasonField, usePolled } from "./common";

interface PanelProps {
  api: Api;
  lang: Lang;
  role: Role;
}

// ---------------------------------------------------------------------------
// Dashboard
// ---------------------------------------------------------------------------

export function Dashboard({ api, lang }: PanelProps) {
  const { data, error } = usePolled(() => api.stats(24), 15_000, [api]);
  if (error) return <ErrorNote message={error} lang={lang} />;
  if (!data) return <Empty>{t("common.loading", lang)}</Empty>;

  const tiles: [string, number, boolean][] = [
    [t("stats.reads", lang), data.reads, false],
    [t("stats.plates", lang), data.distinct_plates, false],
    [t("stats.open", lang), data.open_sessions, false],
    [t("stats.review", lang), data.pending_review, false],
    [t("stats.anomalies", lang), data.open_anomalies, data.open_anomalies > 0],
    // Unverified reads are a security signal, not a statistic: any non-zero
    // value means a node produced something that did not check out against its
    // own key, and that should never be quietly absorbed into a dashboard.
    [t("stats.unverified", lang), data.unverified, data.unverified > 0],
  ];

  return (
    <div className="tiles">
      {tiles.map(([label, value, alarm]) => (
        <div className={`tile${alarm ? " tile-alarm" : ""}`} key={label}>
          <span className="tile-value">{num(value, lang)}</span>
          <span className="tile-label">{label}</span>
        </div>
      ))}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Live feed
// ---------------------------------------------------------------------------

export function LiveFeed({ api, lang }: PanelProps) {
  const [camera, setCamera] = useState("");
  const { data, error, loading } = usePolled<Read[]>(
    () => api.liveReads(camera || undefined, 15),
    5_000,
    [api, camera],
  );

  return (
    <Panel
      title={t("nav.live", lang)}
      actions={
        <input
          className="filter"
          placeholder={t("read.camera", lang)}
          value={camera}
          onChange={(e) => setCamera(e.target.value)}
        />
      }
    >
      {error && <ErrorNote message={error} lang={lang} />}
      {loading && !data && <Empty>{t("common.loading", lang)}</Empty>}
      {data && data.length === 0 && <Empty>{t("read.none", lang)}</Empty>}
      {data && data.length > 0 && (
        <table>
          <thead>
            <tr>
              <th>{t("read.plate", lang)}</th>
              <th>{t("read.camera", lang)}</th>
              <th>{t("read.time", lang)}</th>
              <th>{t("read.confidence", lang)}</th>
              <th>{t("read.frames", lang)}</th>
              <th>{t("read.owner", lang)}</th>
              <th />
            </tr>
          </thead>
          <tbody>
            {data.map((r) => (
              <ReadRow key={r.id} read={r} lang={lang} />
            ))}
          </tbody>
        </table>
      )}
    </Panel>
  );
}

// ---------------------------------------------------------------------------
// Alerts
// ---------------------------------------------------------------------------

export function Alerts({ api, lang }: PanelProps) {
  const { data, error, refresh } = usePolled(() => api.hits(true), 8_000, [api]);

  async function acknowledge(id: number) {
    await api.acknowledgeHit(id);
    refresh();
  }

  return (
    <Panel title={t("nav.alerts", lang)}>
      {error && <ErrorNote message={error} lang={lang} />}
      {data && data.length === 0 && <Empty>{t("alert.empty", lang)}</Empty>}
      <ul className="alerts">
        {data?.map((h) => (
          <li key={h.id} className={h.actionable ? "alert actionable" : "alert"}>
            <div className="alert-main">
              <span className="plate">{h.plate}</span>
              <span className="alert-meta">
                {h.camera_id} · {h.captured_at ? relativeTime(h.captured_at, lang) : ""} ·{" "}
                {h.category}
              </span>
            </div>
            <span className={h.actionable ? "badge conf-high" : "badge conf-medium"}>
              {h.actionable ? t("alert.actionable", lang) : t("alert.reviewOnly", lang)}
            </span>
            <button onClick={() => acknowledge(h.id)}>{t("alert.acknowledge", lang)}</button>
          </li>
        ))}
      </ul>
    </Panel>
  );
}

// ---------------------------------------------------------------------------
// Zones
// ---------------------------------------------------------------------------

export function Zones({ api, lang }: PanelProps) {
  const [reason, setReason] = useState("");
  const [submitted, setSubmitted] = useState("");
  const occ = usePolled(() => api.occupancy(), 10_000, [api]);
  const sessions = usePolled(
    () => (submitted ? api.zoneSessions(submitted, { open_only: false }) : Promise.resolve([])),
    0,
    [api, submitted],
  );

  function closeLabel(row: { close_reason: string | null; open: boolean }): string {
    if (row.open) return t("zone.stillInside", lang);
    if (row.close_reason === "track_lost") return t("zone.trackLost", lang);
    if (row.close_reason === "timed_out") return t("zone.timedOut", lang);
    return t("zone.exited", lang);
  }

  return (
    <>
      <Panel title={t("zone.occupancy", lang)}>
        {occ.error && <ErrorNote message={occ.error} lang={lang} />}
        <div className="tiles">
          {Object.entries(occ.data ?? {}).map(([zone, n]) => (
            <div className="tile" key={zone}>
              <span className="tile-value">{num(n, lang)}</span>
              <span className="tile-label">{zone}</span>
            </div>
          ))}
        </div>
      </Panel>

      <Panel title={t("nav.zones", lang)}>
        <form
          className="search-form"
          onSubmit={(e) => {
            e.preventDefault();
            setSubmitted(reason);
          }}
        >
          <ReasonField value={reason} onChange={setReason} lang={lang} />
          <button type="submit" disabled={reason.trim().length < 8}>
            {t("search.go", lang)}
          </button>
        </form>

        {sessions.error && <ErrorNote message={sessions.error} lang={lang} />}
        {sessions.data && sessions.data.length > 0 && (
          <table>
            <thead>
              <tr>
                <th>{t("read.plate", lang)}</th>
                <th>{t("nav.zones", lang)}</th>
                <th>{t("zone.entered", lang)}</th>
                <th>{t("zone.dwell", lang)}</th>
                <th />
              </tr>
            </thead>
            <tbody>
              {sessions.data.map((s) => (
                <tr key={s.session_id} className={s.open ? "row-open" : ""}>
                  <td className="plate">{s.plate ?? "—"}</td>
                  <td>{s.zone_id}</td>
                  <td title={s.entered_at}>{relativeTime(s.entered_at, lang)}</td>
                  <td className="num">
                    {s.dwell_seconds != null ? `${Math.round(s.dwell_seconds)}s` : "—"}
                  </td>
                  <td>
                    <span className={`badge ${s.open ? "conf-medium" : "conf-high"}`}>
                      {closeLabel(s)}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </Panel>
    </>
  );
}

// ---------------------------------------------------------------------------
// Review queue -- the active-learning loop
// ---------------------------------------------------------------------------

export function Review({ api, lang, role }: PanelProps) {
  const { data, error, refresh } = usePolled(() => api.reviewQueue(50), 20_000, [api]);
  const [edits, setEdits] = useState<Record<number, string>>({});

  if (!can(role, "review")) return <Empty>{t("common.denied", lang)}</Empty>;

  async function submit(id: number, corrected?: string) {
    await api.submitReview(id, corrected ? { corrected_plate: corrected } : { confirmed: true });
    refresh();
  }

  return (
    <Panel title={t("review.title", lang)} help={t("review.help", lang)}>
      {error && <ErrorNote message={error} lang={lang} />}
      {data && data.length === 0 && <Empty>{t("review.empty", lang)}</Empty>}
      <ul className="review-list">
        {data?.map((item) => (
          <li key={item.id} className="review-item">
            {item.plate_image_sha256 && (
              <img
                className="review-crop"
                src={api.imageUrl(item.plate_image_sha256, `review of read ${item.read_id}`)}
                alt=""
              />
            )}
            <div className="review-body">
              <div className="review-meta">
                {t("review.machineRead", lang)}:{" "}
                <span className="plate">{item.machine_plate}</span>
                {item.repaired_fields.length > 0 && (
                  <span className="flag flag-repaired">
                    {t("read.repaired", lang)}: {item.repaired_fields.join(", ")}
                  </span>
                )}
              </div>
              <div className="review-actions">
                <button className="primary" onClick={() => submit(item.id)}>
                  {t("review.confirm", lang)}
                </button>
                <input
                  className="plate-input"
                  placeholder="बा १ च १२३४"
                  value={edits[item.id] ?? ""}
                  onChange={(e) => setEdits({ ...edits, [item.id]: e.target.value })}
                />
                <button
                  disabled={!edits[item.id]?.trim()}
                  onClick={() => submit(item.id, edits[item.id])}
                >
                  {t("review.correct", lang)}
                </button>
              </div>
            </div>
          </li>
        ))}
      </ul>
    </Panel>
  );
}

// ---------------------------------------------------------------------------
// Anomalies
// ---------------------------------------------------------------------------

const ANOMALY_LABEL: Record<string, string> = {
  colour_class_mismatch: "anom.colourClass",
  plate_vehicle_mismatch: "anom.plateVehicle",
  impossible_movement: "anom.impossible",
  registry_mismatch: "anom.registry",
};

export function Anomalies({ api, lang }: PanelProps) {
  const { data, error } = usePolled(() => api.anomalies(true), 30_000, [api]);
  return (
    <Panel title={t("nav.anomalies", lang)} help={t("anom.caution", lang)}>
      {error && <ErrorNote message={error} lang={lang} />}
      {data && data.length === 0 && <Empty>—</Empty>}
      <ul className="anomalies">
        {data?.map((a) => (
          <li key={a.id} className={`anomaly sev-${a.severity}`}>
            <div>
              <span className="plate">{a.plate}</span>
              <span className="anomaly-kind">
                {t((ANOMALY_LABEL[a.kind] ?? "nav.anomalies") as never, lang)}
              </span>
            </div>
            <pre className="anomaly-detail">{JSON.stringify(a.detail, null, 2)}</pre>
            <span className="anomaly-time">{relativeTime(a.detected_at, lang)}</span>
          </li>
        ))}
      </ul>
    </Panel>
  );
}

// ---------------------------------------------------------------------------
// Audit -- auditors only
// ---------------------------------------------------------------------------

export function Audit({ api, lang, role }: PanelProps) {
  const { data, error } = usePolled(() => api.auditTrail(undefined, 200), 30_000, [api]);
  if (!can(role, "audit")) return <Empty>{t("common.denied", lang)}</Empty>;
  return (
    <Panel title={t("nav.audit", lang)}>
      {error && <ErrorNote message={error} lang={lang} />}
      <table>
        <thead>
          <tr>
            <th>{t("read.time", lang)}</th>
            <th>Actor</th>
            <th>Action</th>
            <th>{t("search.reason", lang)}</th>
            <th>Records</th>
          </tr>
        </thead>
        <tbody>
          {data?.map((row, i) => (
            <tr key={i} className={row.result === "ok" ? "" : "row-unverified"}>
              <td title={row.at}>{relativeTime(row.at, lang)}</td>
              <td>
                {row.actor} <span className="muted">({row.role})</span>
              </td>
              <td>{row.action}</td>
              <td className="reason-cell">{row.reason}</td>
              <td className="num">{row.n_records}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </Panel>
  );
}
