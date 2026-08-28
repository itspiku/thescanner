/** Investigation search.
 *
 * Every query here goes into the permanent access log against the operator's
 * name, so the reason field is the most prominent control on the screen and the
 * submit button stays disabled until it is filled. That is a deliberate piece
 * of friction: it is the moment a person decides they have a reason to look at
 * someone's movements, and it should feel like one.
 */
import { useState } from "react";
import type { Api, Read, Role } from "../api";
import { can } from "../api";
import { type Lang, t } from "../i18n";
import { Empty, ErrorNote, Panel, ReadRow, ReasonField } from "./common";

type Mode = "plate" | "partial" | "convoy";

export function Search({ api, lang, role }: { api: Api; lang: Lang; role: Role }) {
  const [mode, setMode] = useState<Mode>("plate");
  const [query, setQuery] = useState("");
  const [reason, setReason] = useState("");
  const [reads, setReads] = useState<Read[] | null>(null);
  const [convoy, setConvoy] = useState<
    { plate: string; encounters: number; distinct_cameras: number }[] | null
  >(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  if (!can(role, "search")) {
    return (
      <Panel title={t("nav.search", lang)}>
        <Empty>{t("common.denied", lang)}</Empty>
      </Panel>
    );
  }

  const ready = query.trim().length > 0 && reason.trim().length >= 8;

  async function run(e: React.FormEvent) {
    e.preventDefault();
    if (!ready) return;
    setBusy(true);
    setError(null);
    setReads(null);
    setConvoy(null);
    try {
      if (mode === "plate") setReads(await api.searchPlate(query.trim(), reason.trim()));
      else if (mode === "partial") setReads(await api.searchPartial(query.trim(), reason.trim()));
      else setConvoy(await api.convoy(query.trim(), reason.trim()));
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }

  return (
    <Panel title={t("nav.search", lang)}>
      <form className="search-form" onSubmit={run}>
        <div className="mode-tabs" role="tablist">
          {(["plate", "partial", "convoy"] as Mode[]).map((m) => (
            <button
              key={m}
              type="button"
              role="tab"
              aria-selected={mode === m}
              className={mode === m ? "active" : ""}
              onClick={() => setMode(m)}
            >
              {m === "plate"
                ? t("search.plate", lang)
                : m === "partial"
                  ? t("search.partial", lang)
                  : t("search.convoy", lang)}
            </button>
          ))}
        </div>

        <label className="query-field">
          <span>
            {mode === "partial" ? t("search.partial", lang) : t("search.plate", lang)}
          </span>
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder={mode === "partial" ? "1234" : "बा १ च १२३४  /  BA 1 CHA 1234"}
            autoComplete="off"
          />
          {/* Both scripts are accepted: the platform canonicalises before
              matching, so an officer typing romanised finds reads a camera
              emitted in Devanagari. */}
        </label>

        <ReasonField value={reason} onChange={setReason} lang={lang} />

        <button type="submit" className="primary" disabled={!ready || busy}>
          {busy ? t("common.loading", lang) : t("search.go", lang)}
        </button>
      </form>

      {error && <ErrorNote message={error} lang={lang} />}

      {reads && (
        <>
          <p className="result-count">
            {reads.length} {t("search.results", lang)}
          </p>
          {reads.length > 0 && (
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
                {reads.map((r) => (
                  <ReadRow
                    key={r.id}
                    read={r}
                    lang={lang}
                    imageUrl={(sha) => api.imageUrl(sha, reason.trim())}
                  />
                ))}
              </tbody>
            </table>
          )}
        </>
      )}

      {convoy && (
        <>
          <p className="result-count">
            {convoy.length} {t("search.results", lang)}
          </p>
          <table>
            <thead>
              <tr>
                <th>{t("read.plate", lang)}</th>
                <th>Encounters</th>
                <th>Distinct cameras</th>
              </tr>
            </thead>
            <tbody>
              {convoy.map((c) => (
                <tr key={c.plate}>
                  <td className="plate">{c.plate}</td>
                  <td className="num">{c.encounters}</td>
                  <td className="num">{c.distinct_cameras}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </>
      )}
    </Panel>
  );
}
