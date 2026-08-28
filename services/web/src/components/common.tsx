/** Shared UI pieces. */
import { useCallback, useEffect, useRef, useState } from "react";
import type { Confidence, Read } from "../api";
import { type Lang, relativeTime, t } from "../i18n";

/** Poll an async source, with in-flight de-duplication.
 *
 * The console refreshes several panels on timers. Without the in-flight guard a
 * slow query on a loaded server stacks up requests behind itself and the
 * console makes the problem it is displaying worse.
 */
export function usePolled<T>(
  fn: () => Promise<T>,
  intervalMs: number,
  deps: unknown[] = [],
): { data: T | null; error: string | null; loading: boolean; refresh: () => void } {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const inFlight = useRef(false);
  const mounted = useRef(true);

  const run = useCallback(async () => {
    if (inFlight.current) return;
    inFlight.current = true;
    try {
      const result = await fn();
      if (mounted.current) {
        setData(result);
        setError(null);
      }
    } catch (e) {
      if (mounted.current) setError(e instanceof Error ? e.message : String(e));
    } finally {
      inFlight.current = false;
      if (mounted.current) setLoading(false);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);

  useEffect(() => {
    mounted.current = true;
    run();
    if (intervalMs <= 0) return () => { mounted.current = false; };
    const id = setInterval(run, intervalMs);
    return () => {
      mounted.current = false;
      clearInterval(id);
    };
  }, [run, intervalMs]);

  return { data, error, loading, refresh: run };
}

export function ConfidenceBadge({ value, lang }: { value: Confidence; lang: Lang }) {
  const label = t(`conf.${value}` as never, lang);
  return <span className={`badge conf-${value}`}>{label}</span>;
}

/**
 * A read row.
 *
 * Two markers are always shown rather than hidden behind a detail view:
 * `repaired` (the grammar overrode the pixels) and `unverified` (the signature
 * did not check out). An operator acting on a read needs to know both without
 * clicking, because both change what the read is worth.
 */
export function ReadRow({
  read,
  lang,
  imageUrl,
}: {
  read: Read;
  lang: Lang;
  imageUrl?: (sha: string) => string;
}) {
  return (
    <tr className={read.verified ? "" : "row-unverified"}>
      <td className="plate-cell">
        <span className="plate">{read.plate_display || read.plate}</span>
        {read.repaired_fields.length > 0 && (
          <span className="flag flag-repaired" title={t("read.repairedHelp", lang)}>
            {t("read.repaired", lang)}: {read.repaired_fields.join(", ")}
          </span>
        )}
        {!read.verified && (
          <span className="flag flag-unverified" title={t("read.unverifiedHelp", lang)}>
            {t("read.unverified", lang)}
          </span>
        )}
      </td>
      <td>{read.camera_id}</td>
      <td title={read.captured_at}>{relativeTime(read.captured_at, lang)}</td>
      <td>
        <ConfidenceBadge value={read.confidence} lang={lang} />
      </td>
      <td className="num">{read.n_frames}</td>
      <td>{read.ownership}</td>
      <td>
        {read.plate_image_sha256 && imageUrl && (
          <img
            className="plate-thumb"
            src={imageUrl(read.plate_image_sha256)}
            alt=""
            loading="lazy"
          />
        )}
      </td>
    </tr>
  );
}

/**
 * The reason-for-access field.
 *
 * Given real prominence on purpose. It is a legal obligation under the Privacy
 * Act, it is permanently attributed to the person typing it, and burying it as
 * a small optional-looking input is how it becomes boilerplate. The helper text
 * says what happens to it.
 */
export function ReasonField({
  value,
  onChange,
  lang,
}: {
  value: string;
  onChange: (v: string) => void;
  lang: Lang;
}) {
  const tooShort = value.trim().length > 0 && value.trim().length < 8;
  return (
    <label className="reason-field">
      <span className="reason-label">
        {t("search.reason", lang)} <span className="required">*</span>
      </span>
      <input
        type="text"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder="e.g. case 2026/114 — stolen vehicle trace"
        aria-describedby="reason-help"
      />
      <small id="reason-help" className={tooShort ? "warn" : ""}>
        {t("search.reasonHelp", lang)}
      </small>
    </label>
  );
}

export function Panel({
  title,
  help,
  children,
  actions,
}: {
  title: string;
  help?: string;
  children: React.ReactNode;
  actions?: React.ReactNode;
}) {
  return (
    <section className="panel">
      <header>
        <div>
          <h2>{title}</h2>
          {help && <p className="help">{help}</p>}
        </div>
        {actions}
      </header>
      {children}
    </section>
  );
}

export function Empty({ children }: { children: React.ReactNode }) {
  return <p className="empty">{children}</p>;
}

export function ErrorNote({ message, lang }: { message: string; lang: Lang }) {
  return (
    <p className="error-note" role="alert">
      {t("common.error", lang)}: {message}
    </p>
  );
}
