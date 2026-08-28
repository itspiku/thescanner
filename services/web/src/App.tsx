import { useMemo, useState } from "react";
import { Api, can, type Role, type Session } from "./api";
import { type Lang, t } from "./i18n";
import { Alerts, Anomalies, Audit, Dashboard, LiveFeed, Review, Zones } from "./components/panels";
import { Search } from "./components/search";

type Tab = "live" | "alerts" | "zones" | "search" | "review" | "anomalies" | "audit";

/** Which tabs a role can see. Presentation only — the server re-checks every
 *  route. Hiding a control is a courtesy, never a control. */
const TABS_FOR: Record<Role, Tab[]> = {
  operator: ["live", "alerts", "zones", "review"],
  investigator: ["live", "alerts", "zones", "search", "review", "anomalies"],
  admin: ["live", "alerts", "zones", "search", "review", "anomalies"],
  // Auditors see the access log and nothing else. An oversight role that can
  // also perform the activity it oversees is not oversight.
  auditor: ["audit"],
};

function Login({
  api,
  lang,
  onSession,
}: {
  api: Api;
  lang: Lang;
  onSession: (s: Session) => void;
}) {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      onSession(await api.login(username, password));
    } catch {
      // One message for both unknown user and wrong password. Distinguishing
      // them is a free username-enumeration oracle.
      setError(t("auth.failed", lang));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="login-shell">
      <form className="login" onSubmit={submit}>
        <h1>{t("app.title", lang)}</h1>
        <p className="subtitle">{t("app.subtitle", lang)}</p>
        <label>
          <span>{t("auth.username", lang)}</span>
          <input value={username} onChange={(e) => setUsername(e.target.value)} autoFocus />
        </label>
        <label>
          <span>{t("auth.password", lang)}</span>
          <input
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
          />
        </label>
        <button type="submit" className="primary" disabled={busy || !username || !password}>
          {busy ? t("common.loading", lang) : t("auth.signIn", lang)}
        </button>
        {error && (
          <p className="error-note" role="alert">
            {error}
          </p>
        )}
      </form>
    </div>
  );
}

export default function App() {
  const [lang, setLang] = useState<Lang>("ne");
  const [session, setSession] = useState<Session | null>(null);
  const [tab, setTab] = useState<Tab>("live");

  const api = useMemo(() => new Api(session), [session]);

  if (!session) {
    return <Login api={api} lang={lang} onSession={(s) => {
      setSession(s);
      setTab(TABS_FOR[s.role][0]);
    }} />;
  }

  const tabs = TABS_FOR[session.role];
  const props = { api, lang, role: session.role };

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <strong>{t("app.title", lang)}</strong>
          <span className="subtitle">{t("app.subtitle", lang)}</span>
        </div>
        <nav>
          {tabs.map((name) => (
            <button
              key={name}
              className={tab === name ? "active" : ""}
              onClick={() => setTab(name)}
            >
              {t(`nav.${name}` as never, lang)}
            </button>
          ))}
        </nav>
        <div className="account">
          <button
            className="lang"
            onClick={() => setLang(lang === "ne" ? "en" : "ne")}
            aria-label="Switch language"
          >
            {lang === "ne" ? "EN" : "नेपाली"}
          </button>
          <span className="who">
            {session.username} <span className="muted">({session.role})</span>
          </span>
          <button onClick={() => setSession(null)}>{t("auth.signOut", lang)}</button>
        </div>
      </header>

      {/* An account without MFA is a standing risk on a system holding national
          movement data, so it is stated in the chrome rather than buried in an
          admin page nobody opens. */}
      {!session.mfaEnrolled && (
        <p className="banner-warn" role="status">
          {t("auth.noMfa", lang)}
        </p>
      )}

      <main>
        {can(session.role, "live") && tab !== "audit" && <Dashboard {...props} />}
        {tab === "live" && <LiveFeed {...props} />}
        {tab === "alerts" && <Alerts {...props} />}
        {tab === "zones" && <Zones {...props} />}
        {tab === "search" && <Search {...props} />}
        {tab === "review" && <Review {...props} />}
        {tab === "anomalies" && <Anomalies {...props} />}
        {tab === "audit" && <Audit {...props} />}
      </main>
    </div>
  );
}
