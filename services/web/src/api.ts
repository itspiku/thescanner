/**
 * Typed client for the platform API.
 *
 * One design rule runs through this file: **every call that touches personal
 * data takes a `reason` argument, and the type system enforces it.** The
 * platform rejects those requests without one, so making it optional here would
 * only move the failure from compile time to a 400 in front of an operator who
 * is mid-investigation. Requiring it in the signature means a developer cannot
 * add a new screen that quietly searches without a stated purpose.
 */

export type Role = "operator" | "investigator" | "admin" | "auditor";
export type Confidence = "high" | "medium" | "low" | "reject";

export interface Session {
  token: string;
  role: Role;
  username: string;
  mfaEnrolled: boolean;
}

export interface Read {
  id: number;
  captured_at: string;
  camera_id: string;
  site_id: string;
  plate: string;
  plate_display: string;
  confidence: Confidence;
  score: number;
  n_frames: number;
  ownership: string;
  /** Fields where grammar repair overrode the pixels. Shown to the operator:
   *  they must be able to see that the system inferred rather than observed. */
  repaired_fields: string[];
  verified: boolean;
  plate_image_sha256: string | null;
}

export interface Hit {
  id: number;
  matched_at: string;
  confidence: Confidence;
  actionable: boolean;
  plate: string | null;
  camera_id: string | null;
  captured_at: string | null;
  category: string | null;
}

export interface ZoneSessionRow {
  session_id: string;
  zone_id: string;
  camera_id: string;
  plate: string | null;
  entered_at: string;
  exited_at: string | null;
  dwell_seconds: number | null;
  close_reason: string | null;
  open: boolean;
}

export interface ReviewRow {
  id: number;
  read_id: number;
  reason: string;
  machine_plate: string;
  queued_at: string;
  camera_id: string | null;
  captured_at: string | null;
  plate_image_sha256: string | null;
  confidence: Confidence | null;
  repaired_fields: string[];
}

export interface AnomalyRow {
  id: number;
  kind: string;
  plate: string;
  severity: string;
  detected_at: string;
  detail: Record<string, unknown>;
  read_ids: number[];
}

export interface Stats {
  window_hours: number;
  reads: number;
  high_confidence: number;
  unverified: number;
  distinct_plates: number;
  open_sessions: number;
  pending_review: number;
  open_anomalies: number;
}

export class ApiError extends Error {
  constructor(
    readonly status: number,
    message: string,
  ) {
    super(message);
  }
}

function qs(params: Record<string, string | number | boolean | undefined>): string {
  const s = new URLSearchParams();
  for (const [k, v] of Object.entries(params)) {
    if (v !== undefined && v !== "") s.set(k, String(v));
  }
  const out = s.toString();
  return out ? `?${out}` : "";
}

export class Api {
  constructor(private session: Session | null = null) {}

  withSession(session: Session | null): Api {
    return new Api(session);
  }

  private async request<T>(path: string, init: RequestInit = {}): Promise<T> {
    const headers: Record<string, string> = {
      "Content-Type": "application/json",
      ...((init.headers as Record<string, string>) ?? {}),
    };
    if (this.session) headers.Authorization = `Bearer ${this.session.token}`;

    const res = await fetch(path, { ...init, headers });
    if (!res.ok) {
      let detail = res.statusText;
      try {
        const body = await res.json();
        detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
      } catch {
        /* keep statusText */
      }
      throw new ApiError(res.status, detail);
    }
    if (res.status === 204) return undefined as T;
    return (await res.json()) as T;
  }

  async login(username: string, password: string): Promise<Session> {
    const r = await this.request<{
      token: string;
      role: Role;
      mfa_enrolled: boolean;
    }>("/api/v1/auth/token", {
      method: "POST",
      body: JSON.stringify({ username, password }),
    });
    return { token: r.token, role: r.role, username, mfaEnrolled: r.mfa_enrolled };
  }

  // -- monitoring: no reason required ----------------------------------
  //
  // An operator watching a live feed cannot type a purpose per vehicle, and
  // demanding one would train everybody to paste boilerplate -- which destroys
  // the value of the reasons that do matter.

  liveReads(cameraId?: string, minutes = 15): Promise<Read[]> {
    return this.request(`/api/v1/reads/live${qs({ camera_id: cameraId, minutes })}`);
  }

  stats(hours = 24): Promise<Stats> {
    return this.request(`/api/v1/stats${qs({ hours })}`);
  }

  occupancy(): Promise<Record<string, number>> {
    return this.request("/api/v1/occupancy");
  }

  hits(unacknowledgedOnly = true): Promise<Hit[]> {
    return this.request(`/api/v1/hits${qs({ unacknowledged_only: unacknowledgedOnly })}`);
  }

  acknowledgeHit(id: number): Promise<unknown> {
    return this.request(`/api/v1/hits/${id}/acknowledge`, { method: "POST" });
  }

  anomalies(openOnly = true): Promise<AnomalyRow[]> {
    return this.request(`/api/v1/anomalies${qs({ open_only: openOnly })}`);
  }

  reviewQueue(limit = 50): Promise<ReviewRow[]> {
    return this.request(`/api/v1/review${qs({ limit })}`);
  }

  submitReview(id: number, body: { corrected_plate?: string; confirmed?: boolean }) {
    return this.request(`/api/v1/review/${id}`, {
      method: "POST",
      body: JSON.stringify(body),
    });
  }

  // -- investigation: reason is mandatory, and typed as such ------------

  searchPlate(plate: string, reason: string, limit = 200): Promise<Read[]> {
    return this.request(`/api/v1/reads/search${qs({ plate, reason, limit })}`);
  }

  searchPartial(fragment: string, reason: string, limit = 200): Promise<Read[]> {
    return this.request(`/api/v1/reads/partial${qs({ fragment, reason, limit })}`);
  }

  convoy(plate: string, reason: string): Promise<
    { plate: string; encounters: number; distinct_cameras: number }[]
  > {
    return this.request(`/api/v1/reads/convoy${qs({ plate, reason })}`);
  }

  zoneSessions(
    reason: string,
    opts: { zone_id?: string; plate?: string; open_only?: boolean } = {},
  ): Promise<ZoneSessionRow[]> {
    return this.request(`/api/v1/sessions${qs({ reason, ...opts })}`);
  }

  /** Image URL. The reason travels as a query parameter because the browser
   *  sets `src` directly and cannot attach a body — the platform logs it the
   *  same way either route. */
  imageUrl(sha256: string, reason: string): string {
    return `/api/v1/blobs/${sha256}${qs({ reason })}`;
  }

  auditTrail(actor?: string, limit = 200): Promise<
    {
      at: string;
      actor: string;
      role: string;
      action: string;
      target: string;
      reason: string;
      result: string;
      n_records: number;
      client_ip: string;
    }[]
  > {
    return this.request(`/api/v1/audit${qs({ actor, limit })}`);
  }
}

/** Permissions mirrored from the server, for hiding controls the caller cannot
 *  use. Presentation only — the server is the authority, and every route
 *  re-checks. A UI that hides a button is a courtesy, not a control. */
export const CAN: Record<string, Role[]> = {
  search: ["investigator", "admin"],
  watchlist: ["investigator", "admin"],
  review: ["operator", "investigator", "admin"],
  live: ["operator", "investigator", "admin"],
  audit: ["auditor"],
};

export function can(role: Role | undefined, action: keyof typeof CAN): boolean {
  return role !== undefined && CAN[action].includes(role);
}
