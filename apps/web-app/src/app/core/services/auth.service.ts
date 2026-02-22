import { inject, Injectable } from "@angular/core";
import { SessionState, SessionStore } from "../state/session.store";

const STORAGE_KEY = "noctis.auth.session";

@Injectable({ providedIn: "root" })
export class AuthService {
  private readonly store = inject(SessionStore);

  constructor() {
    this.restoreSession();
  }

  isAuthenticated(): boolean {
    return this.store.isAuthenticated();
  }

  getAccessToken(): string | null {
    return this.store.accessToken();
  }

  getScopes(): string[] {
    return this.store.scopes();
  }

  getSession(): SessionState | null {
    return this.store.session();
  }

  hasScopes(required: string[]): boolean {
    if (required.length === 0) {
      return true;
    }

    const currentScopes = new Set(this.store.scopes());
    return required.every((scope) => currentScopes.has(scope));
  }

  setSession(session: SessionState): void {
    this.store.setSession(session);
    this.persistSession(session);
  }

  setAccessToken(accessToken: string, scopes: string[] = []): void {
    this.setSession({ accessToken, scopes });
  }

  clearSession(): void {
    this.store.clearSession();
    this.clearPersistedSession();
  }

  private restoreSession(): void {
    const session = this.readPersistedSession();
    if (session) {
      this.store.setSession(session);
    }
  }

  private readPersistedSession(): SessionState | null {
    if (typeof window === "undefined") {
      return null;
    }

    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) {
      return null;
    }

    try {
      const parsed = JSON.parse(raw) as Partial<SessionState> | null;
      if (!parsed || typeof parsed.accessToken !== "string") {
        return null;
      }

      const scopes = Array.isArray(parsed.scopes)
        ? parsed.scopes.filter((scope) => typeof scope === "string")
        : [];

      const tenantId = typeof parsed.tenantId === "string" ? parsed.tenantId : null;

      return {
        accessToken: parsed.accessToken,
        scopes,
        tenantId,
      };
    } catch {
      return null;
    }
  }

  private persistSession(session: SessionState): void {
    if (typeof window === "undefined") {
      return;
    }

    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(session));
  }

  private clearPersistedSession(): void {
    if (typeof window === "undefined") {
      return;
    }

    window.localStorage.removeItem(STORAGE_KEY);
  }
}
