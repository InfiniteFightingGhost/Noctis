import { Injectable, computed, signal } from "@angular/core";

export type SessionState = {
  accessToken: string;
  scopes: string[];
  tenantId?: string | null;
};

@Injectable({ providedIn: "root" })
export class SessionStore {
  readonly session = signal<SessionState | null>(null);
  readonly accessToken = computed(() => this.session()?.accessToken ?? null);
  readonly scopes = computed(() => this.session()?.scopes ?? []);
  readonly isAuthenticated = computed(() => Boolean(this.accessToken()));

  setSession(session: SessionState): void {
    this.session.set(session);
  }

  clearSession(): void {
    this.session.set(null);
  }
}
