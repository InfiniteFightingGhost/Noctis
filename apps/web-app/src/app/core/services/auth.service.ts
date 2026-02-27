import { inject, Injectable } from "@angular/core";
import { firstValueFrom } from "rxjs";
import { AuthApi } from "../api/auth.api";
import { SessionState, SessionStore } from "../state/session.store";

@Injectable({ providedIn: "root" })
export class AuthService {
  private readonly store = inject(SessionStore);
  private readonly api = inject(AuthApi);

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
  }

  setAccessToken(accessToken: string, scopes: string[] = []): void {
    this.setSession({ accessToken, scopes });
  }

  clearSession(): void {
    this.store.clearSession();
  }

  async login(email: string, password: string): Promise<void> {
    const authResponse = await firstValueFrom(this.api.login({ email, password }));
    const accessToken = this.parseAccessToken(authResponse);
    this.store.setSession({ accessToken, scopes: [] });

    try {
      await this.refreshSessionProfile();
    } catch (error) {
      this.store.clearSession();
      throw error;
    }
  }

  async signup(email: string, password: string): Promise<void> {
    const authResponse = await firstValueFrom(this.api.register({ email, password }));
    const accessToken = this.parseAccessToken(authResponse);
    this.store.setSession({ accessToken, scopes: [] });

    try {
      await this.refreshSessionProfile();
    } catch (error) {
      this.store.clearSession();
      throw error;
    }
  }

  async refreshSessionProfile(): Promise<void> {
    const profile = await firstValueFrom(this.api.me());
    const current = this.store.session();
    if (!current) {
      return;
    }

    this.store.setSession({
      accessToken: current.accessToken,
      scopes: this.parseScopes(profile.scopes),
      tenantId: this.parseTenantId(profile),
    });
  }

  private parseAccessToken(response: unknown): string {
    if (!response || typeof response !== "object") {
      throw new Error("Login failed. Please try again.");
    }

    const payload = response as {
      access_token?: unknown;
      accessToken?: unknown;
    };
    const token =
      typeof payload.access_token === "string"
        ? payload.access_token
        : typeof payload.accessToken === "string"
          ? payload.accessToken
          : null;

    if (!token) {
      throw new Error("Login failed. Please try again.");
    }

    return token;
  }

  private parseScopes(scopes: unknown): string[] {
    if (!Array.isArray(scopes)) {
      return [];
    }

    return scopes.filter((scope): scope is string => typeof scope === "string");
  }

  private parseTenantId(profile: unknown): string | null {
    if (!profile || typeof profile !== "object") {
      return null;
    }

    const payload = profile as {
      tenant_id?: unknown;
      tenantId?: unknown;
    };
    const tenantId =
      typeof payload.tenant_id === "string"
        ? payload.tenant_id
        : typeof payload.tenantId === "string"
          ? payload.tenantId
          : null;

    return tenantId;
  }
}
