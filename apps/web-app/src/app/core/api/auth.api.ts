import { inject, Injectable } from "@angular/core";
import { ApiClient } from "./http-client";

export type AuthCredentials = {
  email: string;
  password: string;
};

export type AuthMeResponse = {
  scopes?: unknown;
  tenant_id?: string | null;
  tenantId?: string | null;
} & Record<string, unknown>;

@Injectable({ providedIn: "root" })
export class AuthApi {
  private readonly api = inject(ApiClient);

  register(payload: AuthCredentials) {
    return this.api.post<unknown>("/v1/auth/register", payload);
  }

  login(payload: AuthCredentials) {
    return this.api.post<unknown>("/v1/auth/login", payload);
  }

  me() {
    return this.api.get<AuthMeResponse>("/v1/auth/me");
  }
}
