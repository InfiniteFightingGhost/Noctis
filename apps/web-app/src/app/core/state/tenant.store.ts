import { inject, Injectable, signal } from "@angular/core";
import { firstValueFrom } from "rxjs";
import { toApiError } from "../api/api-errors";
import { AuthService } from "../services/auth.service";
import { Tenant, TenantService } from "../services/tenant.service";

export type TenantStatus =
  | "idle"
  | "loading"
  | "success"
  | "error"
  | "unauthenticated";

@Injectable({ providedIn: "root" })
export class TenantStore {
  private readonly service = inject(TenantService);
  private readonly auth = inject(AuthService);

  readonly tenant = signal<Tenant | null>(null);
  readonly status = signal<TenantStatus>("idle");
  readonly errorMessage = signal<string | null>(null);
  readonly isFetching = signal(false);

  async loadCurrent(): Promise<void> {
    if (!this.auth.isAuthenticated()) {
      this.tenant.set(null);
      this.status.set("unauthenticated");
      this.errorMessage.set(null);
      return;
    }

    this.status.set("loading");
    this.errorMessage.set(null);
    this.isFetching.set(true);

    try {
      const tenant = await firstValueFrom(this.service.getCurrentTenant());
      this.tenant.set(tenant);
      this.status.set("success");
    } catch (error) {
      const parsed = toApiError(error);
      this.errorMessage.set(parsed.message);
      this.status.set("error");
    } finally {
      this.isFetching.set(false);
    }
  }

  clear(): void {
    this.tenant.set(null);
    this.status.set("idle");
    this.errorMessage.set(null);
  }
}
