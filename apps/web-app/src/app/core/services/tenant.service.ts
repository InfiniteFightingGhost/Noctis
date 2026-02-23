import { inject, Injectable } from "@angular/core";
import { Observable } from "rxjs";
import { ApiClient } from "../api/http-client";

export type Tenant = {
  id: string;
  name: string;
  slug?: string | null;
} & Record<string, unknown>;

@Injectable({ providedIn: "root" })
export class TenantService {
  private readonly api = inject(ApiClient);

  getCurrentTenant(): Observable<Tenant> {
    return this.api.get<Tenant>("/v1/tenants/me");
  }
}
