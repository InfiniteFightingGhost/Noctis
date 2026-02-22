import { inject, Injectable } from "@angular/core";
import { ApiClient } from "../../../core/api/http-client";

export type HomeOverview = {
  headline: string;
  lede: string;
  updated_at?: string | null;
};

@Injectable({ providedIn: "root" })
export class HomeApi {
  private readonly api = inject(ApiClient);

  getOverview() {
    return this.api.get<HomeOverview>("/v1/home/overview");
  }
}
