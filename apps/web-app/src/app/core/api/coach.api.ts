import { inject, Injectable } from "@angular/core";
import { ApiClient } from "./http-client";

export type CoachInsight = {
  id: string;
  title: string;
  message: string;
  tags?: string[];
};

export type CoachSummary = {
  generated_at: string;
  is_partial: boolean;
  insights: CoachInsight[];
};

@Injectable({ providedIn: "root" })
export class CoachApi {
  private readonly api = inject(ApiClient);

  getSummary() {
    return this.api.get<CoachSummary>("/v1/coach/summary");
  }
}
