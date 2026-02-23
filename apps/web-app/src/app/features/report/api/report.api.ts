import { inject, Injectable } from "@angular/core";
import { ApiClient } from "../../../core/api/http-client";
import { SleepSummary } from "../../sleep-summary/api/sleep-summary.types";

export type ReportSummary = SleepSummary;

@Injectable({ providedIn: "root" })
export class ReportApi {
  private readonly api = inject(ApiClient);

  getLatestReport() {
    return this.api.get<ReportSummary>("/v1/report/latest");
  }
}
