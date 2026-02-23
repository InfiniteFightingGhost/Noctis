import { inject, Injectable } from "@angular/core";
import { ApiClient } from "../../../core/api/http-client";
import { SleepSummary } from "./sleep-summary.types";

export type SyncStatus = {
  status: "ok" | "syncing" | "error";
  lastSyncAtLocal?: string;
};

@Injectable({ providedIn: "root" })
export class SleepSummaryApi {
  private readonly api = inject(ApiClient);

  getLatestSummary() {
    return this.api.get<SleepSummary>("/v1/sleep/latest/summary");
  }

  getSyncStatus() {
    return this.api.get<SyncStatus>("/v1/sync/status");
  }

  sendInsightFeedback(recordingId: string, feedback: "up" | "down") {
    return this.api.post<void>("/v1/insights/feedback", {
      recordingId,
      feedback,
    });
  }
}
