import { computed, inject, Injectable, signal } from "@angular/core";
import { firstValueFrom } from "rxjs";
import { toApiError } from "../../../core/api/api-errors";
import { DataQualityStatus } from "../../sleep-summary/api/sleep-summary.types";
import { ReportApi, ReportSummary } from "../api/report.api";

export type ReportViewState =
  | "loading"
  | "success"
  | "no-data"
  | "syncing"
  | "error";

const mapDataQualityToViewState = (status: DataQualityStatus): ReportViewState => {
  if (status === "missing") {
    return "no-data";
  }
  if (status === "syncing") {
    return "syncing";
  }
  if (status === "error") {
    return "error";
  }
  return "success";
};

@Injectable({ providedIn: "root" })
export class ReportStore {
  private readonly api = inject(ReportApi);
  private readonly cachedReport = signal<ReportSummary | null>(null);

  readonly report = signal<ReportSummary | null>(null);
  readonly status = signal<ReportViewState>("loading");
  readonly errorMessage = signal<string | null>(null);
  readonly isFetching = signal(false);

  readonly isPartial = computed(
    () => this.report()?.dataQuality.status === "partial",
  );

  async loadLatest(): Promise<void> {
    this.status.set("loading");
    this.errorMessage.set(null);
    this.isFetching.set(true);

    try {
      const report = await firstValueFrom(this.api.getLatestReport());
      this.report.set(report);
      const viewState = mapDataQualityToViewState(report.dataQuality.status);
      this.status.set(viewState);

      if (viewState === "success") {
        this.cachedReport.set(report);
      }
    } catch (error) {
      const parsed = toApiError(error);
      this.errorMessage.set(parsed.message);
      this.status.set("error");
      if (this.cachedReport()) {
        this.report.set(this.cachedReport());
      }
    } finally {
      this.isFetching.set(false);
    }
  }
}
