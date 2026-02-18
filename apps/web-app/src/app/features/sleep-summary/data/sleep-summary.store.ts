import { inject, Injectable, signal, computed } from "@angular/core";
import { firstValueFrom } from "rxjs";
import { toApiError } from "../../../core/api/api-errors";
import { SleepSummaryApi } from "../api/sleep-summary.api";
import { SleepSummary } from "../api/sleep-summary.types";
import {
  mapQualityToViewState,
  resolvePrimaryActionLabel,
  SleepSummaryViewState,
} from "./sleep-summary.utils";

@Injectable({ providedIn: "root" })
export class SleepSummaryStore {
  private readonly api = inject(SleepSummaryApi);
  private readonly cachedSummary = signal<SleepSummary | null>(null);

  readonly summary = signal<SleepSummary | null>(null);
  readonly status = signal<SleepSummaryViewState>("loading");
  readonly errorMessage = signal<string | null>(null);
  readonly isFetching = signal(false);

  readonly primaryActionLabel = computed(() =>
    resolvePrimaryActionLabel(
      this.summary()?.dataQuality.status,
      this.summary()?.primaryAction,
    ),
  );

  readonly canRenderSummary = computed(() => this.status() === "success");

  async loadLatest(): Promise<void> {
    this.status.set("loading");
    this.errorMessage.set(null);
    this.isFetching.set(true);

    try {
      const summary = await firstValueFrom(this.api.getLatestSummary());
      this.summary.set(summary);
      const viewState = mapQualityToViewState(summary.dataQuality.status);
      this.status.set(viewState);

      if (viewState === "success") {
        this.cachedSummary.set(summary);
      }
    } catch (error) {
      const parsed = toApiError(error);
      this.errorMessage.set(parsed.message);
      this.status.set("error");
      if (this.cachedSummary()) {
        this.summary.set(this.cachedSummary());
      }
    } finally {
      this.isFetching.set(false);
    }
  }
}
