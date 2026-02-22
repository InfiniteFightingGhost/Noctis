import { computed, inject, Injectable, signal } from "@angular/core";
import { firstValueFrom } from "rxjs";
import { toApiError } from "../../../core/api/api-errors";
import { CoachApi, CoachSummary } from "../../../core/api/coach.api";

export type CoachViewState =
  | "loading"
  | "success"
  | "no-data"
  | "syncing"
  | "error";

@Injectable({ providedIn: "root" })
export class CoachStore {
  private readonly api = inject(CoachApi);

  readonly summary = signal<CoachSummary | null>(null);
  readonly status = signal<CoachViewState>("loading");
  readonly errorMessage = signal<string | null>(null);
  readonly isFetching = signal(false);

  readonly isPartial = computed(() => this.summary()?.is_partial ?? false);

  async loadSummary(): Promise<void> {
    this.status.set("loading");
    this.errorMessage.set(null);
    this.isFetching.set(true);

    try {
      const summary = await firstValueFrom(this.api.getSummary());
      this.summary.set(summary);
      this.status.set(summary.insights.length > 0 ? "success" : "no-data");
    } catch (error) {
      const parsed = toApiError(error);
      this.errorMessage.set(parsed.message);
      this.status.set("error");
    } finally {
      this.isFetching.set(false);
    }
  }
}
