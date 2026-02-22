import { inject, Injectable, signal } from "@angular/core";
import { firstValueFrom } from "rxjs";
import { toApiError } from "../../../core/api/api-errors";
import {
  ChallengesApi,
  ChallengesSummary,
} from "../../../core/api/challenges.api";

export type ChallengesViewState =
  | "loading"
  | "success"
  | "no-data"
  | "syncing"
  | "error";

@Injectable({ providedIn: "root" })
export class ChallengesStore {
  private readonly api = inject(ChallengesApi);

  readonly summary = signal<ChallengesSummary | null>(null);
  readonly status = signal<ChallengesViewState>("loading");
  readonly errorMessage = signal<string | null>(null);
  readonly isFetching = signal(false);

  async loadChallenges(): Promise<void> {
    this.status.set("loading");
    this.errorMessage.set(null);
    this.isFetching.set(true);

    try {
      const summary = await firstValueFrom(this.api.getChallenges());
      this.summary.set(summary);
      this.status.set(summary.challenges.length > 0 ? "success" : "no-data");
    } catch (error) {
      const parsed = toApiError(error);
      this.errorMessage.set(parsed.message);
      this.status.set("error");
    } finally {
      this.isFetching.set(false);
    }
  }
}
