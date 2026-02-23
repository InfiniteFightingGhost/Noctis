import { inject, Injectable, signal } from "@angular/core";
import { firstValueFrom } from "rxjs";
import { toApiError } from "../../../core/api/api-errors";
import { HomeApi, HomeOverview } from "../api/home.api";

export type HomeViewState = "loading" | "success" | "error";

@Injectable({ providedIn: "root" })
export class HomeStore {
  private readonly api = inject(HomeApi);

  readonly overview = signal<HomeOverview | null>(null);
  readonly status = signal<HomeViewState>("loading");
  readonly errorMessage = signal<string | null>(null);
  readonly isFetching = signal(false);

  async loadOverview(): Promise<void> {
    this.status.set("loading");
    this.errorMessage.set(null);
    this.isFetching.set(true);

    try {
      const overview = await firstValueFrom(this.api.getOverview());
      this.overview.set(overview);
      this.status.set("success");
    } catch (error) {
      const parsed = toApiError(error);
      this.errorMessage.set(parsed.message);
      this.status.set("error");
    } finally {
      this.isFetching.set(false);
    }
  }
}
