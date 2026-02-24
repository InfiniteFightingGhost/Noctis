import { computed, inject, Injectable, signal } from "@angular/core";
import { firstValueFrom } from "rxjs";
import { toApiError } from "../../../core/api/api-errors";
import { SearchApi, SearchResponse } from "../../../core/api/search.api";

export type SearchViewState = "idle" | "loading" | "success" | "no-data" | "error";

@Injectable({ providedIn: "root" })
export class SearchStore {
  private readonly api = inject(SearchApi);

  readonly query = signal("");
  readonly response = signal<SearchResponse | null>(null);
  readonly status = signal<SearchViewState>("idle");
  readonly errorMessage = signal<string | null>(null);
  readonly isFetching = signal(false);

  readonly results = computed(() => this.response()?.results ?? []);

  updateQuery(query: string): void {
    this.query.set(query);
    if (!query.trim()) {
      this.status.set("idle");
    }
  }

  async runSearch(query: string): Promise<void> {
    const trimmed = query.trim();
    this.query.set(trimmed);

    if (!trimmed) {
      this.response.set({ query: "", results: [] });
      this.status.set("no-data");
      this.errorMessage.set(null);
      return;
    }

    this.status.set("loading");
    this.errorMessage.set(null);
    this.isFetching.set(true);

    try {
      const response = await firstValueFrom(this.api.search(trimmed));
      this.response.set(response);
      this.status.set(response.results.length > 0 ? "success" : "no-data");
    } catch (error) {
      const parsed = toApiError(error);
      this.errorMessage.set(parsed.message);
      this.status.set("error");
    } finally {
      this.isFetching.set(false);
    }
  }
}
