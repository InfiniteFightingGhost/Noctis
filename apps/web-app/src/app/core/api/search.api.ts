import { inject, Injectable } from "@angular/core";
import { ApiClient } from "./http-client";

export type SearchResult = {
  id: string;
  type:
    | "coach"
    | "alarm"
    | "routine"
    | "challenge"
    | "device"
    | "recording"
    | "user"
    | "account"
    | "unknown";
  title: string;
  subtitle?: string | null;
};

export type SearchResponse = {
  query: string;
  results: SearchResult[];
};

@Injectable({ providedIn: "root" })
export class SearchApi {
  private readonly api = inject(ApiClient);

  search(query: string) {
    return this.api.get<SearchResponse>("/v1/search", {
      params: { q: query },
    });
  }
}
