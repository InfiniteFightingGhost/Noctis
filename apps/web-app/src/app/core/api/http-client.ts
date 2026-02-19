import { HttpClient } from "@angular/common/http";
import { inject, Injectable, InjectionToken } from "@angular/core";
import { Observable } from "rxjs";

export const API_BASE_URL = new InjectionToken<string>("API_BASE_URL");

@Injectable({ providedIn: "root" })
export class ApiClient {
  private readonly http = inject(HttpClient);
  private readonly baseUrl = inject(API_BASE_URL, { optional: true }) ?? "";

  get<T>(path: string, options?: object): Observable<T> {
    return this.http.get<T>(this.buildUrl(path), options);
  }

  post<T>(path: string, body: unknown, options?: object): Observable<T> {
    return this.http.post<T>(this.buildUrl(path), body, options);
  }

  private buildUrl(path: string): string {
    if (path.startsWith("http")) {
      return path;
    }

    const trimmedBase = this.baseUrl.replace(/\/$/, "");
    const trimmedPath = path.startsWith("/") ? path : `/${path}`;

    return `${trimmedBase}${trimmedPath}`;
  }
}
