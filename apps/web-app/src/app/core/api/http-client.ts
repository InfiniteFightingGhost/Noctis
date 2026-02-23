import { HttpClient, HttpContext, HttpHeaders, HttpParams } from "@angular/common/http";
import { inject, Injectable, InjectionToken } from "@angular/core";
import { Observable } from "rxjs";

export const API_BASE_URL = new InjectionToken<string>("API_BASE_URL");

export type ApiRequestParams = Record<
  string,
  string | number | boolean | ReadonlyArray<string | number | boolean>
>;

export type ApiRequestOptions = {
  headers?: HttpHeaders | Record<string, string | string[]>;
  params?: HttpParams | ApiRequestParams;
  context?: HttpContext;
  withCredentials?: boolean;
  reportProgress?: boolean;
  responseType?: "json";
  observe?: "body";
};

@Injectable({ providedIn: "root" })
export class ApiClient {
  private readonly http = inject(HttpClient);
  private readonly baseUrl = inject(API_BASE_URL, { optional: true }) ?? "";

  get<T>(path: string, options?: ApiRequestOptions): Observable<T> {
    return this.http.get<T>(this.buildUrl(path), options);
  }

  post<T>(path: string, body: unknown, options?: ApiRequestOptions): Observable<T> {
    return this.http.post<T>(this.buildUrl(path), body, options);
  }

  put<T>(path: string, body: unknown, options?: ApiRequestOptions): Observable<T> {
    return this.http.put<T>(this.buildUrl(path), body, options);
  }

  delete<T>(path: string, options?: ApiRequestOptions): Observable<T> {
    return this.http.delete<T>(this.buildUrl(path), options);
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
