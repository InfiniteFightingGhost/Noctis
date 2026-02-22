import { EnvironmentProviders } from "@angular/core";
import { provideHttpClient, withInterceptors } from "@angular/common/http";
import { provideRouter } from "@angular/router";
import { APP_ROUTES } from "../app.routes";
import { API_BASE_URL } from "./api/http-client";
import { authInterceptor } from "./interceptors/auth.interceptor";
import { environment } from "../../environments/environment";

export const APP_PROVIDERS: EnvironmentProviders[] = [
  provideRouter(APP_ROUTES),
  provideHttpClient(withInterceptors([authInterceptor])),
  {
    provide: API_BASE_URL,
    useValue: environment.apiBaseUrl,
  },
];
