import { Provider } from "@angular/core";
import { provideHttpClient } from "@angular/common/http";
import { provideRouter } from "@angular/router";
import { APP_ROUTES } from "../app.routes";

export const APP_PROVIDERS: Provider[] = [
  provideRouter(APP_ROUTES),
  provideHttpClient(),
];
