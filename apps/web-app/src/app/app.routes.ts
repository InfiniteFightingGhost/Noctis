import { Routes } from "@angular/router";
import { authGuard } from "./core/guards/auth.guard";
import { SCOPES_DATA_KEY, scopesGuard } from "./core/guards/scopes.guard";

export const APP_ROUTES: Routes = [
  {
    path: "",
    redirectTo: "dashboard",
    pathMatch: "full",
  },
  {
    path: "dashboard",
    canActivate: [authGuard],
    loadChildren: () =>
      import("./features/sleep-summary/sleep-summary.routes").then(
        (module) => module.SLEEP_SUMMARY_ROUTES,
      ),
  },
  {
    path: "sleep-summary",
    redirectTo: "dashboard",
    pathMatch: "full",
  },
  {
    path: "report",
    canActivate: [authGuard],
    loadChildren: () =>
      import("./features/report/report.routes").then(
        (module) => module.REPORT_ROUTES,
      ),
  },
  {
    path: "coach",
    canActivate: [authGuard],
    loadChildren: () =>
      import("./features/coach/coach.routes").then(
        (module) => module.COACH_ROUTES,
      ),
  },
  {
    path: "alarm",
    canActivate: [authGuard],
    loadChildren: () =>
      import("./features/alarm/alarm.routes").then(
        (module) => module.ALARM_ROUTES,
      ),
  },
  {
    path: "routine",
    canActivate: [authGuard],
    loadChildren: () =>
      import("./features/routine/routine.routes").then(
        (module) => module.ROUTINE_ROUTES,
      ),
  },
  {
    path: "challenges",
    canActivate: [authGuard],
    loadChildren: () =>
      import("./features/challenges/challenges.routes").then(
        (module) => module.CHALLENGES_ROUTES,
      ),
  },
  {
    path: "device",
    canActivate: [authGuard],
    loadChildren: () =>
      import("./features/device/device.routes").then(
        (module) => module.DEVICE_ROUTES,
      ),
  },
  {
    path: "home",
    canActivate: [authGuard],
    loadChildren: () =>
      import("./features/home/home.routes").then(
        (module) => module.HOME_ROUTES,
      ),
  },
  {
    path: "account",
    canActivate: [scopesGuard],
    data: {
      [SCOPES_DATA_KEY]: ["admin"],
    },
    loadChildren: () =>
      import("./features/account/account.routes").then(
        (module) => module.ACCOUNT_ROUTES,
      ),
  },
  {
    path: "search",
    canActivate: [authGuard],
    loadChildren: () =>
      import("./features/search/search.routes").then(
        (module) => module.SEARCH_ROUTES,
      ),
  },
  {
    path: "**",
    redirectTo: "",
  },
];
