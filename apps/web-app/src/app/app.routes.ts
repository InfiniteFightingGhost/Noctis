import { Routes } from "@angular/router";

export const APP_ROUTES: Routes = [
  {
    path: "",
    loadChildren: () =>
      import("./features/home/home.routes").then(
        (module) => module.HOME_ROUTES,
      ),
  },
  {
    path: "account",
    loadChildren: () =>
      import("./features/account/account.routes").then(
        (module) => module.ACCOUNT_ROUTES,
      ),
  },
  {
    path: "search",
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
