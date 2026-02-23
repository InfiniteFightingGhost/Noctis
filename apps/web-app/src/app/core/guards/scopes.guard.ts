import { inject } from "@angular/core";
import { CanActivateFn, Router } from "@angular/router";
import { AuthService } from "../services/auth.service";

export const SCOPES_DATA_KEY = "scopes";

export const scopesGuard: CanActivateFn = (route) => {
  const auth = inject(AuthService);
  const router = inject(Router);
  const requiredScopes =
    (route.data?.[SCOPES_DATA_KEY] as string[] | undefined) ?? [];

  if (!auth.isAuthenticated()) {
    return router.parseUrl("/account");
  }

  if (requiredScopes.length === 0 || auth.hasScopes(requiredScopes)) {
    return true;
  }

  return router.parseUrl("/account");
};
