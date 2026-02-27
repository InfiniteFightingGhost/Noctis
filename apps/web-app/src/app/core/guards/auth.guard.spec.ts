import { createEnvironmentInjector, EnvironmentInjector, Provider, runInInjectionContext } from "@angular/core";
import { Router } from "@angular/router";

vi.mock("../services/auth.service", () => ({
  AuthService: class AuthService {},
}));

const runGuard = async (providers: Provider[]) => {
  const [{ authGuard }] = await Promise.all([import("./auth.guard")]);
  const parent = createEnvironmentInjector([]);
  const injector: EnvironmentInjector = createEnvironmentInjector(providers, parent);

  return runInInjectionContext(injector, () =>
    authGuard({} as any, { url: "/report" } as any),
  );
};

describe("authGuard", () => {
  it("allows authenticated users", async () => {
    const [{ AuthService }] = await Promise.all([import("../services/auth.service")]);
    const router = { createUrlTree: vi.fn() };

    const result = await runGuard([
      { provide: AuthService, useValue: { isAuthenticated: () => true } },
      { provide: Router, useValue: router },
    ]);

    expect(result).toBe(true);
    expect(router.createUrlTree).not.toHaveBeenCalled();
  });

  it("redirects unauthenticated users to login", async () => {
    const [{ AuthService }] = await Promise.all([import("../services/auth.service")]);
    const tree = { redirect: true };
    const router = { createUrlTree: vi.fn().mockReturnValue(tree) };

    const result = await runGuard([
      { provide: AuthService, useValue: { isAuthenticated: () => false } },
      { provide: Router, useValue: router },
    ]);

    expect(result).toBe(tree);
    expect(router.createUrlTree).toHaveBeenCalledWith(["/login"], {
      queryParams: { redirect: "/report" },
    });
  });
});
