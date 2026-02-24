import { HttpErrorResponse, HttpRequest } from "@angular/common/http";
import { createEnvironmentInjector, EnvironmentInjector, Provider, runInInjectionContext } from "@angular/core";
import { Router } from "@angular/router";
import { firstValueFrom, throwError } from "rxjs";

vi.mock("../services/auth.service", () => ({
  AuthService: class AuthService {},
}));

const runInterceptor = async (providers: Provider[], status: number): Promise<void> => {
  const [{ authErrorInterceptor }] = await Promise.all([
    import("./auth-error.interceptor"),
  ]);
  const parent = createEnvironmentInjector([]);
  const injector: EnvironmentInjector = createEnvironmentInjector(providers, parent);

  await runInInjectionContext(injector, async () => {
    const request = new HttpRequest("GET", "/v1/test");
    const response$ = authErrorInterceptor(request, () =>
      throwError(() => new HttpErrorResponse({ status })),
    );
    await firstValueFrom(response$);
  });
};

describe("authErrorInterceptor", () => {
  it("clears session and redirects to login on 401", async () => {
    const [{ AuthService }] = await Promise.all([import("../services/auth.service")]);
    const auth = { clearSession: vi.fn() };
    const router = { navigateByUrl: vi.fn().mockResolvedValue(true) };

    await expect(
      runInterceptor(
        [
          { provide: AuthService, useValue: auth },
          { provide: Router, useValue: router },
        ],
        401,
      ),
    ).rejects.toBeInstanceOf(HttpErrorResponse);

    expect(auth.clearSession).toHaveBeenCalledTimes(1);
    expect(router.navigateByUrl).toHaveBeenCalledWith("/login");
  });

  it("does not clear session for non-401 errors", async () => {
    const [{ AuthService }] = await Promise.all([import("../services/auth.service")]);
    const auth = { clearSession: vi.fn() };
    const router = { navigateByUrl: vi.fn().mockResolvedValue(true) };

    await expect(
      runInterceptor(
        [
          { provide: AuthService, useValue: auth },
          { provide: Router, useValue: router },
        ],
        500,
      ),
    ).rejects.toBeInstanceOf(HttpErrorResponse);

    expect(auth.clearSession).not.toHaveBeenCalled();
    expect(router.navigateByUrl).not.toHaveBeenCalled();
  });
});
