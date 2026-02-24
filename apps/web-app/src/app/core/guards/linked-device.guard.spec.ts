import { createEnvironmentInjector, EnvironmentInjector, Provider, runInInjectionContext } from "@angular/core";
import { Router } from "@angular/router";
import { of, throwError } from "rxjs";

vi.mock("../api/devices.api", () => ({
  DevicesApi: class DevicesApi {},
}));

const runGuard = async (providers: Provider[]) => {
  const [{ linkedDeviceGuard }] = await Promise.all([import("./linked-device.guard")]);
  const parent = createEnvironmentInjector([]);
  const injector: EnvironmentInjector = createEnvironmentInjector(providers, parent);

  return runInInjectionContext(injector, () =>
    linkedDeviceGuard({} as any, { url: "/report" } as any),
  );
};

describe("linkedDeviceGuard", () => {
  it("allows access when user has a linked device", async () => {
    const [{ DevicesApi }] = await Promise.all([import("../api/devices.api")]);
    const router = { createUrlTree: vi.fn() };

    const result = await runGuard([
      {
        provide: DevicesApi,
        useValue: {
          getDevices: () =>
            of([
              {
                id: "1",
                name: "device",
                external_id: "ext-1",
                user_id: "user-1",
                created_at: "2026-01-01T00:00:00Z",
              },
            ]),
        },
      },
      { provide: Router, useValue: router },
    ]);

    expect(result).toBe(true);
    expect(router.createUrlTree).not.toHaveBeenCalled();
  });

  it("redirects to claim page when no linked device", async () => {
    const [{ DevicesApi }] = await Promise.all([import("../api/devices.api")]);
    const tree = { redirect: true };
    const router = { createUrlTree: vi.fn().mockReturnValue(tree) };

    const result = await runGuard([
      {
        provide: DevicesApi,
        useValue: {
          getDevices: () =>
            of([
              {
                id: "1",
                name: "device",
                external_id: "ext-1",
                user_id: null,
                created_at: "2026-01-01T00:00:00Z",
              },
            ]),
        },
      },
      { provide: Router, useValue: router },
    ]);

    expect(result).toBe(tree);
    expect(router.createUrlTree).toHaveBeenCalledWith(["/device/claim"], {
      queryParams: { redirect: "/report" },
    });
  });

  it("redirects to claim page when device lookup fails", async () => {
    const [{ DevicesApi }] = await Promise.all([import("../api/devices.api")]);
    const tree = { redirect: true };
    const router = { createUrlTree: vi.fn().mockReturnValue(tree) };

    const result = await runGuard([
      {
        provide: DevicesApi,
        useValue: {
          getDevices: () => throwError(() => new Error("failed")),
        },
      },
      { provide: Router, useValue: router },
    ]);

    expect(result).toBe(tree);
    expect(router.createUrlTree).toHaveBeenCalledWith(["/device/claim"], {
      queryParams: { redirect: "/report" },
    });
  });
});
