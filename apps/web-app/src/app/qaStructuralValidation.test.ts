import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";
import { ROUTE_TABLE, routeObjects } from "./routes";

const ROOT = resolve(__dirname, "..");

function readSource(relativePath: string): string {
  return readFileSync(resolve(ROOT, relativePath), "utf8");
}

describe("phase 7 structural validation", () => {
  it("keeps exact top-level tabs and exact route ownership", () => {
    expect(ROUTE_TABLE).toEqual([
      { path: "/", label: "Home" },
      { path: "/trends", label: "Trends" },
      { path: "/night", label: "Night" },
      { path: "/settings", label: "Settings" },
    ]);

    const rootRoute = routeObjects[0];
    const children = rootRoute.children?.map((child) => ("index" in child && child.index ? "index" : child.path));
    expect(children).toEqual(["index", "trends", "night", "settings"]);
    expect(ROUTE_TABLE.some((route) => String(route.label) === "Devices" || String(route.path) === "/devices")).toBe(false);
    expect(ROUTE_TABLE.some((route) => String(route.label) === "Insights" || String(route.path) === "/insights")).toBe(false);
  });

  it("binds api client calls to endpoint map semantics", () => {
    const apiSource = readSource("api/mockApiClient.ts");
    const contractsSource = readSource("api/contracts.ts");

    ["E-001", "E-002", "E-003", "E-004", "E-006", "E-007", "E-008", "E-009", "E-010", "E-011"].forEach((endpointId) => {
      expect(apiSource).toContain(`"${endpointId}":`);
    });

    const directMethodMap = [
      ["getHome", "E-001"],
      ["getNightsList", "E-003"],
      ["getNight", "E-004"],
      ["getSettingsProfile", "E-006"],
      ["getSettingsDevice", "E-007"],
      ["logout", "E-009"],
      ["updatePreferences", "E-009"],
    ] as const;

    directMethodMap.forEach(([methodName, endpointId]) => {
      expect(apiSource).toContain(`${methodName}: () => fetchEndpoint`);
      expect(apiSource).toContain(`("${endpointId}")`);
    });

    expect(apiSource).toContain("getTrends: (filter: TrendsFilter = \"30D\")");
    expect(apiSource).toContain("fetchEndpoint<TrendsResponse>(\"E-002\", filter)");
    expect(apiSource).toContain("connectDevice: (payload: ConnectDeviceRequest)");
    expect(apiSource).toContain("connectDeviceRequestSchema.parse(payload)");
    expect(apiSource).toContain("replaceDevice: (payload: DevicePairingStartRequest)");
    expect(apiSource).toContain("devicePairingStartRequestSchema.parse(payload)");
    expect(apiSource).toContain("fetchEndpoint<ActionResponse>(\"E-009\")");

    expect(apiSource).toContain('getSettings: async (): Promise<SettingsResponse>');
    expect(apiSource).toContain('fetchEndpoint<SettingsProfileResponse>("E-006")');
    expect(apiSource).toContain('fetchEndpoint<SettingsDeviceResponse>("E-007")');
    expect(contractsSource).toContain("export const nightsListSchema");
    expect(contractsSource).toContain("export const settingsProfileSchemaEndpoint");
    expect(contractsSource).toContain("export const settingsDeviceSchemaEndpoint");
    expect(contractsSource).not.toContain("export const insightsSchema");
  });

  it("keeps single-device management logic scoped to settings", () => {
    const settingsSource = readSource("pages/SettingsPage.tsx");
    const homeSource = readSource("pages/HomePage.tsx");
    const trendsSource = readSource("pages/TrendsPage.tsx");
    const nightSource = readSource("pages/NightPage.tsx");

    expect(settingsSource).toContain("single-device-section");
    expect(settingsSource).toContain("replaceDevice");
    expect(settingsSource).toContain("getSettings");
    expect(settingsSource).toContain("requestDataExport");
    expect(settingsSource).toContain("logout");

    [homeSource, trendsSource, nightSource].forEach((source) => {
      expect(source).not.toContain("replaceDevice");
      expect(source).not.toContain("getSettings");
      expect(source).not.toContain("requestDataExport");
      expect(source).not.toContain("single-device-section");
    });
  });

  it("avoids duplicated top-level visualization ownership", () => {
    const homeSource = readSource("pages/HomePage.tsx");
    const trendsSource = readSource("pages/TrendsPage.tsx");
    const nightSource = readSource("pages/NightPage.tsx");

    expect(homeSource).toContain('<Hypnogram\n          mode="summary"\n          context="home"');
    expect(nightSource).toContain('<Hypnogram mode="detail" context="night"');

    expect(trendsSource).not.toContain("Hypnogram");
    expect(homeSource).not.toContain("LineChart");
    expect(nightSource).not.toContain("LineChart");
    expect(trendsSource).toContain("LineChart");
  });

  it("covers required trends metrics and night CAP unavailable handling", () => {
    const trendsSource = readSource("pages/TrendsPage.tsx");
    const contractsSource = readSource("api/contracts.ts");
    const nightSource = readSource("pages/NightPage.tsx");

    [
      "Sleep Score Longitudinal",
      "TST Trend",
      "REM Percent Longitudinal",
      "Deep Percent Longitudinal",
      "Fragmentation Longitudinal",
      "HR Mean Longitudinal",
      "HRV Longitudinal",
    ].forEach((label) => {
      expect(trendsSource).toContain(label);
    });

    ["fragmentationIndex", "hrMean", "hrv"].forEach((field) => {
      expect(contractsSource).toContain(field);
    });

    expect(contractsSource).toContain("available: z.literal(false)");
    expect(nightSource).toContain("Unavailable");
    expect(nightSource).toContain("if available");
  });
});
