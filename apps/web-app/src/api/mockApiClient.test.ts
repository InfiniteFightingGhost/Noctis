import { describe, expect, it } from "vitest";
import { apiClient } from "./mockApiClient";

describe("api client endpoint semantics", () => {
  it("binds home and trends payloads to E-001 and E-002 semantics", async () => {
    const home = await apiClient.getHome();
    const trends = await apiClient.getTrends();

    expect(home.metrics.sleepScore).toBeTypeOf("number");
    expect(home.summaryHypnogram.epochs.length).toBeGreaterThan(0);
    expect(trends.nights.length).toBeGreaterThan(0);
    expect(trends.nights[0]).toMatchObject({
      fragmentationIndex: expect.any(Number),
      hrMean: expect.any(Number),
      hrv: expect.any(Number),
    });
  });

  it("returns nights list shape for E-003 and night detail for E-004", async () => {
    const nightsList = await apiClient.getNightsList();
    const nightDetail = await apiClient.getNight();

    expect(nightsList.nights[0]).toMatchObject({
      nightId: expect.any(String),
      hasCapData: expect.any(Boolean),
    });
    expect(nightDetail.epochs[0]).toMatchObject({
      epochIndex: expect.any(Number),
      confidence: expect.any(Number),
    });
  });

  it("splits settings profile/device across E-006 and E-007", async () => {
    const profile = await apiClient.getSettingsProfile();
    const device = await apiClient.getSettingsDevice();
    const settings = await apiClient.getSettings();

    expect(profile.profile.id).toMatch(/^usr_/);
    expect(profile.profile.username.length).toBeGreaterThan(0);
    expect(profile.profile.email).toContain("@");
    expect(device.device.id).toMatch(/^dev_/);
    expect(device.device.externalId).toContain("halo");
    expect(settings.profile.id).toBe(profile.profile.id);
    expect(settings.device.name).toBe(device.device.name);
  });

  it("maps actions to E-008 and E-009 semantics", async () => {
    const replaceResult = await apiClient.replaceDevice({ deviceExternalId: "noctis-halo-s1-001" });
    const exportResult = await apiClient.requestDataExport();
    const logoutResult = await apiClient.logout();
    const connectResult = await apiClient.connectDevice({ deviceExternalId: "noctis-halo-s1-001" });

    expect(replaceResult.message).toContain("Pairing started");
    expect(exportResult.message).toContain("Data export prepared");
    expect(exportResult.fileName).toContain("noctis-report-");
    expect(exportResult.report.recordingId).toBe("recording-001");
    expect(logoutResult.message).toBe("Action completed");
    expect(connectResult.message).toContain("noctis-halo-s1-001");
  });

  it("maps authentication flows to login and signup endpoints", async () => {
    const loginResult = await apiClient.login({
      email: "sample@noctis.example",
      password: "passw0rd!",
    });
    const signupResult = await apiClient.signup({
      username: "sample_user",
      email: "sample@noctis.example",
      password: "passw0rd!",
    });

    expect(loginResult.token_type).toBe("bearer");
    expect(signupResult.token_type).toBe("bearer");
    expect(loginResult.access_token).toBeTypeOf("string");
    expect(signupResult.access_token).toBeTypeOf("string");
    expect(loginResult.user.id).toMatch(/^usr_/);
    expect(signupResult.user.id).toMatch(/^usr_/);
  });
});
