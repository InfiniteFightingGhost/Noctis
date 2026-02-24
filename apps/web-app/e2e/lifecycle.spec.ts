import { APIRequestContext, expect, test } from "@playwright/test";

const apiBaseUrl = process.env.E2E_API_BASE_URL ?? "http://localhost:8000";

test("lifecycle e2e: auth to nightly iteration", async ({ page, request }) => {
  const runId = Date.now();
  const email = `e2e-${runId}@example.com`;
  const password = "Str0ngPassw0rd!";

  await page.goto("/signup");

  const loginResponsePromise = page.waitForResponse(
    (response) =>
      response.url().includes("/v1/auth/login") && response.request().method() === "POST",
  );

  await page.getByLabel("Email").fill(email);
  await page.getByLabel("Password", { exact: true }).fill(password);
  await page.getByLabel("Confirm password").fill(password);
  await page.getByRole("button", { name: "Create account" }).click();

  await expect(page).toHaveURL(/\/dashboard/);
  const loginResponse = await loginResponsePromise;
  const loginPayload = (await loginResponse.json()) as { access_token: string };
  const accessToken = loginPayload.access_token;

  const authHeaders = {
    Authorization: `Bearer ${accessToken}`,
  };

  const deviceName = `E2E Device ${runId}`;
  const createDeviceResponse = await request.post(`${apiBaseUrl}/v1/devices`, {
    headers: authHeaders,
    data: { name: deviceName },
  });
  expect(createDeviceResponse.ok()).toBeTruthy();

  await page.getByRole("link", { name: "Device" }).click();
  await expect(page).toHaveURL(/\/device/);
  await expect(page.getByRole("heading", { name: deviceName })).toBeVisible();

  const alarmUpdateResponse = await request.put(`${apiBaseUrl}/v1/alarm`, {
    headers: authHeaders,
    data: {
      wake_time: "06:30",
      wake_window_minutes: 15,
      sunrise_enabled: true,
      sunrise_intensity: 4,
      sound_id: "forest",
    },
  });
  expect(alarmUpdateResponse.ok()).toBeTruthy();

  await page.getByRole("link", { name: "Alarm" }).click();
  await expect(page).toHaveURL(/\/alarm/);
  await page.getByRole("link", { name: "Advanced settings" }).click();
  await expect(page).toHaveURL(/\/alarm\/settings/);
  await expect(page.getByText("Wake time: 06:30 with 15 minute window")).toBeVisible();

  await page.getByRole("link", { name: "Start Routine" }).click();
  await expect(page).toHaveURL(/\/routine/);

  const routineUpdateResponse = await request.put(`${apiBaseUrl}/v1/routines/current`, {
    headers: authHeaders,
    data: {
      title: "Night Winddown",
      steps: [
        {
          title: "Read book",
          duration_minutes: 12,
          emoji: ":)",
        },
      ],
    },
  });
  expect(routineUpdateResponse.ok()).toBeTruthy();

  await page.getByRole("link", { name: "Alarm" }).click();
  await page.getByRole("link", { name: "Advanced settings" }).click();
  await page.getByRole("link", { name: "Start Routine" }).click();
  await expect(page.getByText("Read book", { exact: true })).toBeVisible();

  const firstRecordingId = await createSleepCycle(request, authHeaders, 0);
  await page.getByRole("link", { name: "Dashboard" }).click();
  await expect(page).toHaveURL(/\/dashboard/);
  await expect(page.getByText("Last night")).toBeVisible();
  await expect(page.getByRole("button", { name: "Improve Tonight" })).toBeVisible();

  const secondRecordingId = await createSleepCycle(request, authHeaders, 1);
  expect(secondRecordingId).not.toEqual(firstRecordingId);

  const latestSummaryResponse = await request.get(`${apiBaseUrl}/v1/sleep/latest/summary`, {
    headers: authHeaders,
  });
  expect(latestSummaryResponse.ok()).toBeTruthy();
  const latestSummary = (await latestSummaryResponse.json()) as { recordingId: string };
  expect(latestSummary.recordingId).toEqual(secondRecordingId);
});

async function createSleepCycle(
  request: APIRequestContext,
  authHeaders: Record<string, string>,
  cycle: number,
): Promise<string> {
  const devicesResponse = await request.get(`${apiBaseUrl}/v1/devices`, {
    headers: authHeaders,
  });
  expect(devicesResponse.ok()).toBeTruthy();
  const devices = (await devicesResponse.json()) as Array<{ id: string }>;
  expect(devices.length).toBeGreaterThan(0);

  const startedAt = new Date(Date.now() + cycle * 60_000).toISOString();
  const recordingResponse = await request.post(`${apiBaseUrl}/v1/recordings`, {
    headers: authHeaders,
    data: {
      device_id: devices[0].id,
      started_at: startedAt,
    },
  });
  expect(recordingResponse.ok()).toBeTruthy();
  const recording = (await recordingResponse.json()) as { id: string };

  const recordingStart = new Date(startedAt).getTime();
  const epochs = Array.from({ length: 21 }).map((_, index) => ({
    epoch_index: index,
    epoch_start_ts: new Date(recordingStart + index * 30_000).toISOString(),
    feature_schema_version: "v1",
    features: Array.from({ length: 10 }).map((__, featureIndex) =>
      Number((0.1 + cycle * 0.01 + featureIndex * 0.001).toFixed(3)),
    ),
  }));

  const ingestResponse = await request.post(`${apiBaseUrl}/v1/epochs:ingest`, {
    headers: authHeaders,
    data: {
      recording_id: recording.id,
      epochs,
    },
  });
  expect(ingestResponse.ok()).toBeTruthy();

  const predictResponse = await request.post(`${apiBaseUrl}/v1/predict`, {
    headers: authHeaders,
    data: {
      recording_id: recording.id,
    },
  });
  expect(predictResponse.ok()).toBeTruthy();

  return recording.id;
}
