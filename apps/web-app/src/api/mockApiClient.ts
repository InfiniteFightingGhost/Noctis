import {
  actionResponseSchema,
  authResponseSchema,
  backendDevicePairingStartResponseSchema,
  backendSleepSummarySchema,
  connectDeviceRequestSchema,
  dataExportResponseSchema,
  devicePairingStartRequestSchema,
  homeSchema,
  loginRequestSchema,
  nightsListSchema,
  nightSchema,
  settingsDeviceSchemaEndpoint,
  settingsProfileSchemaEndpoint,
  signupRequestSchema,
  trendsSchema,
  type ActionResponse,
  type AuthResponse,
  type ConnectDeviceRequest,
  type DataExportResponse,
  type DevicePairingStartRequest,
  type HomeResponse,
  type TrendsFilter,
  type LoginRequest,
  type NightsListResponse,
  type NightResponse,
  type SettingsDeviceResponse,
  type SettingsProfileResponse,
  type SettingsResponse,
  type SignupRequest,
  type TrendsResponse,
} from "./contracts";

const mockHomePayload: HomeResponse = {
  date: "2026-02-26",
  metrics: {
    sleepScore: 84,
    totalSleepMinutes: 418,
    sleepEfficiency: 92,
    remPercent: 23,
    deepPercent: 19,
    deltaVs7DayBaseline: {
      sleepScore: 3,
      totalSleepMinutes: 22,
      sleepEfficiency: 1.4,
      remPercent: -0.8,
      deepPercent: 1.1,
    },
  },
  summaryHypnogram: {
    confidenceSummary: "Median confidence 0.89 with 7 low-confidence transitions.",
    epochs: Array.from({ length: 60 }).map((_, i) => {
      const cycle = i % 14;
      const stage = cycle < 2 ? "wake" : cycle < 9 ? "light" : cycle < 12 ? "deep" : "rem";
      const confidence = 0.78 + (cycle / 15) * 0.2;
      return {
        epochIndex: i,
        stage,
        confidence,
        probabilities: {
          wake: stage === "wake" ? 0.82 : 0.06,
          light: stage === "light" ? 0.86 : 0.05,
          deep: stage === "deep" ? 0.81 : 0.04,
          rem: stage === "rem" ? 0.84 : 0.05,
        },
      };
    }),
  },
  stageBreakdown: [
    { stage: "wake", minutes: 33, percent: 8 },
    { stage: "light", minutes: 209, percent: 50 },
    { stage: "deep", minutes: 79, percent: 19 },
    { stage: "rem", minutes: 97, percent: 23 },
  ],
  latencyTriplet: {
    sleepOnsetMinutes: 14,
    remLatencyMinutes: 79,
    deepLatencyMinutes: 37,
  },
  continuityMetrics: {
    fragmentationIndex: 0.17,
    entropy: 0.44,
    wasoMinutes: 31,
  },
  aiSummary:
    "Sleep continuity and architecture are improved relative to baseline, with modest REM suppression early in the night and stable deep sleep capture across the final two cycles.",
};

const mockTrendsPayload: TrendsResponse = {
  activeFilter: "30D",
  nights: Array.from({ length: 12 }).map((_, i) => ({
    date: `2026-02-${String(i + 1).padStart(2, "0")}`,
    sleepScore: 70 + i,
    totalSleepMinutes: 360 + i * 7,
    sleepEfficiency: 86 + i * 0.4,
    remPercent: 20 + i * 0.2,
    deepPercent: 16 + i * 0.15,
    fragmentationIndex: 0.28 - i * 0.005,
    hrMean: 63 - i * 0.2,
    hrv: 38 + i * 0.9,
    consistencyIndex: 73 + i * 0.8,
  })),
  movingAverageWindow: 7,
  varianceBand: {
    lower: 74,
    upper: 88,
  },
  worstNightDecile: {
    date: "2026-02-03",
    sleepScore: 71,
  },
};

function getWindowSize(filter: TrendsFilter): number {
  if (filter === "7D") {
    return 7;
  }
  if (filter === "30D") {
    return 30;
  }
  if (filter === "90D") {
    return 90;
  }
  return 5;
}

function buildTrendsPayload(filter: TrendsFilter): TrendsResponse {
  const windowSize = getWindowSize(filter);
  const nights = mockTrendsPayload.nights.slice(-Math.min(windowSize, mockTrendsPayload.nights.length));
  const worstNight = nights.reduce(
    (lowest, current) => (current.sleepScore < lowest.sleepScore ? current : lowest),
    nights[0] ?? mockTrendsPayload.worstNightDecile,
  );

  return trendsSchema.parse({
    ...mockTrendsPayload,
    activeFilter: filter,
    nights,
    worstNightDecile: {
      date: worstNight.date,
      sleepScore: worstNight.sleepScore,
    },
  });
}

const mockNightsListPayload: NightsListResponse = {
  nights: mockTrendsPayload.nights.map((night, index) => ({
    nightId: `night-${index + 1}`,
    date: night.date,
    label: `Night ${index + 1}`,
    hasCapData: index % 3 !== 0,
  })),
};

const mockNightPayload: NightResponse = {
  date: "2026-02-26",
  epochs: mockHomePayload.summaryHypnogram.epochs.map((epoch, index) => ({
    ...epoch,
    epochIndex: index,
    confidence: Math.max(0.65, epoch.confidence - (index % 11) * 0.01),
  })),
  transitions: [
    [12, 6, 1, 1],
    [5, 42, 8, 10],
    [1, 9, 17, 4],
    [1, 6, 5, 20],
  ],
  arousalIndex: 8.4,
  capRateConditional: {
    available: true,
    value: 0.27,
  },
  cardiopulmonary: {
    avgRespiratoryRate: 14.8,
    minSpO2: 92,
    avgHeartRate: 58,
  },
};

const mockSettingsProfilePayload: SettingsProfileResponse = {
  profile: {
    id: "usr_01J6QK9F6N3M4B5C6D7E8F9G",
    username: "jordan_leung",
    email: "jordan.leung@noctishealth.example",
    createdAt: "2026-02-20T08:12:00Z",
    updatedAt: "2026-02-27T09:20:00Z",
  },
};

const mockSettingsDevicePayload: SettingsDeviceResponse = {
  device: {
    id: "dev_01J6QK9F6N3M4B5C6D7E8F9H",
    name: "Noctis Halo S1 Mount",
    externalId: "noctis-halo-s1-001",
    userId: "usr_01J6QK9F6N3M4B5C6D7E8F9G",
    createdAt: "2026-02-19T21:14:00Z",
  },
};

const knownDeviceExternalIds = new Set(["noctis-halo-s1-001"]);

const okAction: ActionResponse = {
  success: true,
  message: "Operation completed",
};

const mockAuthResponse: AuthResponse = {
  access_token: "mock-access-token",
  token_type: "bearer",
  expires_in: 3600,
  user: {
    id: "usr_01J6QK9F6N3M4B5C6D7E8F9G",
    username: "sample_user",
    email: "sample@noctis.example",
    created_at: "2026-02-27T00:00:00Z",
    updated_at: "2026-02-27T00:00:00Z",
  },
};

const endpointRegistry = {
  "E-001": () => homeSchema.parse(mockHomePayload),
  "E-002": (filter: TrendsFilter) => buildTrendsPayload(filter),
  "E-003": () => nightsListSchema.parse(mockNightsListPayload),
  "E-004": () => nightSchema.parse(mockNightPayload),
  "E-006": () => settingsProfileSchemaEndpoint.parse(mockSettingsProfilePayload),
  "E-007": () => settingsDeviceSchemaEndpoint.parse(mockSettingsDevicePayload),
  "E-008": () => actionResponseSchema.parse({ ...okAction, message: "Device replacement initiated" }),
  "E-009": () => actionResponseSchema.parse({ ...okAction, message: "Action completed" }),
  "E-010": () => authResponseSchema.parse(mockAuthResponse),
  "E-011": () =>
    authResponseSchema.parse({
      ...mockAuthResponse,
      user: {
        ...mockAuthResponse.user,
        id: "usr_01J6QK9F6N3M4B5C6D7E8F9G_SIGNUP",
      },
    }),
};

type EndpointId = keyof typeof endpointRegistry;

async function fetchEndpoint<T>(endpointId: EndpointId, ...args: unknown[]): Promise<T> {
  await new Promise((resolve) => setTimeout(resolve, 120));
  const resolver = endpointRegistry[endpointId] as (...params: unknown[]) => unknown;
  return resolver(...args) as T;
}

export const apiClient = {
  getHome: () => fetchEndpoint<HomeResponse>("E-001"),
  getTrends: (filter: TrendsFilter = "30D") => fetchEndpoint<TrendsResponse>("E-002", filter),
  getNightsList: () => fetchEndpoint<NightsListResponse>("E-003"),
  getNight: () => fetchEndpoint<NightResponse>("E-004"),
  getSettingsProfile: () => fetchEndpoint<SettingsProfileResponse>("E-006"),
  getSettingsDevice: () => fetchEndpoint<SettingsDeviceResponse>("E-007"),
  getSettings: async (): Promise<SettingsResponse> => {
    const [profile, device] = await Promise.all([
      fetchEndpoint<SettingsProfileResponse>("E-006"),
      fetchEndpoint<SettingsDeviceResponse>("E-007"),
    ]);
    return { ...profile, ...device };
  },
  replaceDevice: (payload: DevicePairingStartRequest) => {
    const parsed = devicePairingStartRequestSchema.parse(payload);
    const pairing = backendDevicePairingStartResponseSchema.parse({
      pairing_session_id: "pairing-session-001",
      pairing_code: "A1B2C3",
      expires_at: new Date(Date.now() + 10 * 60 * 1000).toISOString(),
    });
    return actionResponseSchema.parse({
      ...okAction,
      message: `Pairing started for ${parsed.deviceExternalId ?? parsed.deviceId}. Code ${pairing.pairing_code}.`,
    });
  },
  requestDataExport: async (): Promise<DataExportResponse> => {
    const report = backendSleepSummarySchema.parse({
      recordingId: "recording-001",
      dateLocal: "2026-02-26",
      score: 84,
      totals: {
        totalSleepMin: 418,
        timeInBedMin: 455,
        sleepEfficiencyPct: 92,
      },
      metrics: {
        deepPct: 19,
        avgHrBpm: 58,
        avgRrBrpm: 14,
        movementPct: 12,
      },
    });

    return dataExportResponseSchema.parse({
      ...okAction,
      message: `Data export prepared for recording ${report.recordingId}.`,
      fileName: `noctis-report-${report.dateLocal}.json`,
      report,
    });
  },
  connectDevice: (payload: ConnectDeviceRequest) => {
    connectDeviceRequestSchema.parse(payload);
    if (!knownDeviceExternalIds.has(payload.deviceExternalId)) {
      throw new Error("Device not found. Check the external ID and try again.");
    }

    return actionResponseSchema.parse({
      ...okAction,
      message: `Mountable device ${payload.deviceExternalId} connected.`,
    });
  },
  logout: () => fetchEndpoint<ActionResponse>("E-009"),
  updatePreferences: () => fetchEndpoint<ActionResponse>("E-009"),
  login: (payload: LoginRequest) => {
    loginRequestSchema.parse(payload);
    return fetchEndpoint<AuthResponse>("E-010");
  },
  signup: (payload: SignupRequest) => {
    signupRequestSchema.parse(payload);
    return fetchEndpoint<AuthResponse>("E-011").then((result) =>
      authResponseSchema.parse({
        ...result,
        user: {
          ...result.user,
          username: payload.username,
        },
      }),
    );
  },
};
