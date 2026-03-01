import { z, type ZodSchema } from "zod";
import {
  actionResponseSchema,
  authResponseSchema,
  backendAuthMeResponseSchema,
  backendDeviceResponseSchema,
  backendEpochResponseSchema,
  backendPredictionResponseSchema,
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
  trendsFilterSchema,
  trendsSchema,
  type ActionResponse,
  type AuthResponse,
  type BackendDeviceResponse,
  type BackendEpochResponse,
  type BackendPredictionResponse,
  type BackendSleepSummary,
  type ConnectDeviceRequest,
  type DataExportResponse,
  type DevicePairingStartRequest,
  type HomeResponse,
  type LoginRequest,
  type NightsListResponse,
  type NightResponse,
  type SettingsResponse,
  type SignupRequest,
  type TrendsFilter,
  type TrendsResponse,
} from "./contracts";
import { clearAuthSession, getAccessToken, getRefreshToken, setAuthSession } from "../auth/session";

type ErrorKind = "network" | "timeout" | "client" | "server" | "unknown";

export class ApiError extends Error {
  kind: ErrorKind;
  status: number | null;

  constructor(message: string, kind: ErrorKind, status: number | null = null) {
    super(message);
    this.name = "ApiError";
    this.kind = kind;
    this.status = status;
  }
}

type RequestConfig<T> = {
  method: "GET" | "POST";
  path: string;
  schema: ZodSchema<T>;
  body?: unknown;
  auth?: boolean;
};

const DEFAULT_TIMEOUT_MS = 6000;
const MAX_RETRIES = 1;
const NO_DATA_STATUSES = new Set([404, 405]);

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL ?? "").replace(/\/$/, "");
const API_PREFIX = ((import.meta.env.VITE_API_V1_PREFIX as string | undefined)?.trim() || "/v1").replace(/\/$/, "");
const REFRESH_PATH = "/auth/refresh";

let refreshInFlight: Promise<boolean> | null = null;

function deriveUsername(username: string | undefined, email: string): string {
  if (username && username.trim().length > 0) {
    return username.trim();
  }

  const emailPrefix = email.split("@")[0]?.trim();
  return emailPrefix && emailPrefix.length > 0 ? emailPrefix : "user";
}

function getErrorMessage(status: number): string {
  if (status >= 500) {
    return "Server error. Please try again.";
  }
  if (status >= 400) {
    return "Request could not be completed.";
  }
  return "Unexpected API response.";
}

function extractApiErrorMessage(payload: unknown, fallbackMessage: string): string {
  if (!payload || typeof payload !== "object") {
    return fallbackMessage;
  }

  if ("message" in payload && typeof payload.message === "string") {
    return payload.message;
  }

  if (
    "error" in payload &&
    payload.error &&
    typeof payload.error === "object" &&
    "message" in payload.error &&
    typeof payload.error.message === "string"
  ) {
    return payload.error.message;
  }

  if ("detail" in payload && typeof payload.detail === "string") {
    return payload.detail;
  }

  return fallbackMessage;
}

function isNoDataError(error: unknown): error is ApiError {
  return error instanceof ApiError && error.status !== null && NO_DATA_STATUSES.has(error.status);
}

function normalizeStageLabel(stage: string): "wake" | "light" | "deep" | "rem" {
  const normalized = stage.trim().toLowerCase();
  if (normalized === "wake" || normalized === "awake" || normalized === "w") {
    return "wake";
  }
  if (normalized === "deep" || normalized === "n3") {
    return "deep";
  }
  if (normalized === "rem" || normalized === "r") {
    return "rem";
  }
  return "light";
}

function buildEmptyHome(): HomeResponse {
  return homeSchema.parse({
    date: new Date().toISOString().slice(0, 10),
    metrics: {
      sleepScore: 0,
      totalSleepMinutes: 0,
      sleepEfficiency: 0,
      remPercent: 0,
      deepPercent: 0,
      deltaVs7DayBaseline: {
        sleepScore: 0,
        totalSleepMinutes: 0,
        sleepEfficiency: 0,
        remPercent: 0,
        deepPercent: 0,
      },
    },
    summaryHypnogram: {
      confidenceSummary: "No data for now.",
      epochs: [],
    },
    stageBreakdown: [],
    latencyTriplet: {
      sleepOnsetMinutes: 0,
      remLatencyMinutes: 0,
      deepLatencyMinutes: 0,
    },
    continuityMetrics: {
      fragmentationIndex: 0,
      entropy: 0,
      wasoMinutes: 0,
    },
    aiSummary: "No data for now.",
  });
}

function buildHomeFromSummary(summary: BackendSleepSummary): HomeResponse {
  const bins = summary.stages?.bins ?? [];
  const stagePct = summary.stages?.pct;
  const totalSleepMinutes = summary.totals.totalSleepMin;

  return homeSchema.parse({
    date: summary.dateLocal,
    metrics: {
      sleepScore: summary.score,
      totalSleepMinutes,
      sleepEfficiency: summary.totals.sleepEfficiencyPct,
      remPercent: stagePct?.rem ?? 0,
      deepPercent: summary.metrics.deepPct ?? stagePct?.deep ?? 0,
      deltaVs7DayBaseline: {
        sleepScore: 0,
        totalSleepMinutes: 0,
        sleepEfficiency: 0,
        remPercent: 0,
        deepPercent: 0,
      },
    },
    summaryHypnogram: {
      confidenceSummary: summary.insight?.text ?? "No confidence summary available.",
      epochs: bins.map((bin, index) => {
        const stage = normalizeStageLabel(bin.stage);
        return {
          epochIndex: index,
          stage,
          confidence: 0,
          probabilities: {
            wake: stage === "wake" ? 1 : 0,
            light: stage === "light" ? 1 : 0,
            deep: stage === "deep" ? 1 : 0,
            rem: stage === "rem" ? 1 : 0,
          },
        };
      }),
    },
    stageBreakdown: stagePct
      ? [
          { stage: "wake", minutes: Math.round((totalSleepMinutes * stagePct.awake) / 100), percent: stagePct.awake },
          { stage: "light", minutes: Math.round((totalSleepMinutes * stagePct.light) / 100), percent: stagePct.light },
          { stage: "deep", minutes: Math.round((totalSleepMinutes * stagePct.deep) / 100), percent: stagePct.deep },
          { stage: "rem", minutes: Math.round((totalSleepMinutes * stagePct.rem) / 100), percent: stagePct.rem },
        ]
      : [],
    latencyTriplet: {
      sleepOnsetMinutes: 0,
      remLatencyMinutes: 0,
      deepLatencyMinutes: 0,
    },
    continuityMetrics: {
      fragmentationIndex: 0,
      entropy: 0,
      wasoMinutes: 0,
    },
    aiSummary: summary.insight?.text ?? "No data for now.",
  });
}

function buildEmptyTrends(filter: TrendsFilter): TrendsResponse {
  return trendsSchema.parse({
    activeFilter: filter,
    nights: [],
    movingAverageWindow: 7,
    varianceBand: { lower: 0, upper: 0 },
    worstNightDecile: { date: "", sleepScore: 0 },
  });
}

function buildTrendsFromSummary(summary: BackendSleepSummary, filter: TrendsFilter): TrendsResponse {
  const stagePct = summary.stages?.pct;
  return trendsSchema.parse({
    activeFilter: filter,
    nights: [
      {
        date: summary.dateLocal,
        sleepScore: summary.score,
        totalSleepMinutes: summary.totals.totalSleepMin,
        sleepEfficiency: summary.totals.sleepEfficiencyPct,
        remPercent: stagePct?.rem ?? 0,
        deepPercent: summary.metrics.deepPct ?? stagePct?.deep ?? 0,
        fragmentationIndex: 0,
        hrMean: summary.metrics.avgHrBpm,
        hrv: 0,
        consistencyIndex: summary.score,
      },
    ],
    movingAverageWindow: 1,
    varianceBand: { lower: summary.score, upper: summary.score },
    worstNightDecile: { date: summary.dateLocal, sleepScore: summary.score },
  });
}

function buildEmptyNight(): NightResponse {
  return nightSchema.parse({
    date: new Date().toISOString().slice(0, 10),
    epochs: [],
    transitions: [],
    arousalIndex: 0,
    capRateConditional: {
      available: false,
      reason: "No data for now.",
    },
    cardiopulmonary: {
      avgRespiratoryRate: 0,
      minSpO2: 0,
      avgHeartRate: 0,
    },
  });
}

function buildNightFromSummary(summary: BackendSleepSummary): NightResponse {
  const bins = summary.stages?.bins ?? [];
  return nightSchema.parse({
    date: summary.dateLocal,
    epochs: bins.map((bin, index) => {
      const stage = normalizeStageLabel(bin.stage);
      return {
        epochIndex: index,
        stage,
        confidence: 0,
        probabilities: {
          wake: stage === "wake" ? 1 : 0,
          light: stage === "light" ? 1 : 0,
          deep: stage === "deep" ? 1 : 0,
          rem: stage === "rem" ? 1 : 0,
        },
      };
    }),
    transitions: [],
    arousalIndex: 0,
    capRateConditional: {
      available: false,
      reason: "No CAP data for now.",
    },
    cardiopulmonary: {
      avgRespiratoryRate: summary.metrics.avgRrBrpm,
      minSpO2: 0,
      avgHeartRate: summary.metrics.avgHrBpm,
    },
  });
}

function buildNightsListFromSummary(summary: BackendSleepSummary): NightsListResponse {
  return nightsListSchema.parse({
    nights: [
      {
        nightId: summary.recordingId,
        date: summary.dateLocal,
        label: summary.dateLocal,
        hasCapData: false,
      },
    ],
  });
}

function shouldRetry(status: number | null, error: unknown): boolean {
  if (error instanceof ApiError && (error.kind === "network" || error.kind === "timeout")) {
    return true;
  }
  if (status !== null && status >= 500) {
    return true;
  }
  return false;
}

function buildApiUrl(path: string): string {
  const normalizedPath = path.startsWith("/") ? path : `/${path}`;
  if (!API_BASE_URL) {
    return `${API_PREFIX}${normalizedPath}`;
  }

  const alreadyPrefixed = API_BASE_URL.endsWith(API_PREFIX);
  return alreadyPrefixed
    ? `${API_BASE_URL}${normalizedPath}`
    : `${API_BASE_URL}${API_PREFIX}${normalizedPath}`;
}

async function parseJsonSafe(response: Response): Promise<unknown> {
  try {
    return await response.json();
  } catch {
    return null;
  }
}

async function attemptTokenRefresh(): Promise<boolean> {
  const refreshToken = getRefreshToken();

  if (!refreshToken) {
    return false;
  }

  if (refreshInFlight) {
    return refreshInFlight;
  }

  refreshInFlight = (async () => {
    try {
      const response = await fetch(buildApiUrl(REFRESH_PATH), {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ refreshToken }),
      });

      if (!response.ok) {
        return false;
      }

      const payload = await parseJsonSafe(response);
      const parsed = authResponseSchema.parse(payload);
      setAuthSession({
        accessToken: parsed.access_token,
        refreshToken: refreshToken,
        userId: parsed.user.id,
      });
      return true;
    } catch {
      return false;
    } finally {
      refreshInFlight = null;
    }
  })();

  return refreshInFlight;
}

async function sendRequest<T>(config: RequestConfig<T>): Promise<T> {
  let attemptedRefresh = false;

  for (let attempt = 0; attempt <= MAX_RETRIES; attempt += 1) {
    const controller = new AbortController();
    const timeoutId = window.setTimeout(() => controller.abort("timeout"), DEFAULT_TIMEOUT_MS);
    let status: number | null = null;

    try {
      const accessToken = getAccessToken();
      const shouldAttachAuth = config.auth !== false;

      const response = await fetch(buildApiUrl(config.path), {
        method: config.method,
        headers: {
          "Content-Type": "application/json",
          ...(shouldAttachAuth && accessToken ? { Authorization: `Bearer ${accessToken}` } : {}),
        },
        body: config.body === undefined ? undefined : JSON.stringify(config.body),
        signal: controller.signal,
      });

      status = response.status;
      const payload = await parseJsonSafe(response);

      if (!response.ok) {
        if (response.status === 401 && config.auth !== false && !attemptedRefresh) {
          const refreshSucceeded = await attemptTokenRefresh();
          if (refreshSucceeded) {
            attemptedRefresh = true;
            continue;
          }

          clearAuthSession();
          throw new ApiError("Session expired. Please log in again.", "client", 401);
        }

        const fallbackMessage = getErrorMessage(response.status);
        const message = extractApiErrorMessage(payload, fallbackMessage);
        throw new ApiError(message, response.status >= 500 ? "server" : "client", response.status);
      }

      return config.schema.parse(payload);
    } catch (error) {
      const isAbortError = error instanceof DOMException && error.name === "AbortError";
      const mappedError =
        isAbortError
          ? new ApiError("Request timed out. Please try again.", "timeout")
          : error instanceof TypeError
            ? new ApiError("Network unavailable. Check your connection.", "network")
            : error instanceof z.ZodError
              ? new ApiError("Response validation failed.", "server", status)
              : error instanceof ApiError
                ? error
                : new ApiError("Unexpected error.", "unknown", status);

      if (attempt < MAX_RETRIES && shouldRetry(status, mappedError)) {
        continue;
      }

      throw mappedError;
    } finally {
      window.clearTimeout(timeoutId);
    }
  }

  throw new ApiError("Unexpected error.", "unknown");
}

export const httpApiClient = {
  getLatestSleepSummary: () =>
    sendRequest<BackendSleepSummary>({
      method: "GET",
      path: "/sleep/latest/summary",
      schema: backendSleepSummarySchema,
    }),
  getRecordingEpochs: (recordingId: string, fromIso: string, toIso: string) =>
    sendRequest<BackendEpochResponse[]>({
      method: "GET",
      path: `/recordings/${encodeURIComponent(recordingId)}/epochs?from=${encodeURIComponent(fromIso)}&to=${encodeURIComponent(toIso)}`,
      schema: z.array(backendEpochResponseSchema),
    }),
  getRecordingPredictions: (recordingId: string, fromIso: string, toIso: string) =>
    sendRequest<BackendPredictionResponse[]>({
      method: "GET",
      path: `/recordings/${encodeURIComponent(recordingId)}/predictions?from=${encodeURIComponent(fromIso)}&to=${encodeURIComponent(toIso)}`,
      schema: z.array(backendPredictionResponseSchema),
    }),
  getHome: async () => {
    try {
      const summary = await sendRequest({
        method: "GET",
        path: "/sleep/latest/summary",
        schema: backendSleepSummarySchema,
      });
      return buildHomeFromSummary(summary);
    } catch (error) {
      if (isNoDataError(error)) {
        return buildEmptyHome();
      }
      throw error;
    }
  },
  getTrends: async (filter: TrendsFilter = "30D") => {
    const parsedFilter = trendsFilterSchema.parse(filter);
    try {
      return await sendRequest<TrendsResponse>({
        method: "GET",
        path: `/trends?filter=${encodeURIComponent(parsedFilter)}`,
        schema: trendsSchema,
      });
    } catch (error) {
      if (!isNoDataError(error)) {
        throw error;
      }

      try {
        const summary = await sendRequest({
          method: "GET",
          path: "/sleep/latest/summary",
          schema: backendSleepSummarySchema,
        });
        return buildTrendsFromSummary(summary, parsedFilter);
      } catch (summaryError) {
        if (isNoDataError(summaryError)) {
          return buildEmptyTrends(parsedFilter);
        }
        throw summaryError;
      }
    }
  },
  getNightsList: async () => {
    try {
      return await sendRequest<NightsListResponse>({ method: "GET", path: "/nights", schema: nightsListSchema });
    } catch (error) {
      if (!isNoDataError(error)) {
        throw error;
      }

      try {
        const summary = await sendRequest({
          method: "GET",
          path: "/sleep/latest/summary",
          schema: backendSleepSummarySchema,
        });
        return buildNightsListFromSummary(summary);
      } catch (summaryError) {
        if (isNoDataError(summaryError)) {
          return nightsListSchema.parse({ nights: [] });
        }
        throw summaryError;
      }
    }
  },
  getNight: async () => {
    try {
      return await sendRequest<NightResponse>({ method: "GET", path: "/night", schema: nightSchema });
    } catch (error) {
      if (!isNoDataError(error)) {
        throw error;
      }

      try {
        const summary = await sendRequest({
          method: "GET",
          path: "/sleep/latest/summary",
          schema: backendSleepSummarySchema,
        });
        return buildNightFromSummary(summary);
      } catch (summaryError) {
        if (isNoDataError(summaryError)) {
          return buildEmptyNight();
        }
        throw summaryError;
      }
    }
  },
  getSettingsProfile: () =>
    sendRequest({
      method: "GET",
      path: "/auth/me",
        schema: backendAuthMeResponseSchema,
    }).then((profile) =>
      settingsProfileSchemaEndpoint.parse({
        profile: {
          id: profile.id,
          username: deriveUsername(profile.username, profile.email),
          email: profile.email,
          createdAt: profile.created_at,
          updatedAt: profile.updated_at,
        },
      }),
    ),
  getSettingsDevice: () =>
    sendRequest({
      method: "GET",
      path: "/devices",
      schema: z.array(backendDeviceResponseSchema),
    }).then((devices) => {
      const currentDevice = devices[0];
      if (!currentDevice) {
        throw new ApiError("No connected device found.", "client", 404);
      }

      return settingsDeviceSchemaEndpoint.parse({
        device: {
          id: currentDevice.id,
          name: currentDevice.name,
          externalId: currentDevice.external_id ?? null,
          userId: currentDevice.user_id,
          createdAt: currentDevice.created_at,
        },
      });
    }),
  getSettings: async (): Promise<SettingsResponse> => {
    const [profile, device] = await Promise.all([httpApiClient.getSettingsProfile(), httpApiClient.getSettingsDevice()]);
    return { ...profile, ...device };
  },
  requestDataExport: async (): Promise<DataExportResponse> => {
    const report = await sendRequest({
      method: "GET",
      path: "/report/latest",
      schema: backendSleepSummarySchema,
    });

    return dataExportResponseSchema.parse({
      success: true,
      message: `Data export prepared for recording ${report.recordingId}.`,
      fileName: `noctis-report-${report.dateLocal}.json`,
      report,
    });
  },
  connectDevice: async (payload: ConnectDeviceRequest) => {
    const parsedPayload = connectDeviceRequestSchema.parse(payload);
    let device: BackendDeviceResponse;

    try {
      device = await sendRequest({
        method: "POST",
        path: "/devices/claim/by-id",
        body: {
          device_external_id: parsedPayload.deviceExternalId,
        },
        schema: backendDeviceResponseSchema,
      });
    } catch (error) {
      if (error instanceof ApiError && error.status === 404) {
        throw new ApiError("Device not found. Check the external ID and try again.", "client", 404);
      }
      throw error;
    }

    return actionResponseSchema.parse({
      success: true,
      message: `Mountable device ${device.name} connected.`,
    });
  },
  replaceDevice: async (payload: DevicePairingStartRequest) => {
    const parsedPayload = devicePairingStartRequestSchema.parse(payload);
    const pairing = await sendRequest({
      method: "POST",
      path: "/devices/pairing/start",
      body: {
        device_id: parsedPayload.deviceId,
        device_external_id: parsedPayload.deviceExternalId,
      },
      schema: backendDevicePairingStartResponseSchema,
    });

    return actionResponseSchema.parse({
      success: true,
      message: `Replacement pairing started. Code ${pairing.pairing_code}.`,
    });
  },
  logout: () => actionResponseSchema.parse({ success: true, message: "Action completed" }),
  updatePreferences: () =>
    sendRequest<ActionResponse>({ method: "POST", path: "/settings/preferences", schema: actionResponseSchema }),
  login: (payload: LoginRequest) => {
    const body = loginRequestSchema.parse(payload);
    return sendRequest<AuthResponse>({ method: "POST", path: "/auth/login", body, schema: authResponseSchema, auth: false });
  },
  signup: (payload: SignupRequest) => {
    const body = signupRequestSchema.parse(payload);
    return sendRequest<AuthResponse>({ method: "POST", path: "/auth/register", body, schema: authResponseSchema, auth: false });
  },
};
