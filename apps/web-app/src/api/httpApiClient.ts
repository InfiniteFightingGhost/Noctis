import { z, type ZodSchema } from "zod";
import {
  actionResponseSchema,
  authResponseSchema,
  backendAuthMeResponseSchema,
  backendDeviceResponseSchema,
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

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL ?? "").replace(/\/$/, "");
const API_PREFIX = (import.meta.env.VITE_API_V1_PREFIX ?? "/v1").replace(/\/$/, "");
const REFRESH_PATH = "/auth/refresh";

let refreshInFlight: Promise<boolean> | null = null;

function getErrorMessage(status: number): string {
  if (status >= 500) {
    return "Server error. Please try again.";
  }
  if (status >= 400) {
    return "Request could not be completed.";
  }
  return "Unexpected API response.";
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
        credentials: "include",
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
        credentials: "include",
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
        const message =
          payload && typeof payload === "object" && "message" in payload && typeof payload.message === "string"
            ? payload.message
            : fallbackMessage;
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
  getHome: () => sendRequest<HomeResponse>({ method: "GET", path: "/home", schema: homeSchema }),
  getTrends: (filter: TrendsFilter = "30D") => {
    const parsedFilter = trendsFilterSchema.parse(filter);
    return sendRequest<TrendsResponse>({
      method: "GET",
      path: `/trends?filter=${encodeURIComponent(parsedFilter)}`,
      schema: trendsSchema,
    });
  },
  getNightsList: () => sendRequest<NightsListResponse>({ method: "GET", path: "/nights", schema: nightsListSchema }),
  getNight: () => sendRequest<NightResponse>({ method: "GET", path: "/night", schema: nightSchema }),
  getSettingsProfile: () =>
    sendRequest({
      method: "GET",
      path: "/auth/me",
        schema: backendAuthMeResponseSchema,
    }).then((profile) =>
      settingsProfileSchemaEndpoint.parse({
        profile: {
          id: profile.id,
          username: profile.username,
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
