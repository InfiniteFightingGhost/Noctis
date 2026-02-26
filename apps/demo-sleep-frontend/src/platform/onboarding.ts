export type OnboardingStepId = "signup" | "login" | "connect-device" | "start-tracking";

export type OnboardingBannerTone = "info" | "success" | "warning";

export type OnboardingBanner = {
  tone: OnboardingBannerTone;
  message: string;
};

export type OnboardingState = {
  hasConnectedDevice: boolean;
  trackingActive: boolean;
  lastOnboardingStepCompleted: OnboardingStepId | null;
  pendingLoginEmail?: string;
  banner?: OnboardingBanner;
};

export type OnboardingRedirect = {
  path: string;
  message: string;
};

export type OnboardingLiveSignals = {
  isAuthenticated: boolean;
  hasConnectedDevice?: boolean;
  trackingActive?: boolean;
  pendingLoginEmail?: string;
};

const ONBOARDING_STORAGE_KEY = "noctis_onboarding_state";
const TRACKING_ACTIVE_STORAGE_KEY = "noctis_tracking_active";

export const TRACKING_REQUIRED_ROUTE_PATTERNS = new Set([
  "/app/home",
  "/app/sleep/latest",
  "/app/sleep/sync",
  "/app/sleep/insights",
]);

const DEFAULT_ONBOARDING_STATE: OnboardingState = {
  hasConnectedDevice: false,
  trackingActive: false,
  lastOnboardingStepCompleted: null,
};

function normalizeStep(value: unknown): OnboardingStepId | null {
  if (value === "signup" || value === "login" || value === "connect-device" || value === "start-tracking") {
    return value;
  }
  if (value === "connect") {
    return "connect-device";
  }
  if (value === "tracking") {
    return "start-tracking";
  }
  return null;
}

function normalizeBanner(value: unknown): OnboardingBanner | undefined {
  if (typeof value !== "object" || value === null) {
    return undefined;
  }

  const raw = value as { tone?: unknown; message?: unknown };
  if (typeof raw.message !== "string" || raw.message.trim().length === 0) {
    return undefined;
  }

  const tone = raw.tone;
  if (tone !== "info" && tone !== "success" && tone !== "warning") {
    return { tone: "info", message: raw.message };
  }

  return { tone, message: raw.message };
}

function normalizeState(value: unknown): OnboardingState {
  if (typeof value !== "object" || value === null) {
    return { ...DEFAULT_ONBOARDING_STATE };
  }

  const raw = value as Partial<OnboardingState> & {
    deviceConnected?: unknown;
    signupComplete?: unknown;
    loginComplete?: unknown;
  };

  const hasConnectedDevice =
    raw.hasConnectedDevice === true ||
    raw.deviceConnected === true ||
    raw.lastOnboardingStepCompleted === "connect-device" ||
    raw.lastOnboardingStepCompleted === "start-tracking";

  const trackingActive =
    raw.trackingActive === true || raw.lastOnboardingStepCompleted === "start-tracking";

  const lastOnboardingStepCompleted =
    normalizeStep(raw.lastOnboardingStepCompleted) ??
    (trackingActive ? "start-tracking" : hasConnectedDevice ? "connect-device" : raw.loginComplete === true ? "login" : raw.signupComplete === true ? "signup" : null);

  const pendingLoginEmail =
    typeof raw.pendingLoginEmail === "string" && raw.pendingLoginEmail.trim().length > 0 ? raw.pendingLoginEmail : undefined;

  return {
    hasConnectedDevice,
    trackingActive,
    lastOnboardingStepCompleted,
    pendingLoginEmail,
    banner: normalizeBanner(raw.banner),
  };
}

export function readOnboardingState(): OnboardingState {
  if (typeof window === "undefined") {
    return { ...DEFAULT_ONBOARDING_STATE };
  }

  try {
    const sessionRaw = window.sessionStorage.getItem(ONBOARDING_STORAGE_KEY);
    if (sessionRaw) {
      return normalizeState(JSON.parse(sessionRaw));
    }

    const localRaw = window.localStorage.getItem(ONBOARDING_STORAGE_KEY);
    if (localRaw) {
      return normalizeState(JSON.parse(localRaw));
    }

    const localTracking = window.localStorage.getItem(TRACKING_ACTIVE_STORAGE_KEY);
    if (localTracking === "1") {
      return normalizeState({
        hasConnectedDevice: true,
        trackingActive: true,
        lastOnboardingStepCompleted: "start-tracking",
      });
    }

    return { ...DEFAULT_ONBOARDING_STATE };
  } catch {
    return { ...DEFAULT_ONBOARDING_STATE };
  }
}

export function persistOnboardingState(state: OnboardingState): void {
  if (typeof window === "undefined") {
    return;
  }

  const normalized = normalizeState(state);
  const payload = JSON.stringify(normalized);
  window.sessionStorage.setItem(ONBOARDING_STORAGE_KEY, payload);
  window.localStorage.setItem(ONBOARDING_STORAGE_KEY, payload);
  if (normalized.trackingActive) {
    window.localStorage.setItem(TRACKING_ACTIVE_STORAGE_KEY, "1");
  } else {
    window.localStorage.removeItem(TRACKING_ACTIVE_STORAGE_KEY);
  }
}

export function clearOnboardingState(): void {
  if (typeof window === "undefined") {
    return;
  }

  window.sessionStorage.removeItem(ONBOARDING_STORAGE_KEY);
  window.localStorage.removeItem(ONBOARDING_STORAGE_KEY);
  window.localStorage.removeItem(TRACKING_ACTIVE_STORAGE_KEY);
}

export function reconcileOnboardingState(cached: OnboardingState, live: OnboardingLiveSignals): OnboardingState {
  if (!live.isAuthenticated) {
    return { ...DEFAULT_ONBOARDING_STATE };
  }

  const normalizedCached = normalizeState(cached);
  const hasConnectedDevice = live.hasConnectedDevice ?? normalizedCached.hasConnectedDevice;
  const trackingFromSignal = live.trackingActive ?? normalizedCached.trackingActive;
  const trackingActive = trackingFromSignal;
  const normalizedDeviceConnected = hasConnectedDevice || trackingActive;

  const lastOnboardingStepCompleted =
    trackingActive
      ? "start-tracking"
      : normalizedDeviceConnected
        ? "connect-device"
        : "login";

  const pendingLoginEmail =
    typeof live.pendingLoginEmail === "string" && live.pendingLoginEmail.trim().length > 0
      ? live.pendingLoginEmail.trim()
      : normalizedCached.pendingLoginEmail;

  return normalizeState({
    ...normalizedCached,
    hasConnectedDevice: normalizedDeviceConnected,
    trackingActive,
    lastOnboardingStepCompleted,
    pendingLoginEmail,
  });
}

export function isOnboardingIncomplete(state: OnboardingState): boolean {
  return !state.hasConnectedDevice || !state.trackingActive;
}

export function currentOnboardingStep(state: OnboardingState): OnboardingStepId {
  if (!state.hasConnectedDevice) {
    return "connect-device";
  }
  return "start-tracking";
}

export function nextOnboardingRoute(state: OnboardingState): string {
  if (!state.hasConnectedDevice) {
    return "/onboarding/connect-device";
  }
  if (!state.trackingActive) {
    return "/onboarding/start-tracking";
  }
  return "/app/home";
}

export function resolveOnboardingRedirect(routePattern: string, state: OnboardingState): OnboardingRedirect | undefined {
  const isUserRoute = routePattern.startsWith("/app/") || routePattern.startsWith("/onboarding/");
  if (!isUserRoute) {
    return undefined;
  }

  if (!state.hasConnectedDevice && routePattern !== "/onboarding/connect-device") {
    return {
      path: "/onboarding/connect-device",
      message: "Connect your sleep device to continue setup.",
    };
  }

  if (state.hasConnectedDevice && !state.trackingActive && routePattern === "/onboarding/connect-device") {
    return {
      path: "/onboarding/start-tracking",
      message: "Your device is ready. Start tracking to unlock your sleep dashboard.",
    };
  }

  if (state.hasConnectedDevice && !state.trackingActive && TRACKING_REQUIRED_ROUTE_PATTERNS.has(routePattern)) {
    return {
      path: "/onboarding/start-tracking",
      message: "Start tracking to unlock your sleep dashboard.",
    };
  }

  return undefined;
}
