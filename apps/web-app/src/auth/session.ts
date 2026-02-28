type SessionState = {
  accessToken: string | null;
  refreshToken: string | null;
  userId: string | null;
  deviceConnected: boolean;
};

const DEVICE_CONNECTED_STORAGE_KEY = "noctis-device-connected-by-user";

const sessionState: SessionState = {
  accessToken: null,
  refreshToken: null,
  userId: null,
  deviceConnected: false,
};

function readDeviceConnectionMap(): Record<string, boolean> {
  if (typeof window === "undefined") {
    return {};
  }

  try {
    const raw = window.localStorage.getItem(DEVICE_CONNECTED_STORAGE_KEY);
    if (!raw) {
      return {};
    }

    const parsed = JSON.parse(raw);
    if (!parsed || typeof parsed !== "object") {
      return {};
    }

    const map: Record<string, boolean> = {};
    for (const [key, value] of Object.entries(parsed)) {
      map[key] = value === true;
    }
    return map;
  } catch {
    return {};
  }
}

function writeDeviceConnectionMap(map: Record<string, boolean>): void {
  if (typeof window === "undefined") {
    return;
  }

  try {
    window.localStorage.setItem(DEVICE_CONNECTED_STORAGE_KEY, JSON.stringify(map));
  } catch {
    // Ignore storage errors in restricted contexts.
  }
}

function getStoredDeviceConnected(userId: string | null): boolean {
  if (!userId) {
    return false;
  }

  const map = readDeviceConnectionMap();
  return map[userId] === true;
}

type AuthSessionPayload = {
  accessToken: string;
  refreshToken?: string | null;
  userId?: string | null;
};

export function setAuthSession(payload: AuthSessionPayload): void {
  sessionState.accessToken = payload.accessToken;
  sessionState.refreshToken = payload.refreshToken ?? null;
  sessionState.userId = payload.userId ?? null;
  sessionState.deviceConnected = getStoredDeviceConnected(sessionState.userId);
}

export function clearAuthSession(): void {
  sessionState.accessToken = null;
  sessionState.refreshToken = null;
  sessionState.userId = null;
  sessionState.deviceConnected = false;
}

export function getAccessToken(): string | null {
  return sessionState.accessToken;
}

export function getRefreshToken(): string | null {
  return sessionState.refreshToken;
}

export function isLoggedIn(): boolean {
  return sessionState.accessToken !== null;
}

export function setLoggedIn(value: boolean): void {
  if (value) {
    setAuthSession({
      accessToken: "test-session-token",
      refreshToken: "test-refresh-token",
      userId: "usr_test",
    });
    return;
  }

  clearAuthSession();
}

export function isDeviceConnected(): boolean {
  return sessionState.deviceConnected;
}

export function setDeviceConnected(value: boolean): void {
  sessionState.deviceConnected = value;

  if (!sessionState.userId) {
    return;
  }

  const map = readDeviceConnectionMap();
  map[sessionState.userId] = value;
  writeDeviceConnectionMap(map);
}
