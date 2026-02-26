import { beforeEach, describe, expect, it } from "vitest";
import {
  clearOnboardingState,
  nextOnboardingRoute,
  readOnboardingState,
  reconcileOnboardingState,
  resolveOnboardingRedirect,
  persistOnboardingState,
  type OnboardingState,
} from "../src/platform/onboarding";

function createMemoryStorage(): Storage {
  const data = new Map<string, string>();
  return {
    get length() {
      return data.size;
    },
    clear() {
      data.clear();
    },
    getItem(key: string) {
      return data.has(key) ? data.get(key) ?? null : null;
    },
    key(index: number) {
      return [...data.keys()][index] ?? null;
    },
    removeItem(key: string) {
      data.delete(key);
    },
    setItem(key: string, value: string) {
      data.set(key, value);
    },
  };
}

function blankOnboardingState(): OnboardingState {
  return {
    hasConnectedDevice: false,
    trackingActive: false,
    lastOnboardingStepCompleted: null,
  };
}

describe("onboarding routing and guards", () => {
  beforeEach(() => {
    Object.defineProperty(window, "localStorage", {
      value: createMemoryStorage(),
      configurable: true,
    });
    Object.defineProperty(window, "sessionStorage", {
      value: createMemoryStorage(),
      configurable: true,
    });
  });

  it("supports register->login->connect->start progression", () => {
    const signedIn = reconcileOnboardingState(blankOnboardingState(), {
      isAuthenticated: true,
      pendingLoginEmail: "new-user@noctis.local",
    });
    expect(signedIn.lastOnboardingStepCompleted).toBe("login");
    expect(nextOnboardingRoute(signedIn)).toBe("/onboarding/connect-device");

    const connected = reconcileOnboardingState(
      {
        ...signedIn,
        hasConnectedDevice: true,
        lastOnboardingStepCompleted: "connect-device",
      },
      { isAuthenticated: true },
    );
    expect(connected.hasConnectedDevice).toBe(true);
    expect(nextOnboardingRoute(connected)).toBe("/onboarding/start-tracking");

    const tracking = reconcileOnboardingState(
      {
        ...connected,
        trackingActive: true,
        lastOnboardingStepCompleted: "start-tracking",
      },
      { isAuthenticated: true },
    );
    expect(tracking.trackingActive).toBe(true);
    expect(nextOnboardingRoute(tracking)).toBe("/app/home");
  });

  it("redirects deep link start-tracking without device to connect step", () => {
    const redirect = resolveOnboardingRedirect("/onboarding/start-tracking", blankOnboardingState());
    expect(redirect?.path).toBe("/onboarding/connect-device");
  });

  it("redirects deep link app route while onboarding is incomplete", () => {
    const noDevice = resolveOnboardingRedirect("/app/sleep/latest", blankOnboardingState());
    expect(noDevice?.path).toBe("/onboarding/connect-device");

    const deviceConnected = resolveOnboardingRedirect("/app/sleep/latest", {
      hasConnectedDevice: true,
      trackingActive: false,
      lastOnboardingStepCompleted: "connect-device",
    });
    expect(deviceConnected?.path).toBe("/onboarding/start-tracking");
  });

  it("allows core app routes when onboarding is completed", () => {
    const redirect = resolveOnboardingRedirect("/app/sleep/latest", {
      hasConnectedDevice: true,
      trackingActive: true,
      lastOnboardingStepCompleted: "start-tracking",
    });
    expect(redirect).toBeUndefined();
  });

  it("reconciles cached onboarding state with live session signals", () => {
    const cached: OnboardingState = {
      hasConnectedDevice: true,
      trackingActive: true,
      lastOnboardingStepCompleted: "start-tracking",
      pendingLoginEmail: "demo@noctis.local",
    };

    const reconciled = reconcileOnboardingState(cached, {
      isAuthenticated: true,
      hasConnectedDevice: false,
      trackingActive: false,
    });

    expect(reconciled.hasConnectedDevice).toBe(false);
    expect(reconciled.trackingActive).toBe(false);
    expect(reconciled.lastOnboardingStepCompleted).toBe("login");
  });

  it("clears onboarding and tracking local state on logout", () => {
    persistOnboardingState({
      hasConnectedDevice: true,
      trackingActive: true,
      lastOnboardingStepCompleted: "start-tracking",
    });

    expect(window.localStorage.getItem("noctis_onboarding_state")).not.toBeNull();
    expect(window.localStorage.getItem("noctis_tracking_active")).toBe("1");

    clearOnboardingState();

    expect(window.localStorage.getItem("noctis_onboarding_state")).toBeNull();
    expect(window.localStorage.getItem("noctis_tracking_active")).toBeNull();
    expect(readOnboardingState()).toEqual(blankOnboardingState());
  });
});
