import { describe, expect, it } from "vitest";
import {
  clearAuthSession,
  getAccessToken,
  isDeviceConnected,
  isLoggedIn,
  setAuthSession,
  setDeviceConnected,
} from "./session";

describe("session state", () => {
  it("defaults to logged out and disconnected", () => {
    clearAuthSession();
    expect(isLoggedIn()).toBe(false);
    expect(isDeviceConnected()).toBe(false);
    expect(getAccessToken()).toBeNull();
  });

  it("stores and clears auth session tokens", () => {
    setAuthSession({ accessToken: "token-1", refreshToken: "refresh-1", userId: "usr_1" });
    setDeviceConnected(true);
    expect(isLoggedIn()).toBe(true);
    expect(getAccessToken()).toBe("token-1");
    expect(isDeviceConnected()).toBe(true);

    clearAuthSession();
    expect(isLoggedIn()).toBe(false);
    expect(getAccessToken()).toBeNull();
    expect(isDeviceConnected()).toBe(false);
  });
});
