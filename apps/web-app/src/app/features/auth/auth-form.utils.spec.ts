import { HttpErrorResponse } from "@angular/common/http";
import { getAuthErrorMessage } from "../../core/utils/auth-errors";
import {
  doPasswordsMatch,
  getPostAuthRedirect,
  isEmailValid,
  isPasswordValid,
} from "./auth-form.utils";

describe("auth form validation", () => {
  it("validates email and password format", () => {
    expect(isEmailValid("invalid-email")).toBe(false);
    expect(isEmailValid("user@example.com")).toBe(true);
    expect(isPasswordValid("short")).toBe(false);
    expect(isPasswordValid("password123")).toBe(true);
  });

  it("validates signup password confirmation", () => {
    expect(doPasswordsMatch("password123", "different")).toBe(false);
    expect(doPasswordsMatch("password123", "password123")).toBe(true);
  });
});

describe("auth UX helpers", () => {
  it("renders network/backend errors as user-friendly text", () => {
    expect(
      getAuthErrorMessage(new HttpErrorResponse({ status: 0, statusText: "Unknown Error" })),
    ).toBe("Network error. Check your connection and try again.");

    expect(
      getAuthErrorMessage(
        new HttpErrorResponse({ status: 400, error: { detail: "Email already exists" } }),
      ),
    ).toBe("Email already exists");
  });

  it("resolves safe post-auth redirects for login/signup success", () => {
    expect(getPostAuthRedirect("/report")).toBe("/report");
    expect(getPostAuthRedirect("/login")).toBe("/dashboard");
    expect(getPostAuthRedirect("https://evil.example")).toBe("/dashboard");
    expect(getPostAuthRedirect(undefined)).toBe("/dashboard");
  });
});
