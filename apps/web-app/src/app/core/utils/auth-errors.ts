import { HttpErrorResponse } from "@angular/common/http";

export const getAuthErrorMessage = (error: unknown): string => {
  if (error instanceof HttpErrorResponse) {
    if (error.status === 0) {
      return "Network error. Check your connection and try again.";
    }

    if (error.status === 401) {
      return "Authentication failed. Please verify your credentials and try again.";
    }

    if (typeof error.error === "string" && error.error.trim().length > 0) {
      return error.error;
    }

    if (typeof error.error?.detail === "string") {
      return error.error.detail;
    }

    if (typeof error.error?.message === "string") {
      return error.error.message;
    }

    return "Authentication failed. Please verify your credentials and try again.";
  }

  if (error instanceof Error && error.message.trim().length > 0) {
    return error.message;
  }

  return "Authentication failed. Please try again.";
};
