import { HttpErrorResponse } from "@angular/common/http";

export type ApiError = {
  message: string;
  status?: number;
  details?: unknown;
};

export const toApiError = (error: unknown): ApiError => {
  if (error instanceof HttpErrorResponse) {
    return {
      message: error.error?.message ?? error.message,
      status: error.status,
      details: error.error,
    };
  }

  if (error instanceof Error) {
    return { message: error.message };
  }

  return { message: "Unexpected error" };
};
