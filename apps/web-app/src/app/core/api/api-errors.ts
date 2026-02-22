import { HttpErrorResponse } from "@angular/common/http";

export type ApiError = {
  message: string;
  status?: number;
  code?: string;
  classification?: string;
  failureClassification?: string;
  requestId?: string;
  details?: unknown;
};

export type BackendErrorPayload = {
  code: string;
  message: string;
  classification: string;
  failure_classification: string;
  request_id?: string;
  extra?: unknown;
};

export type BackendErrorResponse = {
  error: BackendErrorPayload;
};

const isBackendErrorResponse = (value: unknown): value is BackendErrorResponse => {
  if (!value || typeof value !== "object") {
    return false;
  }

  const envelope = value as { error?: unknown };
  if (!envelope.error || typeof envelope.error !== "object") {
    return false;
  }

  const payload = envelope.error as {
    code?: unknown;
    message?: unknown;
    classification?: unknown;
    failure_classification?: unknown;
  };

  return (
    typeof payload.code === "string" &&
    typeof payload.message === "string" &&
    typeof payload.classification === "string" &&
    typeof payload.failure_classification === "string"
  );
};

export const toApiError = (error: unknown): ApiError => {
  if (error instanceof HttpErrorResponse) {
    if (isBackendErrorResponse(error.error)) {
      return {
        message: error.error.error.message,
        status: error.status,
        code: error.error.error.code,
        classification: error.error.error.classification,
        failureClassification: error.error.error.failure_classification,
        requestId: error.error.error.request_id,
        details: error.error.error.extra,
      };
    }

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
