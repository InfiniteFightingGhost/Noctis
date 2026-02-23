import { DataQualityStatus, PrimaryAction } from "../api/sleep-summary.types";

export type SleepSummaryViewState =
  | "loading"
  | "success"
  | "missing"
  | "syncing"
  | "error";

export const mapQualityToViewState = (
  status?: DataQualityStatus,
): SleepSummaryViewState => {
  if (!status || status === "ok" || status === "partial") {
    return "success";
  }

  return status;
};

export const resolvePrimaryActionLabel = (
  status?: DataQualityStatus,
  action?: PrimaryAction | null,
): string => {
  if (status && status !== "ok" && status !== "partial") {
    return "Fix Tracking";
  }

  return action?.label ?? "Improve Tonight";
};
