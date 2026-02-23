import { clamp } from "../../../../core/utils/clamp";
import { StageBin, SleepStage } from "../../api/sleep-summary.types";

export const mapXToMinute = (
  x: number,
  width: number,
  totalMinutes: number,
): number => {
  if (width <= 0 || totalMinutes <= 0) {
    return 0;
  }

  const ratio = clamp(x / width, 0, 1);
  const minute = Math.floor(ratio * totalMinutes);
  return Math.max(0, Math.min(totalMinutes - 1, minute));
};

export const findStageAtMinute = (
  bins: StageBin[],
  minute: number,
): SleepStage | null => {
  const match = bins.find(
    (bin) => minute >= bin.startMinFromBedtime &&
      minute < bin.startMinFromBedtime + bin.durationMin,
  );

  return match?.stage ?? null;
};
