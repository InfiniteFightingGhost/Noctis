import { SleepStage } from "../../api/sleep-summary.types";

export const STAGE_COLORS: Record<SleepStage, string> = {
  awake: "var(--stage-awake)",
  light: "var(--stage-light)",
  deep: "var(--stage-deep)",
  rem: "var(--stage-rem)",
};

export const stageLabel = (stage: SleepStage): string => {
  switch (stage) {
    case "awake":
      return "Awake";
    case "light":
      return "Light";
    case "deep":
      return "Deep";
    case "rem":
      return "REM";
  }
};
