export type SleepStage = "awake" | "light" | "deep" | "rem";

export type DataQualityStatus =
  | "ok"
  | "partial"
  | "missing"
  | "syncing"
  | "error";

export type ScoreLabel = "Poor" | "Fair" | "Good" | "Excellent";

export type StageBin = {
  startMinFromBedtime: number;
  durationMin: number;
  stage: SleepStage;
};

export type StagePct = {
  awake: number;
  light: number;
  deep: number;
  rem: number;
};

export type SleepTotals = {
  totalSleepMin: number;
  timeInBedMin: number;
  sleepEfficiencyPct: number;
};

export type SleepMetrics = {
  deepPct: number;
  avgHrBpm: number;
  avgRrBrpm: number;
  movementPct: number;
};

export type SleepInsight = {
  text: string;
  tag?: string;
  confidence?: number;
};

export type PrimaryAction = {
  label: string;
  action: string;
  params?: Record<string, string | number | boolean>;
};

export type DataQuality = {
  status: DataQualityStatus;
  issues?: string[];
  lastSyncAtLocal?: string;
};

export type SleepSummary = {
  recordingId: string;
  dateLocal: string;
  bedtimeLocal: string;
  waketimeLocal: string;
  score: number;
  scoreLabel: ScoreLabel;
  totals: SleepTotals;
  stages: {
    bins: StageBin[];
    pct: StagePct;
  };
  metrics: SleepMetrics;
  insight: SleepInsight;
  primaryAction: PrimaryAction;
  dataQuality: DataQuality;
};
