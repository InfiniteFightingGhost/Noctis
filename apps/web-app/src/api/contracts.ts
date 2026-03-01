import { z } from "zod";

const stageSchema = z.enum(["wake", "light", "deep", "rem"]);

const stagePercentageSchema = z.object({
  stage: stageSchema,
  minutes: z.number().min(0),
  percent: z.number().min(0).max(100),
});

const trendNightSchema = z.object({
  date: z.string(),
  sleepScore: z.number(),
  totalSleepMinutes: z.number(),
  sleepEfficiency: z.number(),
  remPercent: z.number(),
  deepPercent: z.number(),
  fragmentationIndex: z.number(),
  hrMean: z.number(),
  hrv: z.number(),
  consistencyIndex: z.number(),
});

const settingsProfileSchema = z.object({
  profile: z.object({
    id: z.string(),
    username: z.string().min(1),
    email: z.string(),
    createdAt: z.string(),
    updatedAt: z.string(),
  }),
});

const settingsDeviceSchema = z.object({
  device: z.object({
    id: z.string(),
    name: z.string(),
    externalId: z.string().nullable(),
    userId: z.string().nullable(),
    createdAt: z.string(),
  }),
});

const epochSchema = z.object({
  epochIndex: z.number().int().min(0),
  stage: stageSchema,
  confidence: z.number().min(0).max(1),
  probabilities: z.object({
    wake: z.number().min(0).max(1),
    light: z.number().min(0).max(1),
    deep: z.number().min(0).max(1),
    rem: z.number().min(0).max(1),
  }),
});

export const homeSchema = z.object({
  date: z.string(),
  metrics: z.object({
    sleepScore: z.number(),
    totalSleepMinutes: z.number(),
    sleepEfficiency: z.number(),
    remPercent: z.number(),
    deepPercent: z.number(),
    deltaVs7DayBaseline: z.object({
      sleepScore: z.number(),
      totalSleepMinutes: z.number(),
      sleepEfficiency: z.number(),
      remPercent: z.number(),
      deepPercent: z.number(),
    }),
  }),
  summaryHypnogram: z.object({
    confidenceSummary: z.string(),
    epochs: z.array(epochSchema),
  }),
  stageBreakdown: z.array(stagePercentageSchema),
  latencyTriplet: z.object({
    sleepOnsetMinutes: z.number(),
    remLatencyMinutes: z.number(),
    deepLatencyMinutes: z.number(),
  }),
  continuityMetrics: z.object({
    fragmentationIndex: z.number(),
    entropy: z.number(),
    wasoMinutes: z.number(),
  }),
  aiSummary: z.string(),
});

export const trendsSchema = z.object({
  activeFilter: z.enum(["7D", "30D", "90D", "Custom"]),
  nights: z.array(trendNightSchema),
  movingAverageWindow: z.number(),
  varianceBand: z.object({
    lower: z.number(),
    upper: z.number(),
  }),
  worstNightDecile: z.object({
    date: z.string(),
    sleepScore: z.number(),
  }),
});

export const trendsFilterSchema = trendsSchema.shape.activeFilter;

export const nightsListSchema = z.object({
  nights: z.array(
    z.object({
      nightId: z.string(),
      date: z.string(),
      label: z.string(),
      hasCapData: z.boolean(),
    }),
  ),
});

export const nightSchema = z.object({
  date: z.string(),
  epochs: z.array(epochSchema),
  transitions: z.array(z.array(z.number())),
  arousalIndex: z.number(),
  capRateConditional: z
    .object({
      available: z.literal(true),
      value: z.number(),
    })
    .or(
      z.object({
        available: z.literal(false),
        reason: z.string(),
      }),
    ),
  cardiopulmonary: z.object({
    avgRespiratoryRate: z.number(),
    minSpO2: z.number(),
    avgHeartRate: z.number(),
  }),
});

export const settingsSchema = settingsProfileSchema.merge(settingsDeviceSchema);

export const settingsProfileSchemaEndpoint = settingsProfileSchema;
export const settingsDeviceSchemaEndpoint = settingsDeviceSchema;

export const actionResponseSchema = z.object({
  success: z.boolean(),
  message: z.string(),
});

export const connectDeviceRequestSchema = z.object({
  deviceExternalId: z.string().trim().min(1),
});

export const backendDeviceResponseSchema = z.object({
  id: z.string(),
  name: z.string(),
  external_id: z.string().nullable().optional(),
  user_id: z.string().nullable(),
  created_at: z.string(),
});

export const devicePairingStartRequestSchema = z
  .object({
    deviceId: z.string().optional(),
    deviceExternalId: z.string().optional(),
  })
  .refine((value) => Boolean(value.deviceId || value.deviceExternalId), {
    message: "deviceId or deviceExternalId is required",
  });

export const backendDevicePairingStartResponseSchema = z.object({
  pairing_session_id: z.string(),
  pairing_code: z.string(),
  expires_at: z.string(),
});

export const backendSleepSummarySchema = z.object({
  recordingId: z.string(),
  dateLocal: z.string(),
  score: z.number(),
  totals: z.object({
    totalSleepMin: z.number(),
    timeInBedMin: z.number(),
    sleepEfficiencyPct: z.number(),
  }),
  metrics: z.object({
    deepPct: z.number(),
    avgHrBpm: z.number(),
    avgRrBrpm: z.number(),
    movementPct: z.number(),
  }),
  stages: z
    .object({
      bins: z
        .array(
          z.object({
            startMinFromBedtime: z.number(),
            durationMin: z.number(),
            stage: z.string(),
          }),
        )
        .optional(),
      pct: z
        .object({
          awake: z.number(),
          light: z.number(),
          deep: z.number(),
          rem: z.number(),
        })
        .optional(),
    })
    .optional(),
  insight: z
    .object({
      text: z.string(),
    })
    .optional(),
});

export const dataExportResponseSchema = z.object({
  success: z.boolean(),
  message: z.string(),
  fileName: z.string(),
  report: backendSleepSummarySchema,
});

export const backendAuthMeResponseSchema = z.object({
  id: z.string(),
  username: z.string().min(1).optional(),
  email: z.string(),
  created_at: z.string(),
  updated_at: z.string(),
});

export const loginRequestSchema = z.object({
  email: z.string().email(),
  password: z.string().min(8),
});

export const signupRequestSchema = z.object({
  username: z.string().min(3).max(64),
  email: z.string().email(),
  password: z.string().min(8),
});

export const authResponseSchema = z.object({
  access_token: z.string().min(1),
  token_type: z.string(),
  expires_in: z.number(),
  user: z.object({
    id: z.string(),
    username: z.string().min(1).optional(),
    email: z.string(),
    created_at: z.string(),
    updated_at: z.string(),
  }),
});

export const backendEpochResponseSchema = z.object({
  recording_id: z.string(),
  epoch_index: z.number().int().min(0),
  epoch_start_ts: z.string(),
  feature_schema_version: z.string(),
  features_payload: z.record(z.unknown()),
});

export const backendPredictionResponseSchema = z.object({
  id: z.string(),
  recording_id: z.string(),
  window_start_ts: z.string(),
  window_end_ts: z.string(),
  model_version: z.string(),
  feature_schema_version: z.string(),
  dataset_snapshot_id: z.string().nullable(),
  predicted_stage: z.string(),
  ground_truth_stage: z.string().nullable(),
  confidence: z.number(),
  probabilities: z.record(z.number()),
  created_at: z.string(),
});

export type HomeResponse = z.infer<typeof homeSchema>;
export type TrendsResponse = z.infer<typeof trendsSchema>;
export type TrendsFilter = z.infer<typeof trendsFilterSchema>;
export type NightsListResponse = z.infer<typeof nightsListSchema>;
export type NightResponse = z.infer<typeof nightSchema>;
export type SettingsResponse = z.infer<typeof settingsSchema>;
export type SettingsProfileResponse = z.infer<typeof settingsProfileSchema>;
export type SettingsDeviceResponse = z.infer<typeof settingsDeviceSchema>;
export type ActionResponse = z.infer<typeof actionResponseSchema>;
export type ConnectDeviceRequest = z.infer<typeof connectDeviceRequestSchema>;
export type BackendDeviceResponse = z.infer<typeof backendDeviceResponseSchema>;
export type DevicePairingStartRequest = z.infer<typeof devicePairingStartRequestSchema>;
export type DataExportResponse = z.infer<typeof dataExportResponseSchema>;
export type LoginRequest = z.infer<typeof loginRequestSchema>;
export type SignupRequest = z.infer<typeof signupRequestSchema>;
export type AuthResponse = z.infer<typeof authResponseSchema>;
export type Stage = z.infer<typeof stageSchema>;
export type BackendSleepSummary = z.infer<typeof backendSleepSummarySchema>;
export type BackendEpochResponse = z.infer<typeof backendEpochResponseSchema>;
export type BackendPredictionResponse = z.infer<typeof backendPredictionResponseSchema>;
