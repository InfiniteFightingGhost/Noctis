import { inject, Injectable } from "@angular/core";
import { ApiClient } from "./http-client";

export type RecordingCreatePayload = {
  deviceId: string;
  startedAt: string;
  timezone?: string | null;
};

export type RecordingResponse = {
  id: string;
  device_id: string;
  started_at: string;
  ended_at: string | null;
  timezone: string | null;
  created_at: string;
};

export type EpochResponse = {
  recording_id: string;
  epoch_index: number;
  epoch_start_ts: string;
  feature_schema_version: string;
  features_payload: Record<string, unknown>;
};

export type PredictionResponse = {
  id: string;
  recording_id: string;
  window_start_ts: string;
  window_end_ts: string;
  model_version: string;
  feature_schema_version: string;
  dataset_snapshot_id: string | null;
  predicted_stage: string;
  ground_truth_stage: string | null;
  confidence: number;
  probabilities: Record<string, number>;
  created_at: string;
};

export type RecordingSummary = {
  recording_id: string;
  from_ts: string;
  to_ts: string;
  total_minutes: number;
  time_in_stage_minutes: Record<string, number>;
  sleep_latency_minutes: number | null;
  waso_minutes: number | null;
};

export type RecordingRangeParams = {
  from: string;
  to: string;
};

@Injectable({ providedIn: "root" })
export class RecordingsApi {
  private readonly api = inject(ApiClient);

  createRecording(payload: RecordingCreatePayload) {
    return this.api.post<RecordingResponse>("/v1/recordings", {
      device_id: payload.deviceId,
      started_at: payload.startedAt,
      timezone: payload.timezone ?? null,
    });
  }

  getRecording(recordingId: string) {
    return this.api.get<RecordingResponse>(`/v1/recordings/${recordingId}`);
  }

  getEpochs(recordingId: string, range: RecordingRangeParams) {
    return this.api.get<EpochResponse[]>(`/v1/recordings/${recordingId}/epochs`, {
      params: {
        from: range.from,
        to: range.to,
      },
    });
  }

  getPredictions(recordingId: string, range: RecordingRangeParams) {
    return this.api.get<PredictionResponse[]>(
      `/v1/recordings/${recordingId}/predictions`,
      {
        params: {
          from: range.from,
          to: range.to,
        },
      },
    );
  }

  getSummary(recordingId: string, range: RecordingRangeParams) {
    return this.api.get<RecordingSummary>(`/v1/recordings/${recordingId}/summary`, {
      params: {
        from: range.from,
        to: range.to,
      },
    });
  }
}
