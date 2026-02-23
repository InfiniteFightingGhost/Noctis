import { inject, Injectable } from "@angular/core";
import { ApiClient } from "./http-client";

export type AlarmSoundOption = {
  id: string;
  label: string;
  mood?: string | null;
};

export type AlarmSettings = {
  id: string;
  wake_time: string;
  wake_window_minutes: number;
  sunrise_enabled: boolean;
  sunrise_intensity: number;
  sound_id: string;
  sound_options: AlarmSoundOption[];
  updated_at: string;
};

export type AlarmUpdatePayload = {
  wakeTime?: string;
  wakeWindowMinutes?: number;
  sunriseEnabled?: boolean;
  sunriseIntensity?: number;
  soundId?: string;
};

@Injectable({ providedIn: "root" })
export class AlarmApi {
  private readonly api = inject(ApiClient);

  getSettings() {
    return this.api.get<AlarmSettings>("/v1/alarm");
  }

  updateSettings(payload: AlarmUpdatePayload) {
    return this.api.put<AlarmSettings>("/v1/alarm", {
      wake_time: payload.wakeTime,
      wake_window_minutes: payload.wakeWindowMinutes,
      sunrise_enabled: payload.sunriseEnabled,
      sunrise_intensity: payload.sunriseIntensity,
      sound_id: payload.soundId,
    });
  }
}
