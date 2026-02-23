import { computed, inject, Injectable, signal } from "@angular/core";
import { firstValueFrom } from "rxjs";
import { toApiError } from "../../../core/api/api-errors";
import { AlarmApi, AlarmSettings } from "../../../core/api/alarm.api";

export type AlarmViewState =
  | "loading"
  | "success"
  | "no-data"
  | "syncing"
  | "error";

@Injectable({ providedIn: "root" })
export class AlarmStore {
  private readonly api = inject(AlarmApi);

  readonly settings = signal<AlarmSettings | null>(null);
  readonly status = signal<AlarmViewState>("loading");
  readonly errorMessage = signal<string | null>(null);
  readonly isFetching = signal(false);

  readonly activeSound = computed(() => {
    const settings = this.settings();
    if (!settings) {
      return null;
    }
    return (
      settings.sound_options.find((option) => option.id === settings.sound_id) ??
      null
    );
  });

  async loadSettings(): Promise<void> {
    this.status.set("loading");
    this.errorMessage.set(null);
    this.isFetching.set(true);

    try {
      const settings = await firstValueFrom(this.api.getSettings());
      this.settings.set(settings);
      this.status.set("success");
    } catch (error) {
      const parsed = toApiError(error);
      this.errorMessage.set(parsed.message);
      this.status.set("error");
    } finally {
      this.isFetching.set(false);
    }
  }
}
