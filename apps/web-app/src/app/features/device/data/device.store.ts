import { computed, inject, Injectable, signal } from "@angular/core";
import { firstValueFrom } from "rxjs";
import { toApiError } from "../../../core/api/api-errors";
import { DeviceResponse, DevicesApi } from "../../../core/api/devices.api";

export type DeviceViewState =
  | "loading"
  | "success"
  | "no-data"
  | "syncing"
  | "error";

@Injectable({ providedIn: "root" })
export class DeviceStore {
  private readonly api = inject(DevicesApi);

  readonly devices = signal<DeviceResponse[]>([]);
  readonly status = signal<DeviceViewState>("loading");
  readonly errorMessage = signal<string | null>(null);
  readonly isFetching = signal(false);

  readonly primaryDevice = computed(() => this.devices()[0] ?? null);
  readonly secondaryDevice = computed(() => this.devices()[1] ?? null);

  async loadDevices(): Promise<void> {
    this.status.set("loading");
    this.errorMessage.set(null);
    this.isFetching.set(true);

    try {
      const devices = await firstValueFrom(this.api.getDevices());
      this.devices.set(devices);
      this.status.set(devices.length > 0 ? "success" : "no-data");
    } catch (error) {
      const parsed = toApiError(error);
      this.errorMessage.set(parsed.message);
      this.status.set("error");
    } finally {
      this.isFetching.set(false);
    }
  }
}
