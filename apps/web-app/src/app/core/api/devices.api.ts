import { inject, Injectable } from "@angular/core";
import { ApiClient } from "./http-client";

export type DeviceCreatePayload = {
  name: string;
  externalId?: string | null;
};

export type DeviceUserLinkPayload = {
  userId: string;
};

export type DeviceResponse = {
  id: string;
  name: string;
  external_id: string | null;
  user_id: string | null;
  created_at: string;
};

@Injectable({ providedIn: "root" })
export class DevicesApi {
  private readonly api = inject(ApiClient);

  createDevice(payload: DeviceCreatePayload) {
    return this.api.post<DeviceResponse>("/v1/devices", {
      name: payload.name,
      external_id: payload.externalId ?? null,
    });
  }

  linkDeviceUser(deviceId: string, payload: DeviceUserLinkPayload) {
    return this.api.put<DeviceResponse>(`/v1/devices/${deviceId}/user`, {
      user_id: payload.userId,
    });
  }

  unlinkDeviceUser(deviceId: string) {
    return this.api.delete<DeviceResponse>(`/v1/devices/${deviceId}/user`);
  }

  getDevices() {
    return this.api.get<DeviceResponse[]>("/v1/devices");
  }
}
