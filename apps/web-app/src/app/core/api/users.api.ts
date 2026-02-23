import { inject, Injectable } from "@angular/core";
import { ApiClient } from "./http-client";

export type CreateUserPayload = {
  name: string;
  externalId?: string | null;
};

export type UserResponse = {
  id: string;
  name: string;
  external_id: string | null;
  created_at: string;
};

@Injectable({ providedIn: "root" })
export class UsersApi {
  private readonly api = inject(ApiClient);

  createUser(payload: CreateUserPayload) {
    return this.api.post<UserResponse>("/v1/users", {
      name: payload.name,
      external_id: payload.externalId ?? null,
    });
  }
}
