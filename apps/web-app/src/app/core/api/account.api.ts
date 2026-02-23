import { HttpHeaders } from "@angular/common/http";
import { inject, Injectable } from "@angular/core";
import { ApiClient } from "./http-client";
import { CreateUserPayload, UserResponse } from "./users.api";

@Injectable({ providedIn: "root" })
export class AccountApi {
  private readonly api = inject(ApiClient);

  createUser(payload: CreateUserPayload, adminToken: string) {
    const headers = new HttpHeaders({
      Authorization: `Bearer ${adminToken}`,
    });

    return this.api.post<UserResponse>(
      "/v1/users",
      {
        name: payload.name,
        external_id: payload.externalId ?? null,
      },
      { headers },
    );
  }
}
