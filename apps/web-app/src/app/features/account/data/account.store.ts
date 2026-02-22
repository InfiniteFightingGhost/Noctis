import { HttpErrorResponse } from "@angular/common/http";
import { computed, inject, Injectable, signal } from "@angular/core";
import { firstValueFrom } from "rxjs";
import { AccountApi, UserResponse } from "../api/account.api";

export type AccountStatus = "idle" | "loading" | "success" | "error";

@Injectable({ providedIn: "root" })
export class AccountStore {
  private readonly api = inject(AccountApi);

  readonly name = signal("");
  readonly externalId = signal("");
  readonly adminToken = signal("");
  readonly status = signal<AccountStatus>("idle");
  readonly errorMessage = signal("Something went wrong. Please try again.");
  readonly createdUser = signal<UserResponse | null>(null);

  readonly canSubmit = computed(() => {
    return (
      this.status() !== "loading" &&
      this.name().trim().length > 0 &&
      this.adminToken().trim().length > 0
    );
  });

  readonly submitLabel = computed(() =>
    this.status() === "loading" ? "Creating..." : "Create user",
  );

  readonly statusLabel = computed(() => {
    if (this.status() === "loading") {
      return "Creating";
    }
    if (this.status() === "success") {
      return "Ready";
    }
    if (this.status() === "error") {
      return "Action needed";
    }
    return "Idle";
  });

  updateName(value: string) {
    this.name.set(value);
    this.resetStatus();
  }

  updateExternalId(value: string) {
    this.externalId.set(value);
    this.resetStatus();
  }

  updateAdminToken(value: string) {
    this.adminToken.set(value);
    this.resetStatus();
  }

  async createUser(): Promise<void> {
    if (!this.canSubmit()) {
      return;
    }

    this.status.set("loading");
    this.errorMessage.set("Something went wrong. Please try again.");
    this.createdUser.set(null);

    try {
      const created = await firstValueFrom(
        this.api.createUser(
          {
            name: this.name().trim(),
            externalId: this.externalId().trim() || null,
          },
          this.adminToken().trim(),
        ),
      );
      this.createdUser.set(created);
      this.status.set("success");
    } catch (error) {
      this.status.set("error");
      this.errorMessage.set(this.extractErrorMessage(error));
    }
  }

  private extractErrorMessage(error: unknown): string {
    if (error instanceof HttpErrorResponse) {
      if (error.status === 401) {
        return "Unauthorized. Provide a valid admin token.";
      }
      if (error.status === 403) {
        return "Forbidden. Admin role required.";
      }
      if (typeof error.error === "string") {
        return error.error;
      }
      if (error.error?.detail) {
        return error.error.detail;
      }
      if (error.error?.error?.message) {
        return error.error.error.message;
      }
    }
    return "Unable to create user. Please try again.";
  }

  private resetStatus() {
    if (this.status() === "loading") {
      return;
    }
    if (this.status() !== "idle") {
      this.status.set("idle");
      this.errorMessage.set("Something went wrong. Please try again.");
      this.createdUser.set(null);
    }
  }
}
