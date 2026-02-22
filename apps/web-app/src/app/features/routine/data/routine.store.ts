import { inject, Injectable, signal } from "@angular/core";
import { firstValueFrom } from "rxjs";
import { toApiError } from "../../../core/api/api-errors";
import { Routine, RoutinesApi } from "../../../core/api/routines.api";

export type RoutineViewState =
  | "loading"
  | "success"
  | "no-data"
  | "syncing"
  | "error";

@Injectable({ providedIn: "root" })
export class RoutineStore {
  private readonly api = inject(RoutinesApi);

  readonly routine = signal<Routine | null>(null);
  readonly status = signal<RoutineViewState>("loading");
  readonly errorMessage = signal<string | null>(null);
  readonly isFetching = signal(false);

  async loadRoutine(): Promise<void> {
    this.status.set("loading");
    this.errorMessage.set(null);
    this.isFetching.set(true);

    try {
      const routine = await firstValueFrom(this.api.getRoutine());
      this.routine.set(routine);
      this.status.set(routine.steps.length > 0 ? "success" : "no-data");
    } catch (error) {
      const parsed = toApiError(error);
      this.errorMessage.set(parsed.message);
      this.status.set("error");
    } finally {
      this.isFetching.set(false);
    }
  }
}
