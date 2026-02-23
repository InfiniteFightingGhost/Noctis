import { inject, Injectable } from "@angular/core";
import { ApiClient } from "./http-client";

export type RoutineStep = {
  id: string;
  title: string;
  duration_minutes: number;
  emoji?: string | null;
};

export type Routine = {
  id: string;
  title: string;
  total_minutes: number;
  steps: RoutineStep[];
  updated_at: string;
};

export type RoutineUpdatePayload = {
  title?: string;
  steps?: Array<{
    id?: string;
    title: string;
    durationMinutes: number;
    emoji?: string | null;
  }>;
};

@Injectable({ providedIn: "root" })
export class RoutinesApi {
  private readonly api = inject(ApiClient);

  getRoutine() {
    return this.api.get<Routine>("/v1/routines/current");
  }

  updateRoutine(payload: RoutineUpdatePayload) {
    return this.api.put<Routine>("/v1/routines/current", {
      title: payload.title,
      steps: payload.steps?.map((step) => ({
        id: step.id,
        title: step.title,
        duration_minutes: step.durationMinutes,
        emoji: step.emoji ?? null,
      })),
    });
  }
}
