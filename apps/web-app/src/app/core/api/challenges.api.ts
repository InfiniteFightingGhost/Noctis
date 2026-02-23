import { inject, Injectable } from "@angular/core";
import { ApiClient } from "./http-client";

export type Challenge = {
  id: string;
  title: string;
  description: string;
  progress_current: number;
  progress_target: number;
  status: "active" | "completed" | "available";
};

export type ChallengesSummary = {
  week_start: string;
  week_end: string;
  challenges: Challenge[];
};

@Injectable({ providedIn: "root" })
export class ChallengesApi {
  private readonly api = inject(ApiClient);

  getChallenges() {
    return this.api.get<ChallengesSummary>("/v1/challenges");
  }
}
