import { Routes } from "@angular/router";
import { SleepSummaryPageComponent } from "./pages/sleep-summary.page";
import { SleepSummaryImprovePageComponent } from "./pages/sleep-summary-improve.page";
import { SleepSummaryAnalysisPageComponent } from "./pages/sleep-summary-analysis.page";

export const SLEEP_SUMMARY_ROUTES: Routes = [
  {
    path: "",
    component: SleepSummaryPageComponent,
  },
  {
    path: "improve",
    component: SleepSummaryImprovePageComponent,
  },
  {
    path: "analysis",
    component: SleepSummaryAnalysisPageComponent,
  },
];
