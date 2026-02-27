import { Component, OnInit, inject } from "@angular/core";
import { RouterLink } from "@angular/router";
import { SleepSummaryStore } from "../data/sleep-summary.store";

@Component({
  selector: "app-sleep-summary-analysis-page",
  standalone: true,
  imports: [RouterLink],
  template: `
    <section class="screen screen--dark sleep-summary-theme">
      @switch (store.status()) {
        @case ("loading") {
          <div class="screen__section">
            <p class="screen__sub">Loading analysis...</p>
          </div>
        }
        @case ("error") {
          <div class="state-panel">
            <h3>Analysis unavailable</h3>
            <p>{{ store.errorMessage() ?? "Could not load analysis." }}</p>
          </div>
        }
        @case ("missing") {
          <div class="state-panel">
            <h3>No data for analysis</h3>
            <p>Sync your device and return once recording is available.</p>
          </div>
        }
        @case ("syncing") {
          <div class="state-panel">
            <h3>Analysis is syncing</h3>
            <p>We'll show full analysis after sync completes.</p>
          </div>
        }
        @default {
          @if (summary(); as sleepSummary) {
            <header class="screen__header">
              <div>
                <p class="screen__sub">Full Analysis</p>
                <h1 class="screen__title">{{ sleepSummary.dateLocal }}</h1>
              </div>
              <a class="primary-link" routerLink="/dashboard">Back</a>
            </header>

            <div class="screen__section">
              <article class="chart-card">
                <h3>Timing</h3>
                <p class="chart-card__summary">
                  In bed {{ sleepSummary.bedtimeLocal }} to
                  {{ sleepSummary.waketimeLocal }}.
                </p>
                <p class="chart-card__summary">
                  Total sleep {{ sleepSummary.totals.totalSleepMin }} min, efficiency
                  {{ sleepSummary.totals.sleepEfficiencyPct }}%.
                </p>
              </article>

              <article class="chart-card">
                <h3>Sleep architecture</h3>
                <p class="chart-card__summary">
                  Light {{ sleepSummary.stages.pct.light }}%, Deep
                  {{ sleepSummary.stages.pct.deep }}%, REM
                  {{ sleepSummary.stages.pct.rem }}%, Awake
                  {{ sleepSummary.stages.pct.awake }}%.
                </p>
              </article>

              <article class="chart-card">
                <h3>Physiology</h3>
                <p class="chart-card__summary">
                  Avg HR {{ sleepSummary.metrics.avgHrBpm }} bpm, respiratory rate
                  {{ sleepSummary.metrics.avgRrBrpm }} brpm, movement
                  {{ sleepSummary.metrics.movementPct }}%.
                </p>
              </article>

              <article class="insight-card">
                <p class="insight-card__text">{{ sleepSummary.insight.text }}</p>
              </article>
            </div>
          }
        }
      }
    </section>
  `,
})
export class SleepSummaryAnalysisPageComponent implements OnInit {
  readonly store = inject(SleepSummaryStore);
  readonly summary = this.store.summary;

  ngOnInit(): void {
    void this.store.loadLatest();
  }
}
