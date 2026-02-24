import { Component, OnInit, computed, inject } from "@angular/core";
import { RouterLink } from "@angular/router";
import { SleepSummaryStore } from "../data/sleep-summary.store";

@Component({
  selector: "app-sleep-summary-improve-page",
  standalone: true,
  imports: [RouterLink],
  template: `
    <section class="screen screen--dark sleep-summary-theme">
      @switch (store.status()) {
        @case ("loading") {
          <div class="screen__section">
            <p class="screen__sub">Loading tonight's plan...</p>
          </div>
        }
        @case ("error") {
          <div class="state-panel">
            <h3>Plan unavailable</h3>
            <p>{{ store.errorMessage() ?? "Could not load recommendations." }}</p>
          </div>
        }
        @case ("missing") {
          <div class="state-panel">
            <h3>No recent sleep session</h3>
            <p>Sync a recording first to generate tonight's improvement plan.</p>
          </div>
        }
        @case ("syncing") {
          <div class="state-panel">
            <h3>Preparing plan</h3>
            <p>We're still processing your latest recording.</p>
          </div>
        }
        @default {
          @if (summary(); as sleepSummary) {
            <header class="screen__header">
              <div>
                <p class="screen__sub">Improve Tonight</p>
                <h1 class="screen__title">Action plan for next sleep</h1>
              </div>
              <a class="primary-link" routerLink="/dashboard">Back</a>
            </header>

            <div class="screen__section">
              <article class="insight-card">
                <p class="insight-card__text">{{ sleepSummary.insight.text }}</p>
              </article>

              <article class="chart-card">
                <h3>Priority target</h3>
                <p class="chart-card__summary">{{ priorityTargetLabel() }}</p>
              </article>

              <div class="screen__cta">
                <a class="sleep-summary__button" routerLink="/routine/edit">
                  Update tonight's routine
                </a>
                <a class="primary-link" routerLink="/alarm">
                  Adjust tomorrow's wake plan
                </a>
              </div>
            </div>
          }
        }
      }
    </section>
  `,
})
export class SleepSummaryImprovePageComponent implements OnInit {
  readonly store = inject(SleepSummaryStore);
  readonly summary = this.store.summary;

  readonly priorityTargetLabel = computed(() => {
    const summary = this.summary();
    if (!summary) {
      return "No target available.";
    }

    if (summary.metrics.deepPct < 20) {
      return "Increase deep sleep by extending wind-down and reducing movement before bedtime.";
    }

    if (summary.totals.sleepEfficiencyPct < 85) {
      return "Raise sleep efficiency by narrowing your wake window and keeping a fixed bedtime.";
    }

    return "Maintain consistency: keep your current bedtime and routine timing.";
  });

  ngOnInit(): void {
    void this.store.loadLatest();
  }
}
