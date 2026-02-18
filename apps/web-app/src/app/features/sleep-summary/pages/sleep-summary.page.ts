import { Component, OnInit, computed, inject } from "@angular/core";
import { Router } from "@angular/router";
import { SleepSummaryStore } from "../data/sleep-summary.store";
import { MorningHeaderComponent } from "../components/morning-header.component";
import { ScoreCardComponent } from "../components/score-card.component";
import { StageVizComponent } from "../components/stage-viz/stage-viz.component";
import { MetricsGridComponent } from "../components/metrics-grid.component";
import { InsightCardComponent } from "../components/insight-card.component";
import { LoadingStateComponent } from "../components/states/loading-state.component";
import { NoDataStateComponent } from "../components/states/no-data-state.component";
import { SyncErrorStateComponent } from "../components/states/sync-error-state.component";

@Component({
  selector: "app-sleep-summary-page",
  standalone: true,
  imports: [
    MorningHeaderComponent,
    ScoreCardComponent,
    StageVizComponent,
    MetricsGridComponent,
    InsightCardComponent,
    LoadingStateComponent,
    NoDataStateComponent,
    SyncErrorStateComponent,
  ],
  template: `
    <section class="sleep-summary sleep-summary-theme">
      @switch (store.status()) {
        @case ("loading") {
          <app-loading-state />
        }
        @case ("missing") {
          <app-no-data-state (action)="reload()" />
        }
        @case ("syncing") {
          <app-sync-error-state
            state="syncing"
            actionLabel="Fix Tracking"
            (retry)="reload()"
          />
        }
        @case ("error") {
          <app-sync-error-state
            state="error"
            actionLabel="Fix Tracking"
            (retry)="reload()"
          />
        }
        @default {
          <div class="sleep-summary__content">
            <app-morning-header
              [dateLocal]="summary()?.dateLocal ?? ''"
              [syncStatus]="syncStatus()"
              [lastSyncAtLocal]="summary()?.dataQuality.lastSyncAtLocal ?? null"
            />
            <app-score-card [summary]="summary()" />
            <app-stage-viz
              [bins]="summary()?.stages.bins ?? []"
              [pct]="summary()?.stages.pct ?? defaultPct"
              [timeInBedMin]="summary()?.totals.timeInBedMin ?? 0"
              [bedtimeLocal]="summary()?.bedtimeLocal ?? ''"
            />
            <app-metrics-grid [summary]="summary()" />
            <app-insight-card [summary]="summary()" />
            <div class="sleep-summary__cta">
              <button
                class="sleep-summary__button"
                type="button"
                (click)="handlePrimaryAction()"
              >
                {{ store.primaryActionLabel() }}
              </button>
            </div>
          </div>
        }
      }
    </section>
  `,
})
export class SleepSummaryPageComponent implements OnInit {
  readonly store = inject(SleepSummaryStore);
  private readonly router = inject(Router);

  readonly summary = this.store.summary;
  readonly defaultPct = { awake: 0, light: 0, deep: 0, rem: 0 };

  readonly syncStatus = computed(() => {
    const status = this.summary()?.dataQuality.status;
    if (status === "syncing" || status === "error") {
      return status;
    }
    return null;
  });

  ngOnInit(): void {
    void this.store.loadLatest();
  }

  reload(): void {
    void this.store.loadLatest();
  }

  handlePrimaryAction(): void {
    const action = this.summary()?.primaryAction.action;
    if (action && action.startsWith("/")) {
      void this.router.navigateByUrl(action);
      return;
    }

    if (action === "open_improve") {
      void this.router.navigateByUrl("/sleep-summary/improve");
      return;
    }

    if (action === "open_analysis") {
      void this.router.navigateByUrl("/sleep-summary/analysis");
      return;
    }

    void this.router.navigateByUrl("/sleep-summary");
  }
}
