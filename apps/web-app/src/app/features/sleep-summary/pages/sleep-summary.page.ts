import { Component, OnInit, computed, inject } from "@angular/core";
import { Router, RouterLink } from "@angular/router";
import { SleepSummaryStore } from "../data/sleep-summary.store";
import { MorningHeaderComponent } from "../components/morning-header.component";
import { ScoreCardComponent } from "../components/score-card.component";
import { StageVizComponent } from "../components/stage-viz/stage-viz.component";
import { MetricsGridComponent } from "../components/metrics-grid.component";
import { InsightCardComponent } from "../components/insight-card.component";
import { LoadingStateComponent } from "../components/states/loading-state.component";
import { NoDataStateComponent } from "../components/states/no-data-state.component";
import { SyncErrorStateComponent } from "../components/states/sync-error-state.component";
import { StatusBannerComponent } from "../../../shared/ui/status-banner/status-banner.component";

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
    StatusBannerComponent,
    RouterLink,
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
          @if (summary(); as sleepSummary) {
            <div class="sleep-summary__content">
              @if (isPartial()) {
                <ui-status-banner
                  variant="partial"
                  title="Partial data"
                  message="Some sensors dropped; accuracy reduced."
                  actionLabel="Details"
                  (action)="openDetails()"
                />
              }
              <app-morning-header
                [dateLocal]="sleepSummary.dateLocal"
                [syncStatus]="syncStatus()"
                [lastSyncAtLocal]="sleepSummary.dataQuality.lastSyncAtLocal ?? null"
              />
              <app-score-card [summary]="sleepSummary" />
              <app-stage-viz
                [bins]="sleepSummary.stages.bins"
                [pct]="sleepSummary.stages.pct"
                [timeInBedMin]="sleepSummary.totals.timeInBedMin"
                [bedtimeLocal]="sleepSummary.bedtimeLocal"
              />
              <app-metrics-grid [summary]="sleepSummary" />
              <app-insight-card [summary]="sleepSummary" />
              <div class="sleep-summary__cta">
                <button
                  class="sleep-summary__button"
                  type="button"
                  (click)="handlePrimaryAction()"
                >
                  {{ store.primaryActionLabel() }}
                </button>
                <a class="primary-link" routerLink="/report">
                  View Full Analysis
                </a>
              </div>
            </div>
          }
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

  readonly isPartial = computed(
    () => this.summary()?.dataQuality.status === "partial",
  );

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
      void this.router.navigateByUrl("/coach");
      return;
    }

    if (action === "open_analysis") {
      void this.router.navigateByUrl("/report");
      return;
    }

    void this.router.navigateByUrl("/coach");
  }

  openDetails(): void {
    void this.router.navigateByUrl("/device");
  }
}
