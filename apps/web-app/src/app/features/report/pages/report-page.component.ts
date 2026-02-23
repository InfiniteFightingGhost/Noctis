import { Component, OnInit, computed, inject } from "@angular/core";
import { RouterLink } from "@angular/router";
import { UiButtonComponent } from "../../../shared/ui/button/button.component";
import { UiSkeletonComponent } from "../../../shared/ui/skeleton/skeleton.component";
import { StatusBannerComponent } from "../../../shared/ui/status-banner/status-banner.component";
import { StatePanelComponent } from "../../../shared/components/state-panel.component";
import { ReportStore } from "../data/report.store";

@Component({
  selector: "app-report-page",
  standalone: true,
  imports: [
    RouterLink,
    UiButtonComponent,
    UiSkeletonComponent,
    StatusBannerComponent,
    StatePanelComponent,
  ],
  template: `
    <section class="screen screen--dark sleep-summary-theme">
      @switch (viewState()) {
        @case ("loading") {
          <div class="screen__section">
            <ui-skeleton [height]="20" />
            <ui-skeleton [height]="100" />
            <ui-skeleton [height]="140" />
            <ui-skeleton [height]="140" />
          </div>
        }
        @case ("no-data") {
            <app-state-panel
              title="No report yet"
              message="Sync your device to generate a nightly report."
              actionLabel="Check Device"
              (action)="reload()"
            />
          }
        @case ("syncing") {
          <app-state-panel
            title="Syncing data"
            message="We're updating your nightly report."
            actionLabel="View Device"
            (action)="reload()"
          />
        }
        @case ("error") {
          <app-state-panel
            title="Report unavailable"
            [message]="store.errorMessage() ?? 'We could not load this report.'"
            actionLabel="Retry"
            (action)="reload()"
          />
        }
        @default {
          <div class="screen__header">
            <div>
              <p class="screen__sub">Night Report</p>
              <h1 class="screen__title">{{ dateLabel() }}</h1>
              <p class="screen__sub">{{ timeRangeLabel() }}</p>
            </div>
            <a class="primary-link" routerLink="/dashboard">Back</a>
          </div>

          <div class="screen__section">
            @if (isPartial()) {
              <ui-status-banner
                variant="partial"
                title="Partial data"
                message="Some sensors dropped; accuracy reduced."
                actionLabel="Details"
              />
            }

            <div class="summary-strip" aria-label="Summary strip">
              <div class="summary-strip__item">
                <div class="summary-strip__label">Score</div>
                <div class="summary-strip__value">{{ scoreLabel() }}</div>
              </div>
              <div class="summary-strip__item">
                <div class="summary-strip__label">TST</div>
                <div class="summary-strip__value">{{ totalSleepLabel() }}</div>
              </div>
              <div class="summary-strip__item">
                <div class="summary-strip__label">Efficiency</div>
                <div class="summary-strip__value">{{ efficiencyLabel() }}</div>
              </div>
              <div class="summary-strip__item">
                <div class="summary-strip__label">Deep</div>
                <div class="summary-strip__value">{{ deepLabel() }}</div>
              </div>
            </div>

            <details class="chart-card" open>
              <summary class="section-header">
                <h3>Overview</h3>
                <span class="screen__sub">Stages + notes</span>
              </summary>
              <div class="screen__section">
                <div class="chart-card" aria-hidden="true">
                  <div class="stage-viz__bar"></div>
                  <p class="chart-card__summary">
                    Awake 6%, Light 52%, Deep 18%, REM 24%
                  </p>
                </div>
                <p class="chart-card__summary">
                  Summary: You held deep sleep longer in the first cycle.
                </p>
              </div>
            </details>

            <details class="chart-card">
              <summary class="section-header">
                <h3>Trends</h3>
                <span class="screen__sub">Last 7 nights</span>
              </summary>
              <div class="screen__section">
                <div class="chart-card" aria-hidden="true">
                  <div class="stage-viz__bar"></div>
                  <p class="chart-card__summary">Deep +12m vs last week</p>
                </div>
              </div>
            </details>

            <details class="chart-card">
              <summary class="section-header">
                <h3>Movement</h3>
                <span class="screen__sub">Peaks labeled</span>
              </summary>
              <div class="screen__section">
                <div class="chart-card" aria-hidden="true">
                  <div class="stage-viz__bar"></div>
                  <p class="chart-card__summary">
                    Movement: Low overall, peak at 03:10
                  </p>
                </div>
              </div>
            </details>

            <details class="chart-card">
              <summary class="section-header">
                <h3>Physiology</h3>
                <span class="screen__sub">HR + RR</span>
              </summary>
              <div class="screen__section">
                <div class="chart-card" aria-hidden="true">
                  <div class="stage-viz__bar"></div>
                  <p class="chart-card__summary">
                    Heart rate 52–67 bpm, breathing steady.
                  </p>
                </div>
              </div>
            </details>

            <div class="screen__cta">
              <ui-button>Share Report</ui-button>
              <a class="primary-link" routerLink="/coach">
                Coach me on this night
              </a>
            </div>
          </div>
        }
      }
    </section>
  `,
})
export class ReportPageComponent implements OnInit {
  readonly store = inject(ReportStore);
  readonly report = this.store.report;

  readonly viewState = this.store.status;
  readonly isPartial = this.store.isPartial;

  readonly dateLabel = computed(() => this.report()?.dateLocal ?? "Tue, Feb 18");
  readonly timeRangeLabel = computed(() => {
    const report = this.report();
    if (!report) {
      return "10:38pm – 6:41am";
    }
    return `${report.bedtimeLocal} – ${report.waketimeLocal}`;
  });
  readonly scoreLabel = computed(() => {
    const score = this.report()?.score;
    return score !== undefined ? String(score) : "—";
  });
  readonly totalSleepLabel = computed(() => {
    const minutes = this.report()?.totals.totalSleepMin;
    return this.formatMinutes(minutes);
  });
  readonly efficiencyLabel = computed(() => {
    const pct = this.report()?.totals.sleepEfficiencyPct;
    return pct !== undefined ? `${Math.round(pct)}%` : "—";
  });
  readonly deepLabel = computed(() => {
    const pct = this.report()?.metrics.deepPct;
    return pct !== undefined ? `${Math.round(pct)}%` : "—";
  });

  ngOnInit(): void {
    void this.store.loadLatest();
  }

  reload(): void {
    void this.store.loadLatest();
  }

  private formatMinutes(value?: number | null): string {
    if (value == null) {
      return "—";
    }
    const hours = Math.floor(value / 60);
    const minutes = Math.round(value % 60);
    if (hours <= 0) {
      return `${minutes}m`;
    }
    return `${hours}h ${minutes}m`;
  }
}
