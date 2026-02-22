import { Component, OnInit, computed, inject } from "@angular/core";
import { RouterLink } from "@angular/router";
import { UiButtonComponent } from "../../../shared/ui/button/button.component";
import { UiSkeletonComponent } from "../../../shared/ui/skeleton/skeleton.component";
import { StatusBannerComponent } from "../../../shared/ui/status-banner/status-banner.component";
import { StatePanelComponent } from "../../../shared/components/state-panel.component";
import { CoachStore } from "../data/coach.store";

@Component({
  selector: "app-coach-page",
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
            <ui-skeleton [height]="24" />
            <ui-skeleton [height]="120" />
            <ui-skeleton [height]="50" />
          </div>
        }
        @case ("no-data") {
          <app-state-panel
            title="No guidance yet"
            message="Sync a recent night to unlock coaching."
            actionLabel="Check Device"
            (action)="reload()"
          />
        }
        @case ("syncing") {
          <app-state-panel
            title="Updating insights"
            message="We're syncing your sleep data."
            actionLabel="View Device"
            (action)="reload()"
          />
        }
        @case ("error") {
          <app-state-panel
            title="Coach unavailable"
            [message]="store.errorMessage() ?? 'We could not load coaching advice.'"
            actionLabel="Retry"
            (action)="reload()"
          />
        }
        @default {
          <div class="screen__header">
            <div>
              <p class="screen__sub">AI Coach</p>
              <h1 class="screen__title">Recommended for tonight</h1>
            </div>
            <a class="primary-link" routerLink="/dashboard">Dashboard</a>
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
            <article class="insight-card">
              <p class="insight-card__text">
                {{ primaryInsight() }}
              </p>
              <p class="chart-card__summary">Based on last 7 nights.</p>
            </article>

            <div class="screen__cta">
              <ui-button>Set Bedtime to 10:15pm</ui-button>
              <div class="chip-row">
                <button class="chip" type="button">Relaxing routine</button>
                <button class="chip" type="button">Optimize alarm</button>
                <button class="chip" type="button">Why this?</button>
              </div>
            </div>

            <div class="chip-row" aria-label="Insight feedback">
              <button class="chip" type="button">👍 Helpful</button>
              <button class="chip" type="button">👎 Not for me</button>
            </div>
          </div>
        }
      }
    </section>
  `,
})
export class CoachPageComponent implements OnInit {
  readonly store = inject(CoachStore);
  readonly viewState = this.store.status;
  readonly isPartial = this.store.isPartial;

  readonly primaryInsight = computed(() => {
    return (
      this.store.summary()?.insights[0]?.message ??
      "Try going to bed around 10:15pm tonight to support deeper sleep."
    );
  });

  ngOnInit(): void {
    void this.store.loadSummary();
  }

  reload(): void {
    void this.store.loadSummary();
  }
}
