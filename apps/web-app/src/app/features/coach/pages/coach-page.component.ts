import { Component, OnInit, computed, inject } from "@angular/core";
import { RouterLink } from "@angular/router";
import { UiSkeletonComponent } from "../../../shared/ui/skeleton/skeleton.component";
import { StatusBannerComponent } from "../../../shared/ui/status-banner/status-banner.component";
import { StatePanelComponent } from "../../../shared/components/state-panel.component";
import { CoachStore } from "../data/coach.store";

@Component({
  selector: "app-coach-page",
  standalone: true,
  imports: [
    RouterLink,
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
              <p class="chart-card__summary">Generated {{ generatedAtLabel() }}.</p>
            </article>

            @if (additionalInsights().length > 0) {
              <div class="chart-card">
                <h3>More recommendations</h3>
                <ul class="coach-list">
                  @for (item of additionalInsights(); track item.id) {
                    <li class="chart-card__summary">{{ item.message }}</li>
                  }
                </ul>
              </div>
            }

            <div class="screen__cta">
              <a class="sleep-summary__button" routerLink="/routine/edit">
                Apply to bedtime routine
              </a>
              <div class="chip-row">
                <a class="chip" routerLink="/routine">Routine</a>
                <a class="chip" routerLink="/alarm">Alarm</a>
                <a class="chip" routerLink="/report">Night report</a>
              </div>
            </div>
          </div>
        }
      }
    </section>
  `,
  styles: [
    `
      .coach-list {
        margin: 0;
        padding-left: 1rem;
        display: grid;
        gap: 0.35rem;
      }
    `,
  ],
})
export class CoachPageComponent implements OnInit {
  readonly store = inject(CoachStore);
  readonly viewState = this.store.status;
  readonly isPartial = this.store.isPartial;

  readonly primaryInsight = computed(() => {
    return (
      this.store.summary()?.insights[0]?.message ??
      "No coaching insight available yet."
    );
  });
  readonly additionalInsights = computed(
    () => this.store.summary()?.insights.slice(1, 4) ?? [],
  );
  readonly generatedAtLabel = computed(
    () => this.store.summary()?.generated_at ?? "just now",
  );

  ngOnInit(): void {
    void this.store.loadSummary();
  }

  reload(): void {
    void this.store.loadSummary();
  }
}
