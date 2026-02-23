import { Component, OnInit, computed, inject } from "@angular/core";
import { RouterLink } from "@angular/router";
import { UiButtonComponent } from "../../../shared/ui/button/button.component";
import { UiSkeletonComponent } from "../../../shared/ui/skeleton/skeleton.component";
import { StatePanelComponent } from "../../../shared/components/state-panel.component";
import { ChallengesStore } from "../data/challenges.store";

@Component({
  selector: "app-challenges-page",
  standalone: true,
  imports: [RouterLink, UiButtonComponent, UiSkeletonComponent, StatePanelComponent],
  template: `
    <section class="screen screen--dark sleep-summary-theme">
      @switch (viewState()) {
        @case ("loading") {
          <div class="screen__section">
            <ui-skeleton [height]="24" />
            <ui-skeleton [height]="140" />
          </div>
        }
        @case ("no-data") {
          <app-state-panel
            title="No challenges yet"
            message="Set a gentle weekly goal to stay consistent."
            actionLabel="Set weekly goal"
            (action)="reload()"
          />
        }
        @case ("syncing") {
          <app-state-panel
            title="Syncing challenges"
            message="Updating your weekly progress."
            actionLabel="View Device"
            (action)="reload()"
          />
        }
        @case ("error") {
          <app-state-panel
            title="Challenges unavailable"
            [message]="store.errorMessage() ?? 'We could not load your challenges.'"
            actionLabel="Retry"
            (action)="reload()"
          />
        }
        @default {
          <div class="screen__header">
            <div>
              <p class="screen__sub">Optional challenges</p>
              <h1 class="screen__title">Keep it gentle</h1>
              <p class="screen__sub">No streak penalties. Focus on consistency.</p>
            </div>
            <a class="primary-link" routerLink="/dashboard">Dashboard</a>
          </div>

          <div class="screen__section">
            <div class="chart-card">
              <p class="chart-card__summary">{{ goalLabel() }}</p>
              <div class="stage-viz__bar" aria-hidden="true"></div>
              <p class="chart-card__summary">{{ progressLabel() }}</p>
            </div>

            <div class="screen__cta">
              <ui-button>Set weekly goal</ui-button>
              <a class="primary-link" routerLink="/coach">View tips</a>
            </div>
          </div>
        }
      }
    </section>
  `,
})
export class ChallengesPageComponent implements OnInit {
  readonly store = inject(ChallengesStore);
  readonly viewState = this.store.status;

  readonly goalLabel = computed(() => {
    const challenge = this.store.summary()?.challenges[0];
    if (!challenge) {
      return "Weekly goal: 5 nights on time";
    }
    return `Weekly goal: ${challenge.title}`;
  });
  readonly progressLabel = computed(() => {
    const challenge = this.store.summary()?.challenges[0];
    if (!challenge) {
      return "Progress: 3 of 5 nights";
    }
    return `Progress: ${challenge.progress_current} of ${challenge.progress_target}`;
  });

  ngOnInit(): void {
    void this.store.loadChallenges();
  }

  reload(): void {
    void this.store.loadChallenges();
  }
}
