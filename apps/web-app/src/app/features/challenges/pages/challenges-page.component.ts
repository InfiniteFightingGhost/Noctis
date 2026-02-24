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
            <p class="screen__sub">Week {{ weekLabel() }}</p>

            @for (challenge of challenges(); track challenge.id) {
              <div class="list-row">
                <div>
                  <strong>{{ challenge.title }}</strong>
                  <div class="list-row__meta">{{ challenge.description }}</div>
                </div>
                <span class="list-row__meta">
                  {{ challenge.progress_current }}/{{ challenge.progress_target }}
                </span>
              </div>
            }

            <div class="screen__cta">
              <ui-button (click)="reload()">Refresh progress</ui-button>
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

  readonly challenges = computed(() => this.store.summary()?.challenges ?? []);
  readonly weekLabel = computed(() => {
    const summary = this.store.summary();
    if (!summary) {
      return "Unavailable";
    }
    return `${summary.week_start} to ${summary.week_end}`;
  });

  ngOnInit(): void {
    void this.store.loadChallenges();
  }

  reload(): void {
    void this.store.loadChallenges();
  }
}
