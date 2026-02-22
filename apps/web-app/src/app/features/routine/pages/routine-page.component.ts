import { Component, OnInit, computed, inject } from "@angular/core";
import { RouterLink } from "@angular/router";
import { UiButtonComponent } from "../../../shared/ui/button/button.component";
import { UiSkeletonComponent } from "../../../shared/ui/skeleton/skeleton.component";
import { StatePanelComponent } from "../../../shared/components/state-panel.component";
import { RoutineStore } from "../data/routine.store";

@Component({
  selector: "app-routine-page",
  standalone: true,
  imports: [RouterLink, UiButtonComponent, UiSkeletonComponent, StatePanelComponent],
  template: `
    <section class="screen screen--dark sleep-summary-theme">
      @switch (viewState()) {
        @case ("loading") {
          <div class="screen__section">
            <ui-skeleton [height]="26" />
            <ui-skeleton [height]="160" />
          </div>
        }
        @case ("no-data") {
          <app-state-panel
            title="No routine yet"
            message="Create a calming routine to wind down."
            actionLabel="Create Routine"
            (action)="reload()"
          />
        }
        @case ("syncing") {
          <app-state-panel
            title="Syncing routine"
            message="Updating your routine steps."
            actionLabel="View Device"
            (action)="reload()"
          />
        }
        @case ("error") {
          <app-state-panel
            title="Routine unavailable"
            [message]="store.errorMessage() ?? 'We could not load your routine.'"
            actionLabel="Retry"
            (action)="reload()"
          />
        }
        @default {
          <div class="screen__header">
            <div>
              <p class="screen__sub">Bedtime Routine</p>
              <h1 class="screen__title">{{ routineTitle() }}</h1>
              <p class="screen__sub">{{ routineDurationLabel() }}</p>
            </div>
            <a class="primary-link" routerLink="/alarm/settings">Back</a>
          </div>

          <div class="screen__section">
            <div class="list-row">
              <div>
                <strong>Dim lights</strong>
                <div class="list-row__meta">2 min</div>
              </div>
              <span>🕯️</span>
            </div>
            <div class="list-row">
              <div>
                <strong>Breathing reset</strong>
                <div class="list-row__meta">4 min</div>
              </div>
              <span>🌬️</span>
            </div>
            <div class="list-row">
              <div>
                <strong>Gentle stretch</strong>
                <div class="list-row__meta">3 min</div>
              </div>
              <span>🧘</span>
            </div>
            <div class="list-row">
              <div>
                <strong>Sleep soundscape</strong>
                <div class="list-row__meta">3 min</div>
              </div>
              <span>🎧</span>
            </div>

            <div class="chart-card">
              <p class="chart-card__summary">
                AI suggested: add a 2-minute journaling step tonight.
              </p>
              <button class="chip" type="button">Apply suggestion</button>
            </div>

            <div class="screen__cta">
              <ui-button>Start Routine</ui-button>
              <a class="primary-link" routerLink="/routine/edit">Edit routine</a>
            </div>
          </div>
        }
      }
    </section>
  `,
})
export class RoutinePageComponent implements OnInit {
  readonly store = inject(RoutineStore);
  readonly viewState = this.store.status;

  readonly routineTitle = computed(
    () => this.store.routine()?.title ?? "Tonight's routine",
  );
  readonly routineDurationLabel = computed(() => {
    const minutes = this.store.routine()?.total_minutes;
    if (minutes == null) {
      return "~12 min total";
    }
    return `~${minutes} min total`;
  });

  ngOnInit(): void {
    void this.store.loadRoutine();
  }

  reload(): void {
    void this.store.loadRoutine();
  }
}
