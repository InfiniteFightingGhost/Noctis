import { Component, OnInit, computed, inject } from "@angular/core";
import { RouterLink } from "@angular/router";
import { UiButtonComponent } from "../../../shared/ui/button/button.component";
import { HomeStore } from "../data/home.store";

@Component({
  selector: "app-home-page",
  standalone: true,
  imports: [RouterLink, UiButtonComponent],
  template: `
    <section class="screen screen--dark sleep-summary-theme">
      <header class="screen__header">
        <div>
          <p class="screen__sub">Home</p>
          <h1 class="screen__title">{{ headline() }}</h1>
        </div>
      </header>

      <section class="screen__section" aria-live="polite">
        @switch (store.status()) {
          @case ("loading") {
            <p class="screen__sub">Loading your home overview...</p>
          }
          @case ("error") {
            <div class="state-panel">
              <h3>Home data unavailable</h3>
              <p>{{ store.errorMessage() ?? "Unable to load home overview." }}</p>
              <ui-button (click)="reload()">Retry</ui-button>
            </div>
          }
          @default {
            <article class="chart-card">
              <p class="chart-card__summary">{{ lede() }}</p>
              <p class="list-row__meta">{{ updatedAtLabel() }}</p>
            </article>

            <nav class="quick-actions" aria-label="Primary workflows">
              <a class="list-row" routerLink="/dashboard">
                <strong>Review last night</strong>
                <span class="list-row__meta">Dashboard</span>
              </a>
              <a class="list-row" routerLink="/alarm">
                <strong>Adjust smart alarm</strong>
                <span class="list-row__meta">Alarm</span>
              </a>
              <a class="list-row" routerLink="/routine/edit">
                <strong>Update bedtime routine</strong>
                <span class="list-row__meta">Routine editor</span>
              </a>
            </nav>
          }
        }
      </section>
    </section>
  `,
  styles: [
    `
      .quick-actions {
        display: grid;
        gap: 0.65rem;
      }

      .quick-actions a {
        text-decoration: none;
        color: inherit;
      }
    `,
  ],
})
export class HomePageComponent implements OnInit {
  readonly store = inject(HomeStore);

  readonly headline = computed(
    () => this.store.overview()?.headline ?? "Home",
  );
  readonly lede = computed(
    () =>
      this.store.overview()?.lede ??
      "Your sleep overview appears here once nightly sync finishes.",
  );
  readonly updatedAtLabel = computed(() => {
    const value = this.store.overview()?.updated_at;
    return value ? `Updated ${value}` : "Updated time unavailable";
  });

  ngOnInit(): void {
    void this.store.loadOverview();
  }

  reload(): void {
    void this.store.loadOverview();
  }
}
