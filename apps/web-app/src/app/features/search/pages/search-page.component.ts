import { Component, OnInit, computed, inject } from "@angular/core";
import { RouterLink } from "@angular/router";
import { UiButtonComponent } from "../../../shared/ui/button/button.component";
import { SearchStore } from "../data/search.store";

@Component({
  selector: "app-search-page",
  standalone: true,
  imports: [RouterLink, UiButtonComponent],
  template: `
    <section class="screen screen--dark sleep-summary-theme">
      <header class="screen__header">
        <div>
          <p class="screen__sub">Search</p>
          <h1 class="screen__title">Find sleep records and settings</h1>
        </div>
      </header>

      <form class="screen__section" role="search" (submit)="onSubmit($event)">
        <label class="form-field">
          <span class="form-field__label">Query</span>
          <input
            class="form-field__input"
            type="search"
            [value]="store.query()"
            minlength="2"
            required
            placeholder="Search by keyword, device, routine, or user"
            (input)="onQueryInput($event)"
          />
        </label>
        <div class="screen__cta">
          <ui-button [disabled]="!canSubmit() || store.isFetching()">
            {{ store.isFetching() ? "Searching..." : "Search" }}
          </ui-button>
        </div>
      </form>

      <section class="screen__section" aria-live="polite">
        @switch (store.status()) {
          @case ("loading") {
            <p class="screen__sub">Searching for "{{ queryLabel() }}"...</p>
          }
          @case ("error") {
            <p class="form-error" role="alert">
              {{ store.errorMessage() ?? "Unable to load search results." }}
            </p>
          }
          @case ("no-data") {
            <div class="state-panel">
              <h3>No matches found</h3>
              <p>Try broader terms like "alarm" or "routine".</p>
            </div>
          }
          @case ("success") {
            <p class="screen__sub">{{ resultsLabel() }}</p>
            <ul class="result-list">
              @for (result of store.results(); track result.id) {
                <li class="list-row">
                  <div>
                    <strong>{{ result.title }}</strong>
                    <div class="list-row__meta">
                      {{ result.subtitle || "No additional details" }}
                    </div>
                  </div>
                  <a class="primary-link" [routerLink]="resolveResultLink(result.type)">
                    Open
                  </a>
                </li>
              }
            </ul>
          }
          @default {
            <p class="screen__sub">Run a search to start navigating faster.</p>
          }
        }
      </section>
    </section>
  `,
  styles: [
    `
      .result-list {
        margin: 0;
        padding: 0;
        list-style: none;
        display: grid;
        gap: 0.65rem;
      }
    `,
  ],
})
export class SearchPageComponent implements OnInit {
  readonly store = inject(SearchStore);

  readonly queryLabel = computed(() => this.store.query() || "sleep");
  readonly resultsLabel = computed(() => {
    const count = this.store.results().length;
    return `${count} result${count === 1 ? "" : "s"} for "${this.queryLabel()}".`;
  });
  readonly canSubmit = computed(() => this.store.query().trim().length >= 2);

  ngOnInit(): void {
    void this.store.runSearch("sleep");
  }

  onQueryInput(event: Event): void {
    const target = event.target as HTMLInputElement;
    this.store.updateQuery(target.value);
  }

  onSubmit(event: Event): void {
    event.preventDefault();
    if (!this.canSubmit()) {
      return;
    }
    void this.store.runSearch(this.store.query());
  }

  resolveResultLink(type: string): string {
    const linkMap: Record<string, string> = {
      coach: "/coach",
      alarm: "/alarm",
      routine: "/routine",
      challenge: "/challenges",
      device: "/device",
      recording: "/dashboard",
      user: "/account",
      account: "/account",
      unknown: "/dashboard",
    };
    return linkMap[type] ?? "/dashboard";
  }
}
