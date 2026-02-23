import { Component, OnInit, computed, inject } from "@angular/core";
import { SearchStore } from "../data/search.store";

@Component({
  selector: "app-search-page",
  standalone: true,
  template: `
    <section class="page">
      <div class="page__intro">
        <p class="eyebrow">Search</p>
        <h1>Search</h1>
        <p class="lede">
          @switch (store.status()) {
            @case ("loading") {
              Searching your workspace...
            }
            @case ("error") {
              {{ store.errorMessage() ?? "Unable to load search results." }}
            }
            @case ("no-data") {
              No results yet for "{{ queryLabel() }}".
            }
            @case ("success") {
              {{ resultsLabel() }}
            }
            @default {
              Start building out your search experience here.
            }
          }
        </p>
      </div>
    </section>
  `,
})
export class SearchPageComponent implements OnInit {
  readonly store = inject(SearchStore);

  readonly queryLabel = computed(() => this.store.query() || "sleep");
  readonly resultsLabel = computed(() => {
    const count = this.store.results().length;
    return `${count} result${count === 1 ? "" : "s"} for "${this.queryLabel()}".`;
  });

  ngOnInit(): void {
    void this.store.runSearch("sleep");
  }
}
