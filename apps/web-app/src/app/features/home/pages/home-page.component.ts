import { Component, OnInit, computed, inject } from "@angular/core";
import { HomeStore } from "../data/home.store";

@Component({
  selector: "app-home-page",
  standalone: true,
  template: `
    <section class="page">
      <div class="page__intro">
        <p class="eyebrow">Welcome</p>
        <h1>{{ headline() }}</h1>
        <p class="lede">
          @switch (store.status()) {
            @case ("loading") {
              Loading your home overview...
            }
            @case ("error") {
              {{ store.errorMessage() ?? "Unable to load home overview." }}
            }
            @default {
              {{ lede() }}
            }
          }
        </p>
      </div>
    </section>
  `,
})
export class HomePageComponent implements OnInit {
  readonly store = inject(HomeStore);

  readonly headline = computed(
    () => this.store.overview()?.headline ?? "Home",
  );
  readonly lede = computed(
    () =>
      this.store.overview()?.lede ??
      "This is a starter home page. Replace this content with your real experience.",
  );

  ngOnInit(): void {
    void this.store.loadOverview();
  }
}
