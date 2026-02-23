import { Component, OnInit, computed, inject } from "@angular/core";
import { RouterLink } from "@angular/router";
import { RoutineStore } from "../data/routine.store";

@Component({
  selector: "app-routine-edit-page",
  standalone: true,
  imports: [RouterLink],
  template: `
    <section class="screen screen--dark sleep-summary-theme">
      <div class="screen__header">
        <div>
          <p class="screen__sub">Edit Routine</p>
          <h1 class="screen__title">{{ routineTitle() }}</h1>
        </div>
        <a class="primary-link" routerLink="/routine">Back</a>
      </div>
      <div class="screen__section">
        <div class="state-panel">
          <h3>Editing coming soon</h3>
          <p>You'll be able to reorder and add steps here.</p>
        </div>
      </div>
    </section>
  `,
})
export class RoutineEditPageComponent implements OnInit {
  readonly store = inject(RoutineStore);

  readonly routineTitle = computed(
    () => this.store.routine()?.title ?? "Customize steps",
  );

  ngOnInit(): void {
    void this.store.loadRoutine();
  }
}
