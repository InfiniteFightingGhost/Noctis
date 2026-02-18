import { Component } from "@angular/core";

@Component({
  selector: "app-loading-state",
  standalone: true,
  template: `
    <div class="sleep-summary__states sleep-summary__skeleton">
      <div class="skeleton-block"></div>
      <div class="skeleton-block skeleton-block--score"></div>
      <div class="skeleton-block"></div>
      <div class="skeleton-block skeleton-block--tall"></div>
      <div class="skeleton-block"></div>
      <div class="skeleton-block"></div>
    </div>
  `,
})
export class LoadingStateComponent {}
