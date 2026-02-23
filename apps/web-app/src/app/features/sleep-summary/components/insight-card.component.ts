import { Component, Input } from "@angular/core";
import { RouterLink } from "@angular/router";
import { SleepSummary } from "../api/sleep-summary.types";

@Component({
  selector: "app-insight-card",
  standalone: true,
  imports: [RouterLink],
  template: `
    <article class="insight-card">
      <p class="insight-card__text">
        {{ summary?.insight?.text ?? "No insight available yet." }}
      </p>
      @if (showWhy) {
        <a class="primary-link" [routerLink]="whyLink">Why this?</a>
      }
    </article>
  `,
})
export class InsightCardComponent {
  @Input() summary: SleepSummary | null = null;
  @Input() showWhy = true;
  @Input() whyLink = "/coach";
}
