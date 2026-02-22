import { Component, Input } from "@angular/core";
import { SleepSummary } from "../api/sleep-summary.types";

@Component({
  selector: "app-insight-card",
  standalone: true,
  template: `
    <article class="insight-card">
      <p class="insight-card__text">
        {{ summary?.insight.text ?? "No insight available yet." }}
      </p>
    </article>
  `,
})
export class InsightCardComponent {
  @Input() summary: SleepSummary | null = null;
}
