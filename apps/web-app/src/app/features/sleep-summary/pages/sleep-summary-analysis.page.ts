import { Component } from "@angular/core";

@Component({
  selector: "app-sleep-summary-analysis-page",
  standalone: true,
  template: `
    <section class="sleep-summary sleep-summary-theme">
      <div class="sleep-summary__content">
        <div class="sleep-summary__greeting">
          <span class="sleep-summary__date">Full Analysis</span>
          <span class="sleep-summary__title">Deep Dive</span>
        </div>
        <article class="insight-card">
          <p class="insight-card__text">
            Coming soon. We'll break down your sleep trends and long-term patterns.
          </p>
        </article>
        <div class="sleep-summary__cta">
          <button class="sleep-summary__button" type="button">
            Notify Me
          </button>
        </div>
      </div>
    </section>
  `,
})
export class SleepSummaryAnalysisPageComponent {}
