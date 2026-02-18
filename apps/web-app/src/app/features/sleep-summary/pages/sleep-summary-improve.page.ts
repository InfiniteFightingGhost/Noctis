import { Component } from "@angular/core";

@Component({
  selector: "app-sleep-summary-improve-page",
  standalone: true,
  template: `
    <section class="sleep-summary sleep-summary-theme">
      <div class="sleep-summary__content">
        <div class="sleep-summary__greeting">
          <span class="sleep-summary__date">Improve Tonight</span>
          <span class="sleep-summary__title">Tonight's Focus</span>
        </div>
        <article class="insight-card">
          <p class="insight-card__text">
            Start your wind-down 20 minutes earlier and dim lights after 9:30pm.
          </p>
        </article>
        <div class="sleep-summary__cta">
          <button class="sleep-summary__button" type="button">
            Set Wind-Down Reminder
          </button>
        </div>
      </div>
    </section>
  `,
})
export class SleepSummaryImprovePageComponent {}
