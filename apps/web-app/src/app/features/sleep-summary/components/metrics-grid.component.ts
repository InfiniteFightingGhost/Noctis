import { Component, Input } from "@angular/core";
import { SleepSummary } from "../api/sleep-summary.types";
import { formatMinutesAsClock, formatPct } from "../../../core/utils/format";

@Component({
  selector: "app-metrics-grid",
  standalone: true,
  template: `
    <div class="metrics-grid">
      <div class="metrics-grid__card">
        <div class="metrics-grid__label">Total Sleep</div>
        <div class="metrics-grid__value">
          {{ formatMinutesAsClock(summary?.totals.totalSleepMin) }}
        </div>
      </div>
      <div class="metrics-grid__card">
        <div class="metrics-grid__label">Deep %</div>
        <div class="metrics-grid__value">
          {{ formatPct(summary?.metrics.deepPct) }}
        </div>
      </div>
      <div class="metrics-grid__card">
        <div class="metrics-grid__label">Avg HR</div>
        <div class="metrics-grid__value">
          {{ summary?.metrics.avgHrBpm ?? "--" }} bpm
        </div>
      </div>
      <div class="metrics-grid__card">
        <div class="metrics-grid__label">Movement %</div>
        <div class="metrics-grid__value">
          {{ formatPct(summary?.metrics.movementPct) }}
        </div>
      </div>
    </div>
  `,
})
export class MetricsGridComponent {
  @Input() summary: SleepSummary | null = null;

  formatMinutesAsClock = formatMinutesAsClock;
  formatPct = formatPct;
}
