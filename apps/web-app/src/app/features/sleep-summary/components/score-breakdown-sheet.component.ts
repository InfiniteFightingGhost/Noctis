import { Component, ElementRef, Input, ViewChild } from "@angular/core";
import { formatMinutesAsClock, formatPct, formatShortTime } from "../../../core/utils/format";
import { SleepSummary } from "../api/sleep-summary.types";

@Component({
  selector: "app-score-breakdown-sheet",
  standalone: true,
  template: `
    <dialog class="sleep-summary__sheet" #dialog>
      <div class="sleep-summary__sheet-inner">
        <header class="sleep-summary__sheet-header">
          <div>
            <h3>Sleep Score Breakdown</h3>
            <p>
              {{ summary?.scoreLabel ?? "--" }} night with
              {{ formatMinutesAsClock(summary?.totals.totalSleepMin) }} asleep.
            </p>
          </div>
          <button type="button" (click)="close()">Close</button>
        </header>
        <div class="sleep-summary__sheet-grid">
          <div>
            <span>Bedtime</span>
            <strong>{{ formatShortTime(summary?.bedtimeLocal) }}</strong>
          </div>
          <div>
            <span>Wake</span>
            <strong>{{ formatShortTime(summary?.waketimeLocal) }}</strong>
          </div>
          <div>
            <span>Efficiency</span>
            <strong>{{ formatPct(summary?.totals.sleepEfficiencyPct) }}</strong>
          </div>
          <div>
            <span>Deep</span>
            <strong>{{ formatPct(summary?.metrics.deepPct) }}</strong>
          </div>
        </div>
      </div>
    </dialog>
  `,
})
export class ScoreBreakdownSheetComponent {
  @Input() summary: SleepSummary | null = null;
  @ViewChild("dialog") dialog?: ElementRef<HTMLDialogElement>;

  open(): void {
    this.dialog?.nativeElement?.showModal();
  }

  close(): void {
    this.dialog?.nativeElement?.close();
  }

  formatShortTime = formatShortTime;
  formatMinutesAsClock = formatMinutesAsClock;
  formatPct = formatPct;
}
