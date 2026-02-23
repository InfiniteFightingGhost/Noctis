import { Component, Input, ViewChild } from "@angular/core";
import { SleepSummary } from "../api/sleep-summary.types";
import { ScoreBreakdownSheetComponent } from "./score-breakdown-sheet.component";

@Component({
  selector: "app-score-card",
  standalone: true,
  imports: [ScoreBreakdownSheetComponent],
  template: `
    <button
      class="sleep-summary__score"
      type="button"
      (click)="openSheet()"
      aria-label="Open sleep score breakdown"
    >
      <div>
        <div class="sleep-summary__score-value">{{ summary?.score ?? "--" }}</div>
        <div class="sleep-summary__score-label">
          {{ summary?.scoreLabel ?? "--" }}
        </div>
        <div class="sleep-summary__score-meta">Last night</div>
      </div>
      <div class="sleep-summary__score-hint">Tap for details</div>
    </button>
    <app-score-breakdown-sheet #sheet [summary]="summary" />
  `,
})
export class ScoreCardComponent {
  @Input() summary: SleepSummary | null = null;
  @ViewChild("sheet") sheet?: ScoreBreakdownSheetComponent;

  openSheet(): void {
    this.sheet?.open();
  }
}
