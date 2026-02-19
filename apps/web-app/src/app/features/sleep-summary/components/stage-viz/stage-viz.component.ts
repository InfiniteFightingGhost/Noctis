import { Component, Input, signal } from "@angular/core";
import { NgFor, NgIf } from "@angular/common";
import { StageBin, StagePct } from "../../api/sleep-summary.types";
import { findStageAtMinute, mapXToMinute } from "./stage-viz.utils";
import { STAGE_COLORS, stageLabel } from "./stage-colors";

@Component({
  selector: "app-stage-viz",
  standalone: true,
  imports: [NgFor, NgIf],
  template: `
    <section class="sleep-summary__stage">
      <div class="stage-viz__bar">
        <button
          class="stage-viz__button"
          type="button"
          (click)="onChartClick($event)"
          [attr.aria-label]="ariaLabel"
        >
          <svg viewBox="0 0 100 20" preserveAspectRatio="none">
            @for (bin of bins; track bin.startMinFromBedtime) {
              <rect
                [attr.x]="segmentX(bin)"
                y="0"
                [attr.width]="segmentWidth(bin)"
                height="20"
                [attr.fill]="stageColor(bin.stage)"
              />
            }
          </svg>
        </button>
      </div>
      @if (tooltip()) {
        <div class="stage-viz__tooltip">{{ tooltip() }}</div>
      }
      <p class="stage-viz__legend" aria-hidden="true">
        <span>Awake {{ pct.awake }}%</span>
        <span>Light {{ pct.light }}%</span>
        <span>Deep {{ pct.deep }}%</span>
        <span>REM {{ pct.rem }}%</span>
      </p>
      <p class="sr-only">{{ accessibilitySummary }}</p>
    </section>
  `,
})
export class StageVizComponent {
  @Input({ required: true }) bins: StageBin[] = [];
  @Input({ required: true }) pct: StagePct = {
    awake: 0,
    light: 0,
    deep: 0,
    rem: 0,
  };
  @Input({ required: true }) timeInBedMin = 0;
  @Input({ required: true }) bedtimeLocal = "";

  readonly tooltip = signal<string | null>(null);

  get accessibilitySummary(): string {
    return `Awake ${this.pct.awake}%, Light ${this.pct.light}%, Deep ${this.pct.deep}%, REM ${this.pct.rem}%.`;
  }

  get ariaLabel(): string {
    return "Sleep stage timeline. Tap for time detail.";
  }

  segmentWidth(bin: StageBin): number {
    if (!this.timeInBedMin) {
      return 0;
    }

    return (bin.durationMin / this.timeInBedMin) * 100;
  }

  segmentX(bin: StageBin): number {
    if (!this.timeInBedMin) {
      return 0;
    }

    return (bin.startMinFromBedtime / this.timeInBedMin) * 100;
  }

  stageColor(stage: StageBin["stage"]): string {
    return STAGE_COLORS[stage];
  }

  onChartClick(event: MouseEvent): void {
    const target = event.currentTarget as HTMLElement | null;
    if (!target) {
      return;
    }

    const rect = target.getBoundingClientRect();
    const offset = event.clientX - rect.left;
    const minute = mapXToMinute(offset, rect.width, this.timeInBedMin);
    const stage = findStageAtMinute(this.bins, minute);
    const timeLabel = this.formatTimeAtMinute(minute);

    if (stage) {
      this.tooltip.set(`${timeLabel} - ${stageLabel(stage)}`);
    }
  }

  private formatTimeAtMinute(minute: number): string {
    if (!this.bedtimeLocal) {
      return "--";
    }

    const base = new Date(this.bedtimeLocal);
    if (Number.isNaN(base.valueOf())) {
      return "--";
    }

    base.setMinutes(base.getMinutes() + minute);
    return new Intl.DateTimeFormat("en-US", {
      hour: "2-digit",
      minute: "2-digit",
    }).format(base);
  }
}
