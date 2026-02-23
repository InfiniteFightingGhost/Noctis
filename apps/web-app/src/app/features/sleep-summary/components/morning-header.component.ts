import { Component, Input } from "@angular/core";
import { formatShortDate, formatShortTime } from "../../../core/utils/format";
import { StatusChipComponent, StatusChipVariant } from "../../../shared/ui/status-chip/status-chip.component";

@Component({
  selector: "app-morning-header",
  standalone: true,
  imports: [StatusChipComponent],
  template: `
    <header class="sleep-summary__header">
      <div class="sleep-summary__greeting">
        <span class="sleep-summary__title">Good morning</span>
        <span class="sleep-summary__sub">Morning summary</span>
      </div>
      <div class="sleep-summary__date-block">
        <span class="sleep-summary__date">{{ formattedDate }}</span>
        @if (syncLabel) {
          <ui-status-chip
            [variant]="syncVariant"
            [label]="syncLabel"
          />
        }
      </div>
    </header>
  `,
})
export class MorningHeaderComponent {
  @Input({ required: true }) dateLocal = "";
  @Input() syncStatus: "ok" | "syncing" | "error" | null = null;
  @Input() lastSyncAtLocal: string | null = null;

  get formattedDate(): string {
    return formatShortDate(this.dateLocal);
  }

  get syncLabel(): string | null {
    if (this.syncStatus === "syncing") {
      return "Syncing...";
    }

    if (this.syncStatus === "error") {
      return "Sync issue";
    }

    if (this.lastSyncAtLocal) {
      return `Last sync ${formatShortTime(this.lastSyncAtLocal)}`;
    }

    return null;
  }

  get syncVariant(): StatusChipVariant {
    if (this.syncStatus === "syncing") {
      return "syncing";
    }

    if (this.syncStatus === "error") {
      return "error";
    }

    return "ok";
  }
}
