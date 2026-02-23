import { Component, Input } from "@angular/core";
import { formatShortDate, formatShortTime } from "../../../core/utils/format";

@Component({
  selector: "app-morning-header",
  standalone: true,
  template: `
    <header class="sleep-summary__header">
      <div class="sleep-summary__greeting">
        <span class="sleep-summary__date">{{ formattedDate }}</span>
        <span class="sleep-summary__title">Good morning</span>
      </div>
      @if (syncLabel) {
        <span class="ui-pill" aria-live="polite">{{ syncLabel }}</span>
      }
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
}
