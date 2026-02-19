import { Component, EventEmitter, Input, Output } from "@angular/core";

@Component({
  selector: "app-sync-error-state",
  standalone: true,
  template: `
    <div class="sleep-summary__states">
      <h3>{{ title }}</h3>
      <p>{{ description }}</p>
      <button class="sleep-summary__button" type="button" (click)="retry.emit()">
        {{ actionLabel }}
      </button>
    </div>
  `,
})
export class SyncErrorStateComponent {
  @Input() state: "syncing" | "error" = "error";
  @Input() actionLabel = "Retry";
  @Output() retry = new EventEmitter<void>();

  get title(): string {
    return this.state === "syncing" ? "Sync in progress" : "Sync issue";
  }

  get description(): string {
    return this.state === "syncing"
      ? "We're pulling in your sleep data. Check back soon."
      : "We couldn't update your sleep yet.";
  }
}
