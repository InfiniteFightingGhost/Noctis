import { Component, EventEmitter, Input, Output } from "@angular/core";

@Component({
  selector: "app-no-data-state",
  standalone: true,
  template: `
    <div class="sleep-summary__states">
      <h3>No sleep recorded</h3>
      <p>Sync your device to capture last night.</p>
      <button class="sleep-summary__button" type="button" (click)="action.emit()">
        {{ actionLabel }}
      </button>
    </div>
  `,
})
export class NoDataStateComponent {
  @Input() actionLabel = "Sync / Check device";
  @Output() action = new EventEmitter<void>();
}
