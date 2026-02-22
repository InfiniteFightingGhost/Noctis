import { Component, EventEmitter, Input, Output } from "@angular/core";
import { UiButtonComponent } from "../ui/button/button.component";

@Component({
  selector: "app-state-panel",
  standalone: true,
  imports: [UiButtonComponent],
  template: `
    <div class="state-panel" role="status">
      <div>
        <h3>{{ title }}</h3>
        <p>{{ message }}</p>
      </div>
      @if (actionLabel) {
        <ui-button (click)="action.emit()">{{ actionLabel }}</ui-button>
      }
    </div>
  `,
})
export class StatePanelComponent {
  @Input() title = "";
  @Input() message = "";
  @Input() actionLabel?: string;
  @Output() action = new EventEmitter<void>();
}
