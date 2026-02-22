import { Component, EventEmitter, Input, Output } from "@angular/core";
import { UiButtonComponent } from "../button/button.component";

export type StatusBannerVariant = "ok" | "syncing" | "error" | "partial";

@Component({
  selector: "ui-status-banner",
  standalone: true,
  imports: [UiButtonComponent],
  template: `
    <section class="status-banner status-banner--{{ variant }}" role="status">
      <div class="status-banner__content">
        <strong>{{ title }}</strong>
        <p>{{ message }}</p>
      </div>
      @if (actionLabel) {
        <ui-button variant="ghost" (click)="action.emit()">
          {{ actionLabel }}
        </ui-button>
      }
    </section>
  `,
})
export class StatusBannerComponent {
  @Input() title = "";
  @Input() message = "";
  @Input() actionLabel?: string;
  @Input() variant: StatusBannerVariant = "ok";
  @Output() action = new EventEmitter<void>();
}
