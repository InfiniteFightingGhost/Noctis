import { Component, Input } from "@angular/core";

export type StatusChipVariant = "ok" | "syncing" | "error" | "partial";

@Component({
  selector: "ui-status-chip",
  standalone: true,
  template: `
    <span class="status-chip status-chip--{{ variant }}" aria-live="polite">
      {{ label }}
    </span>
  `,
})
export class StatusChipComponent {
  @Input() label = "";
  @Input() variant: StatusChipVariant = "ok";
}
