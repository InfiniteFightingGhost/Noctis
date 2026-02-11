import { NgIf } from "@angular/common";
import { Component, Input } from "@angular/core";

@Component({
  selector: "ui-button",
  standalone: true,
  imports: [NgIf],
  template: `
    <button class="ui-button" [attr.type]="type">
      <ng-content />
      <span *ngIf="label">{{ label }}</span>
    </button>
  `,
})
export class ButtonComponent {
  @Input() label = "";
  @Input() type: "button" | "submit" | "reset" = "button";
}
