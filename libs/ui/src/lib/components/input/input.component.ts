import { NgIf } from "@angular/common";
import { Component, Input } from "@angular/core";

@Component({
  selector: "ui-input",
  standalone: true,
  imports: [NgIf],
  template: `
    <label class="ui-input">
      <span *ngIf="label" class="ui-input__label">{{ label }}</span>
      <input
        class="ui-input__field"
        [attr.type]="type"
        [attr.placeholder]="placeholder"
      />
    </label>
  `,
})
export class InputComponent {
  @Input() label = "";
  @Input() placeholder = "";
  @Input() type: "text" | "email" | "password" | "search" = "text";
}
