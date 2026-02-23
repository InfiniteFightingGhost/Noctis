import { Component, Input } from "@angular/core";

@Component({
  selector: "ui-button",
  standalone: true,
  template: `
    <button
      class="ui-button ui-button--{{ variant }}"
      [attr.type]="type"
      [disabled]="disabled"
    >
      <ng-content></ng-content>
    </button>
  `,
})
export class UiButtonComponent {
  @Input() type: "button" | "submit" | "reset" = "button";
  @Input() variant: "primary" | "ghost" = "primary";
  @Input() disabled = false;
}
