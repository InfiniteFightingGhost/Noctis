import { Component, Input } from "@angular/core";

@Component({
  selector: "ui-icon",
  standalone: true,
  template: `
    <span class="ui-icon" [attr.data-name]="name" aria-hidden="true"></span>
  `,
})
export class UiIconComponent {
  @Input() name = "";
}
