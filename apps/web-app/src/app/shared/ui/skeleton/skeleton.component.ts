import { Component, Input } from "@angular/core";

@Component({
  selector: "ui-skeleton",
  standalone: true,
  template: `
    <div class="ui-skeleton" [style.height.px]="height"></div>
  `,
})
export class UiSkeletonComponent {
  @Input() height = 16;
}
