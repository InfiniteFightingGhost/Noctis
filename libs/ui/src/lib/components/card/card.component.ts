import { Component } from "@angular/core";

@Component({
  selector: "ui-card",
  standalone: true,
  template: `
    <div class="ui-card">
      <ng-content />
    </div>
  `,
})
export class CardComponent {}
