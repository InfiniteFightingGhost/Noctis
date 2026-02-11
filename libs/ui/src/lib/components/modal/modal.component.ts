import { NgIf } from "@angular/common";
import { Component, EventEmitter, Input, Output } from "@angular/core";

@Component({
  selector: "ui-modal",
  standalone: true,
  imports: [NgIf],
  template: `
    <div class="ui-modal" *ngIf="open" role="dialog" aria-modal="true">
      <div class="ui-modal__backdrop" (click)="close.emit()"></div>
      <div class="ui-modal__panel">
        <ng-content />
      </div>
    </div>
  `,
})
export class ModalComponent {
  @Input() open = false;
  @Output() close = new EventEmitter<void>();
}
