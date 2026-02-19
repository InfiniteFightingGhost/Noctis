import { Component, ElementRef, ViewChild } from "@angular/core";

@Component({
  selector: "ui-sheet",
  standalone: true,
  template: `
    <dialog class="ui-sheet" #dialog>
      <ng-content></ng-content>
    </dialog>
  `,
})
export class UiSheetComponent {
  @ViewChild("dialog") dialog?: ElementRef<HTMLDialogElement>;

  open(): void {
    this.dialog?.nativeElement?.showModal();
  }

  close(): void {
    this.dialog?.nativeElement?.close();
  }
}
