import { Component, Input } from "@angular/core";

@Component({
  selector: "ui-icon",
  standalone: true,
  template: `
    <span class="ui-icon" [attr.data-name]="name" aria-hidden="true">
      @switch (name) {
        @case ("dashboard") {
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <rect x="3" y="3" width="8" height="8" rx="2" />
            <rect x="13" y="3" width="8" height="5" rx="2" />
            <rect x="13" y="10" width="8" height="11" rx="2" />
            <rect x="3" y="13" width="8" height="8" rx="2" />
          </svg>
        }
        @case ("report") {
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M6 3h9l5 5v13H6z" />
            <path d="M15 3v6h6" />
            <path d="M9 13h6" />
            <path d="M9 17h6" />
          </svg>
        }
        @case ("coach") {
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <circle cx="12" cy="8" r="4" />
            <path d="M5 21a7 7 0 0 1 14 0" />
          </svg>
        }
        @case ("alarm") {
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <circle cx="12" cy="13" r="7" />
            <path d="M12 13V9" />
            <path d="M12 13l3 2" />
            <path d="M5 5 3 7" />
            <path d="M19 5l2 2" />
          </svg>
        }
        @case ("device") {
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <rect x="7" y="2" width="10" height="20" rx="2" />
            <path d="M10 6h4" />
            <circle cx="12" cy="18" r="1" />
          </svg>
        }
        @case ("account") {
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <circle cx="12" cy="8" r="4" />
            <path d="M5 21a7 7 0 0 1 14 0" />
          </svg>
        }
        @default {
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <circle cx="12" cy="12" r="8" />
          </svg>
        }
      }
    </span>
  `,
  styles: [
    `
      .ui-icon {
        width: 18px;
        height: 18px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
      }

      .ui-icon svg {
        width: 100%;
        height: 100%;
      }
    `,
  ],
})
export class UiIconComponent {
  @Input() name = "";
}
