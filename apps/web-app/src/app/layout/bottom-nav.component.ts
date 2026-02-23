import { Component } from "@angular/core";
import { RouterLink, RouterLinkActive } from "@angular/router";
import { UiIconComponent } from "../shared/ui/icon/icon.component";

@Component({
  selector: "app-bottom-nav",
  standalone: true,
  imports: [RouterLink, RouterLinkActive, UiIconComponent],
  template: `
    <nav class="bottom-nav" aria-label="Primary">
      <div class="bottom-nav__inner">
        <a
          class="bottom-nav__link"
          routerLink="/dashboard"
          routerLinkActive="active"
          ariaCurrentWhenActive="page"
        >
          <ui-icon name="dashboard" />
          <span>Dashboard</span>
        </a>
        <a
          class="bottom-nav__link"
          routerLink="/report"
          routerLinkActive="active"
          ariaCurrentWhenActive="page"
        >
          <ui-icon name="report" />
          <span>Report</span>
        </a>
        <a
          class="bottom-nav__link"
          routerLink="/coach"
          routerLinkActive="active"
          ariaCurrentWhenActive="page"
        >
          <ui-icon name="coach" />
          <span>Coach</span>
        </a>
        <a
          class="bottom-nav__link"
          routerLink="/alarm"
          routerLinkActive="active"
          ariaCurrentWhenActive="page"
        >
          <ui-icon name="alarm" />
          <span>Alarm</span>
        </a>
        <a
          class="bottom-nav__link"
          routerLink="/device"
          routerLinkActive="active"
          ariaCurrentWhenActive="page"
        >
          <ui-icon name="device" />
          <span>Device</span>
        </a>
        <a
          class="bottom-nav__link"
          routerLink="/account"
          routerLinkActive="active"
          ariaCurrentWhenActive="page"
        >
          <ui-icon name="account" />
          <span>Account</span>
        </a>
      </div>
    </nav>
  `,
})
export class BottomNavComponent {}
