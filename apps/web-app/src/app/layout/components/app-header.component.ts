import { Component } from "@angular/core";
import { RouterLink, RouterLinkActive } from "@angular/router";
import { UiIconComponent } from "../../shared/ui/icon/icon.component";

@Component({
  selector: "app-header",
  standalone: true,
  imports: [RouterLink, RouterLinkActive, UiIconComponent],
  template: `
    <header class="app-header">
      <div class="app-header__inner">
        <span class="app-header__brand">Noctis</span>
        <nav class="app-header__nav">
          <a
            routerLink="/dashboard"
            routerLinkActive="active"
            ariaCurrentWhenActive="page"
          >
            <ui-icon name="dashboard" />
            Dashboard
          </a>
          <a
            routerLink="/report"
            routerLinkActive="active"
            ariaCurrentWhenActive="page"
          >
            <ui-icon name="report" />
            Report
          </a>
          <a
            routerLink="/coach"
            routerLinkActive="active"
            ariaCurrentWhenActive="page"
          >
            <ui-icon name="coach" />
            Coach
          </a>
          <a
            routerLink="/alarm"
            routerLinkActive="active"
            ariaCurrentWhenActive="page"
          >
            <ui-icon name="alarm" />
            Alarm
          </a>
          <a
            routerLink="/device"
            routerLinkActive="active"
            ariaCurrentWhenActive="page"
          >
            <ui-icon name="device" />
            Device
          </a>
        </nav>
      </div>
    </header>
  `,
})
export class AppHeaderComponent {}
