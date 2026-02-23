import { Component } from "@angular/core";
import { RouterLink, RouterLinkActive } from "@angular/router";

@Component({
  selector: "app-header",
  standalone: true,
  imports: [RouterLink, RouterLinkActive],
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
            Dashboard
          </a>
          <a
            routerLink="/report"
            routerLinkActive="active"
            ariaCurrentWhenActive="page"
          >
            Report
          </a>
          <a
            routerLink="/coach"
            routerLinkActive="active"
            ariaCurrentWhenActive="page"
          >
            Coach
          </a>
          <a
            routerLink="/alarm"
            routerLinkActive="active"
            ariaCurrentWhenActive="page"
          >
            Alarm
          </a>
          <a
            routerLink="/device"
            routerLinkActive="active"
            ariaCurrentWhenActive="page"
          >
            Device
          </a>
        </nav>
      </div>
    </header>
  `,
})
export class AppHeaderComponent {}
