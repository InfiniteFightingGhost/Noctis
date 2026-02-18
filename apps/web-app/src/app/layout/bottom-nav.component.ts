import { Component } from "@angular/core";
import { RouterLink, RouterLinkActive } from "@angular/router";

@Component({
  selector: "app-bottom-nav",
  standalone: true,
  imports: [RouterLink, RouterLinkActive],
  template: `
    <nav class="bottom-nav" aria-label="Primary">
      <div class="bottom-nav__inner">
        <a
          class="bottom-nav__link"
          routerLink="/sleep-summary"
          routerLinkActive="active"
          ariaCurrentWhenActive="page"
        >
          <span>Morning</span>
        </a>
        <a
          class="bottom-nav__link"
          routerLink="/"
          routerLinkActive="active"
          ariaCurrentWhenActive="page"
        >
          <span>Home</span>
        </a>
        <a
          class="bottom-nav__link"
          routerLink="/account"
          routerLinkActive="active"
          ariaCurrentWhenActive="page"
        >
          <span>Account</span>
        </a>
      </div>
    </nav>
  `,
})
export class BottomNavComponent {}
