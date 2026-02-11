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
          <a routerLink="/" routerLinkActive="active" ariaCurrentWhenActive="page">
            Home
          </a>
          <a
            routerLink="/account"
            routerLinkActive="active"
            ariaCurrentWhenActive="page"
          >
            Account
          </a>
          <a
            routerLink="/search"
            routerLinkActive="active"
            ariaCurrentWhenActive="page"
          >
            Search
          </a>
        </nav>
      </div>
    </header>
  `,
})
export class AppHeaderComponent {}
