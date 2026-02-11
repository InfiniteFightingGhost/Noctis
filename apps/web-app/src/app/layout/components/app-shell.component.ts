import { Component } from "@angular/core";
import { RouterOutlet } from "@angular/router";
import { AppFooterComponent } from "./app-footer.component";
import { AppHeaderComponent } from "./app-header.component";

@Component({
  selector: "app-shell",
  standalone: true,
  imports: [AppHeaderComponent, AppFooterComponent, RouterOutlet],
  template: `
    <div class="app-shell">
      <app-header />
      <main class="app-shell__content">
        <router-outlet />
      </main>
      <app-footer />
    </div>
  `,
})
export class AppShellComponent {}
