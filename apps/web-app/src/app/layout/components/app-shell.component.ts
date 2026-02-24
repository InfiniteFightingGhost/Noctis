import { Component, DestroyRef, inject, signal } from "@angular/core";
import { NavigationEnd, Router, RouterOutlet } from "@angular/router";
import { filter } from "rxjs";
import { takeUntilDestroyed } from "@angular/core/rxjs-interop";
import { BottomNavComponent } from "../bottom-nav.component";
import { AppHeaderComponent } from "./app-header.component";

@Component({
  selector: "app-shell",
  standalone: true,
  imports: [AppHeaderComponent, BottomNavComponent, RouterOutlet],
  template: `
    <div class="app-shell">
      @if (showHeader()) {
        <app-header />
      }
      <main class="app-shell__content">
        <router-outlet />
      </main>
      @if (showBottomNav()) {
        <app-bottom-nav />
      }
    </div>
  `,
})
export class AppShellComponent {
  private readonly router = inject(Router);
  private readonly destroyRef = inject(DestroyRef);
  readonly showHeader = signal(true);
  readonly showBottomNav = signal(true);

  constructor() {
    this.updateHeader(this.router.url);
    this.router.events
      .pipe(
        filter((event): event is NavigationEnd => event instanceof NavigationEnd),
        takeUntilDestroyed(this.destroyRef),
      )
      .subscribe((event) => this.updateHeader(event.urlAfterRedirects));
  }

  private updateHeader(url: string): void {
    if (url.startsWith("/login") || url.startsWith("/signup")) {
      this.showHeader.set(false);
      this.showBottomNav.set(false);
      return;
    }

    const hiddenPrefixes = [
      "/dashboard",
      "/report",
      "/coach",
      "/alarm",
      "/routine",
      "/challenges",
      "/device",
      "/sleep-summary",
    ];

    this.showHeader.set(!hiddenPrefixes.some((prefix) => url.startsWith(prefix)));
    this.showBottomNav.set(true);
  }
}
