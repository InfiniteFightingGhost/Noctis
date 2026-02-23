import { Component } from "@angular/core";
import { RouterLink } from "@angular/router";

@Component({
  selector: "app-device-help-page",
  standalone: true,
  imports: [RouterLink],
  template: `
    <section class="screen screen--dark sleep-summary-theme">
      <div class="screen__header">
        <div>
          <p class="screen__sub">Connection help</p>
          <h1 class="screen__title">Troubleshooting</h1>
        </div>
        <a class="primary-link" routerLink="/device">Back</a>
      </div>
      <div class="screen__section">
        <div class="state-panel">
          <h3>Bluetooth + Wi‑Fi tips</h3>
          <p>Make sure Bluetooth is enabled and the device is charged.</p>
        </div>
      </div>
    </section>
  `,
})
export class DeviceHelpPageComponent {}
