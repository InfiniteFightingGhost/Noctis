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
        <article class="state-panel" aria-labelledby="device-help-checklist-title">
          <h3 id="device-help-checklist-title">Before retrying sync</h3>
          <ol class="help-list">
            <li>Keep your phone within 2 meters of the sleep device.</li>
            <li>Confirm Bluetooth is on and Airplane Mode is off.</li>
            <li>Charge the device for at least 10 minutes if LED is red.</li>
            <li>Reopen the app, then use Refresh Devices on the device screen.</li>
          </ol>
        </article>

        <article class="state-panel" aria-labelledby="device-help-support-title">
          <h3 id="device-help-support-title">If issues continue</h3>
          <p>
            Remove the device in system Bluetooth settings, pair again, then run
            a new sync.
          </p>
          <a class="primary-link" routerLink="/device">Return to device sync</a>
        </article>
      </div>
    </section>
  `,
  styles: [
    `
      .help-list {
        margin: 0;
        padding-left: 1.1rem;
        display: grid;
        gap: 0.45rem;
        color: var(--text-1);
      }
    `,
  ],
})
export class DeviceHelpPageComponent {}
