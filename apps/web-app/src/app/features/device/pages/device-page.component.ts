import { Component, OnInit, computed, inject } from "@angular/core";
import { RouterLink } from "@angular/router";
import { UiButtonComponent } from "../../../shared/ui/button/button.component";
import { UiSkeletonComponent } from "../../../shared/ui/skeleton/skeleton.component";
import { StatusBannerComponent } from "../../../shared/ui/status-banner/status-banner.component";
import { StatePanelComponent } from "../../../shared/components/state-panel.component";
import { DeviceStore } from "../data/device.store";

@Component({
  selector: "app-device-page",
  standalone: true,
  imports: [
    RouterLink,
    UiButtonComponent,
    UiSkeletonComponent,
    StatusBannerComponent,
    StatePanelComponent,
  ],
  template: `
    <section class="screen screen--dark sleep-summary-theme">
      @switch (viewState()) {
        @case ("loading") {
          <div class="screen__section">
            <ui-skeleton [height]="24" />
            <ui-skeleton [height]="160" />
          </div>
        }
        @case ("no-data") {
          <app-state-panel
            title="No device connected"
            message="Add a device to sync sleep data."
            actionLabel="Add device"
            (action)="reload()"
          />
        }
        @case ("syncing") {
          <app-state-panel
            title="Syncing device"
            message="Keep the app open while we sync."
            actionLabel="View progress"
            (action)="reload()"
          />
        }
        @case ("error") {
          <app-state-panel
            title="Couldn't connect"
            [message]="store.errorMessage() ?? 'Check Bluetooth or Wi‑Fi, then retry.'"
            actionLabel="Retry"
            (action)="reload()"
          />
        }
        @default {
          <div class="screen__header">
            <div>
              <p class="screen__sub">Device Sync</p>
              <h1 class="screen__title">{{ primaryDeviceName() }}</h1>
            </div>
            <a class="primary-link" routerLink="/dashboard">Dashboard</a>
          </div>

          <div class="screen__section">
            <ui-status-banner
              variant="syncing"
              title="Syncing..."
              message="Last sync 07:42. Keep app open."
            />

            <div class="list-row">
              <div>
                <strong>{{ primaryDeviceName() }}</strong>
                <div class="list-row__meta">Connected • 82% battery</div>
              </div>
              <span class="list-row__meta">Synced</span>
            </div>
            <div class="list-row">
              <div>
                <strong>{{ secondaryDeviceName() }}</strong>
                <div class="list-row__meta">Nearby • 64% battery</div>
              </div>
              <span class="list-row__meta">Idle</span>
            </div>

            <div class="chart-card">
              <p class="chart-card__summary">
                Smart reliability: High (based on signal stability and movement noise)
              </p>
              <div class="stage-viz__bar" aria-hidden="true"></div>
            </div>

            <div class="screen__cta">
              <ui-button>Retry Sync</ui-button>
              <a class="primary-link" routerLink="/device/help">
                Bluetooth / Wi‑Fi help
              </a>
            </div>
          </div>
        }
      }
    </section>
  `,
})
export class DevicePageComponent implements OnInit {
  readonly store = inject(DeviceStore);
  readonly viewState = this.store.status;

  readonly primaryDeviceName = computed(
    () => this.store.primaryDevice()?.name ?? "Sleep Band",
  );
  readonly secondaryDeviceName = computed(
    () => this.store.secondaryDevice()?.name ?? "Bedside Dock",
  );

  ngOnInit(): void {
    void this.store.loadDevices();
  }

  reload(): void {
    void this.store.loadDevices();
  }
}
