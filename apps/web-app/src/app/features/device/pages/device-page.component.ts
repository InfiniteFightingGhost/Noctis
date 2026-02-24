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
              variant="ok"
              [title]="syncBannerTitle()"
              [message]="syncBannerMessage()"
            />

            @for (device of devices(); track device.id) {
              <div class="list-row">
                <div>
                  <strong>{{ device.name }}</strong>
                  <div class="list-row__meta">
                    {{ device.external_id ? "External ID " + device.external_id : "No external ID" }}
                  </div>
                </div>
                <span class="list-row__meta">
                  {{ device.user_id ? "Linked" : "Unlinked" }}
                </span>
              </div>
            }

            <div class="chart-card">
              <p class="chart-card__summary">
                {{ syncBannerMessage() }}
              </p>
              <p class="chart-card__summary">{{ deviceCountLabel() }}</p>
            </div>

            <div class="screen__cta">
              <ui-button (click)="reload()">Refresh devices</ui-button>
              <a class="primary-link" routerLink="/device/claim">
                Claim by external ID
              </a>
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
  readonly devices = this.store.devices;

  readonly primaryDeviceName = computed(
    () => this.store.primaryDevice()?.name ?? "Sleep Band",
  );
  readonly deviceCountLabel = computed(() => {
    const count = this.devices().length;
    return `${count} device${count === 1 ? "" : "s"} registered`;
  });
  readonly syncBannerTitle = computed(() => {
    if (this.store.isFetching()) {
      return "Refreshing devices";
    }
    return "Device registry up to date";
  });
  readonly syncBannerMessage = computed(() => {
    const primaryDevice = this.store.primaryDevice();
    if (!primaryDevice) {
      return "No device connected yet.";
    }
    return `Primary device: ${primaryDevice.name}`;
  });

  ngOnInit(): void {
    void this.store.loadDevices();
  }

  reload(): void {
    void this.store.loadDevices();
  }
}
