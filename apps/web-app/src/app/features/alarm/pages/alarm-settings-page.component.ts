import { Component, OnInit, computed, inject } from "@angular/core";
import { RouterLink } from "@angular/router";
import { UiSkeletonComponent } from "../../../shared/ui/skeleton/skeleton.component";
import { StatusBannerComponent } from "../../../shared/ui/status-banner/status-banner.component";
import { StatePanelComponent } from "../../../shared/components/state-panel.component";
import { AlarmStore } from "../data/alarm.store";

@Component({
  selector: "app-alarm-settings-page",
  standalone: true,
  imports: [
    RouterLink,
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
            <ui-skeleton [height]="140" />
          </div>
        }
        @case ("no-data") {
          <app-state-panel
            title="No alarm configured"
            message="Set up your alarm to enable the routine."
            actionLabel="Set Alarm"
            (action)="reload()"
          />
        }
        @case ("syncing") {
          <app-state-panel
            title="Syncing alarm"
            message="Updating alarm reliability."
            actionLabel="View Device"
            (action)="reload()"
          />
        }
        @case ("error") {
          <app-state-panel
            title="Settings unavailable"
            [message]="store.errorMessage() ?? 'We could not load alarm settings.'"
            actionLabel="Retry"
            (action)="reload()"
          />
        }
        @default {
          <div class="screen__header">
            <div>
              <p class="screen__sub">Alarm settings</p>
              <h1 class="screen__title">Smart reliability</h1>
            </div>
            <a class="primary-link" routerLink="/alarm">Back</a>
          </div>

          <div class="screen__section">
            <ui-status-banner
              variant="ok"
              title="Smart reliability: High"
              message="Based on signal stability and motion noise."
              actionLabel="Show details"
            />

            <div class="chart-card">
              <p class="chart-card__summary">
                Wake time: {{ wakeTimeLabel() }} with {{ wakeWindowLabel() }}
              </p>
              <p class="chart-card__summary">
                Sunrise: {{ sunriseLabel() }}. Sound: {{ soundLabel() }}.
              </p>
            </div>

            <div class="screen__cta">
              <a class="sleep-summary__button" routerLink="/routine">
                Start Routine
              </a>
            </div>
          </div>
        }
      }
    </section>
  `,
})
export class AlarmSettingsPageComponent implements OnInit {
  readonly store = inject(AlarmStore);
  readonly viewState = this.store.status;
  readonly wakeTimeLabel = computed(
    () => this.store.settings()?.wake_time ?? "--:--",
  );
  readonly wakeWindowLabel = computed(() => {
    const windowMinutes = this.store.settings()?.wake_window_minutes;
    return windowMinutes ? `${windowMinutes} minute window` : "no wake window";
  });
  readonly sunriseLabel = computed(() => {
    const settings = this.store.settings();
    if (!settings) {
      return "Not configured";
    }
    return settings.sunrise_enabled
      ? `Enabled (intensity ${settings.sunrise_intensity})`
      : "Disabled";
  });
  readonly soundLabel = computed(
    () => this.store.activeSound()?.label ?? "Not selected",
  );

  ngOnInit(): void {
    void this.store.loadSettings();
  }

  reload(): void {
    void this.store.loadSettings();
  }
}
