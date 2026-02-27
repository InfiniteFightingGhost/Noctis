import { Component, OnInit, computed, inject } from "@angular/core";
import { RouterLink } from "@angular/router";
import { UiSkeletonComponent } from "../../../shared/ui/skeleton/skeleton.component";
import { StatusChipComponent } from "../../../shared/ui/status-chip/status-chip.component";
import { StatePanelComponent } from "../../../shared/components/state-panel.component";
import { AlarmStore } from "../data/alarm.store";

@Component({
  selector: "app-alarm-page",
  standalone: true,
  imports: [
    RouterLink,
    UiSkeletonComponent,
    StatusChipComponent,
    StatePanelComponent,
  ],
  template: `
    <section class="screen screen--dark sleep-summary-theme">
      @switch (viewState()) {
        @case ("loading") {
          <div class="screen__section">
            <ui-skeleton [height]="40" />
            <ui-skeleton [height]="160" />
            <ui-skeleton [height]="60" />
          </div>
        }
        @case ("no-data") {
          <app-state-panel
            title="No alarm set"
            message="Create your first smart alarm."
            actionLabel="Set Alarm"
            (action)="reload()"
          />
        }
        @case ("syncing") {
          <app-state-panel
            title="Syncing alarm"
            message="Updating your alarm settings."
            actionLabel="View Device"
            (action)="reload()"
          />
        }
        @case ("error") {
          <app-state-panel
            title="Alarm unavailable"
            [message]="store.errorMessage() ?? 'We could not load alarm settings.'"
            actionLabel="Retry"
            (action)="reload()"
          />
        }
        @default {
          <div class="screen__header">
            <div>
              <p class="screen__sub">Smart Alarm</p>
              <h1 class="screen__title">{{ wakeTimeLabel() }}</h1>
              <p class="screen__sub">Tomorrow morning</p>
            </div>
            <ui-status-chip
              [variant]="store.isSaving() ? 'syncing' : 'ok'"
              [label]="store.isSaving() ? 'Saving changes' : 'All changes saved'"
            />
          </div>

          <div class="screen__section">
            <div class="list-row">
              <div>
                <strong>Wake window</strong>
                <div class="list-row__meta">{{ wakeWindowLabel() }}</div>
              </div>
              <input
                type="range"
                min="10"
                max="45"
                [value]="wakeWindowMinutes()"
                aria-label="Wake window minutes"
                (change)="onWakeWindowChange($event)"
              />
            </div>

            <div class="list-row">
              <div>
                <strong>Sunrise</strong>
                <div class="list-row__meta">Gentle light up</div>
              </div>
              <input
                type="checkbox"
                [checked]="sunriseEnabled()"
                aria-label="Sunrise toggle"
                (change)="onSunriseToggle($event)"
              />
            </div>

            <div class="list-row">
              <div>
                <strong>Sunrise intensity</strong>
                <div class="list-row__meta">Medium</div>
              </div>
              <input
                type="range"
                min="1"
                max="5"
                [value]="sunriseIntensity()"
                aria-label="Sunrise intensity"
                (change)="onSunriseIntensityChange($event)"
              />
            </div>

            <div class="list-row">
              <div>
                <strong>Alarm sound</strong>
                <div class="list-row__meta">{{ soundLabel() }}</div>
              </div>
              <span class="list-row__meta">Selected</span>
            </div>

            @for (option of soundOptions(); track option.id) {
              <div class="list-row">
                <label>
                  <input
                    type="radio"
                    name="sound"
                    [checked]="option.id === selectedSoundId()"
                    [attr.aria-label]="option.label + ' sound'"
                    (change)="onSoundChange(option.id)"
                  />
                  {{ option.label }}
                </label>
                <span class="list-row__meta">{{ option.mood ?? "" }}</span>
              </div>
            }

            @if (store.errorMessage()) {
              <p class="form-error" role="alert">{{ store.errorMessage() }}</p>
            }

            <a class="primary-link" routerLink="/alarm/settings">
              Advanced settings
            </a>
          </div>
        }
      }
    </section>
  `,
})
export class AlarmPageComponent implements OnInit {
  readonly store = inject(AlarmStore);
  readonly viewState = this.store.status;

  readonly wakeTimeLabel = computed(
    () => this.store.settings()?.wake_time ?? "06:45",
  );
  readonly wakeWindowMinutes = computed(
    () => this.store.settings()?.wake_window_minutes ?? 20,
  );
  readonly wakeWindowLabel = computed(
    () => `${this.wakeWindowMinutes()} min window`,
  );
  readonly sunriseEnabled = computed(
    () => this.store.settings()?.sunrise_enabled ?? true,
  );
  readonly sunriseIntensity = computed(
    () => this.store.settings()?.sunrise_intensity ?? 3,
  );
  readonly soundLabel = computed(
    () => this.store.activeSound()?.label ?? "Ocean Drift",
  );
  readonly soundOptions = computed(
    () => this.store.settings()?.sound_options ?? [],
  );
  readonly selectedSoundId = computed(() => this.store.settings()?.sound_id ?? "");

  ngOnInit(): void {
    void this.store.loadSettings();
  }

  reload(): void {
    void this.store.loadSettings();
  }

  onWakeWindowChange(event: Event): void {
    const target = event.target as HTMLInputElement;
    const value = Number(target.value);
    if (!Number.isNaN(value)) {
      void this.store.updateSettings({ wakeWindowMinutes: value });
    }
  }

  onSunriseToggle(event: Event): void {
    const target = event.target as HTMLInputElement;
    void this.store.updateSettings({ sunriseEnabled: target.checked });
  }

  onSunriseIntensityChange(event: Event): void {
    const target = event.target as HTMLInputElement;
    const value = Number(target.value);
    if (!Number.isNaN(value)) {
      void this.store.updateSettings({ sunriseIntensity: value });
    }
  }

  onSoundChange(soundId: string): void {
    void this.store.updateSettings({ soundId });
  }
}
