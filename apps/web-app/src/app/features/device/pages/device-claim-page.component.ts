import { Component, OnInit, computed, inject, signal } from "@angular/core";
import { ActivatedRoute, Router, RouterLink } from "@angular/router";
import { StatusBannerComponent } from "../../../shared/ui/status-banner/status-banner.component";
import { UiButtonComponent } from "../../../shared/ui/button/button.component";
import { DeviceStore } from "../data/device.store";

@Component({
  selector: "app-device-claim-page",
  standalone: true,
  imports: [RouterLink, StatusBannerComponent, UiButtonComponent],
  template: `
    <section class="screen screen--dark sleep-summary-theme">
      <div class="screen__header">
        <div>
          <p class="screen__sub">Device setup</p>
          <h1 class="screen__title">Claim by external ID</h1>
        </div>
        <a class="primary-link" routerLink="/device">Back</a>
      </div>

      <form class="screen__section" (submit)="onSubmit($event)">
        @if (showPendingBanner()) {
          <ui-status-banner
            variant="syncing"
            title="Claim in progress"
            message="We are linking this device to your account."
          />
        }

        @if (showSuccessBanner()) {
          <ui-status-banner
            variant="ok"
            title="Device claimed"
            [message]="successMessage()"
          />
        }

        <label class="form-field" for="device-external-id">
          <span class="form-field__label">Device external ID</span>
          <input
            id="device-external-id"
            class="form-field__input"
            type="text"
            autocomplete="off"
            maxlength="200"
            [value]="externalIdInput()"
            placeholder="e.g. dodh-lite-001"
            (input)="onExternalIdInput($event)"
            required
          />
        </label>

        <div class="chart-card">
          <p class="chart-card__summary">
            Enter the external ID printed in app pairing or device settings.
          </p>
        </div>

        @if (store.claimErrorMessage()) {
          <p class="form-error" role="alert">{{ store.claimErrorMessage() }}</p>
        }

        <div class="screen__cta">
          <ui-button type="submit" [disabled]="!canSubmit() || isPending()">
            {{ submitLabel() }}
          </ui-button>
          <a class="primary-link" routerLink="/device/help">Need help finding the ID?</a>
        </div>
      </form>
    </section>
  `,
})
export class DeviceClaimPageComponent implements OnInit {
  readonly store = inject(DeviceStore);
  private readonly router = inject(Router);
  private readonly route = inject(ActivatedRoute);

  readonly externalIdInput = signal("");
  readonly isPending = computed(() => this.store.claimStatus() === "pending");
  readonly showPendingBanner = computed(() => this.store.claimStatus() === "pending");
  readonly showSuccessBanner = computed(() => this.store.claimStatus() === "success");
  readonly submitLabel = computed(() =>
    this.isPending() ? "Claiming..." : "Claim device",
  );
  readonly canSubmit = computed(() => this.externalIdInput().trim().length > 0);
  readonly successMessage = computed(
    () => `Linked ${this.externalIdInput().trim()} to this account.`,
  );

  ngOnInit(): void {
    this.store.resetClaimState();
  }

  onExternalIdInput(event: Event): void {
    const target = event.target as HTMLInputElement;
    this.externalIdInput.set(target.value);
    if (this.store.claimStatus() !== "idle") {
      this.store.resetClaimState();
    }
  }

  async onSubmit(event: Event): Promise<void> {
    event.preventDefault();
    if (!this.canSubmit() || this.isPending()) {
      return;
    }

    await this.store.claimDeviceByExternalId(this.externalIdInput());
    if (this.store.claimStatus() !== "success") {
      return;
    }
    const candidate = this.route.snapshot.queryParamMap.get("redirect");
    const redirect =
      candidate && candidate.startsWith("/") && !candidate.startsWith("//")
        ? candidate
        : "/dashboard";
    await this.router.navigateByUrl(redirect);
  }
}
