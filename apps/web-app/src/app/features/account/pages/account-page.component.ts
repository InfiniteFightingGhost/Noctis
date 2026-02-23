import { Component, inject } from "@angular/core";
import { AccountStore } from "../data/account.store";

@Component({
  selector: "app-account-page",
  standalone: true,
  template: `
    <section class="page">
      <div class="page__intro">
        <p class="eyebrow">Account</p>
        <h1>Your account</h1>
        <p class="lede">Manage profile settings and preferences here.</p>
      </div>

      <div class="page__section">
        <div class="section-head">
          <div>
            <p class="eyebrow">Admin tools</p>
            <h2>Create a user</h2>
            <p class="lede">
              Use an admin token to create a user in the active tenant.
            </p>
          </div>
          <span class="status-pill" [class.is-live]="store.status() === 'success'">
            {{ store.statusLabel() }}
          </span>
        </div>

        <form class="account-form" (submit)="onSubmit($event)">
          <label class="form-field">
            <span>Name</span>
            <input
              type="text"
              [value]="store.name()"
              (input)="onNameInput($event)"
              placeholder="e.g., Alex Morgan"
              required
            />
          </label>

          <label class="form-field">
            <span>External ID (optional)</span>
            <input
              type="text"
              [value]="store.externalId()"
              (input)="onExternalIdInput($event)"
              placeholder="crm-12452"
            />
          </label>

          <label class="form-field">
            <span>Admin token</span>
            <input
              type="password"
              [value]="store.adminToken()"
              (input)="onAdminTokenInput($event)"
              placeholder="Bearer token"
              autocomplete="off"
              required
            />
          </label>

          <div class="form-actions">
            <button
              class="primary-button"
              type="submit"
              [disabled]="!store.canSubmit()"
            >
              {{ store.submitLabel() }}
            </button>
            <p class="hint">Requires admin role in the API.</p>
          </div>
        </form>

        @if (store.status() === "error") {
          <div class="notice notice--error">{{ store.errorMessage() }}</div>
        }

        @if (store.createdUser()) {
          <div class="result-card">
            <h3>User created</h3>
            <div class="result-grid">
              <div>
                <span class="result-label">Name</span>
                <strong>{{ store.createdUser()?.name }}</strong>
              </div>
              <div>
                <span class="result-label">External ID</span>
                <strong>{{ store.createdUser()?.external_id || "—" }}</strong>
              </div>
              <div>
                <span class="result-label">User ID</span>
                <strong>{{ store.createdUser()?.id }}</strong>
              </div>
              <div>
                <span class="result-label">Created</span>
                <strong>{{ store.createdUser()?.created_at }}</strong>
              </div>
            </div>
          </div>
        }
      </div>
    </section>
  `,
  styles: [
    `
      .page__section {
        margin-top: var(--space-4);
        display: grid;
        gap: var(--space-3);
      }

      .section-head {
        display: flex;
        gap: var(--space-3);
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
      }

      .status-pill {
        padding: 0.35rem 0.8rem;
        border-radius: 999px;
        background: var(--color-accent-soft);
        color: var(--color-accent);
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
      }

      .status-pill.is-live {
        background: rgba(15, 107, 109, 0.12);
        color: #0f6b6d;
      }

      .account-form {
        display: grid;
        gap: var(--space-2);
      }

      .form-field {
        display: grid;
        gap: 0.4rem;
        font-size: 0.9rem;
        color: var(--color-muted);
      }

      .form-field input {
        border: 1px solid rgba(15, 27, 45, 0.12);
        border-radius: var(--radius-md);
        padding: 0.75rem 0.85rem;
        font-size: 1rem;
        font-family: var(--font-body);
        color: var(--color-ink);
        background: #fff;
      }

      .form-field input:focus {
        outline: 2px solid rgba(15, 107, 109, 0.35);
        border-color: rgba(15, 107, 109, 0.6);
      }

      .form-actions {
        display: flex;
        gap: var(--space-2);
        align-items: center;
        flex-wrap: wrap;
      }

      .primary-button {
        border: none;
        border-radius: 999px;
        padding: 0.75rem 1.6rem;
        background: var(--color-accent);
        color: #fff;
        font-size: 0.95rem;
        cursor: pointer;
      }

      .primary-button[disabled] {
        opacity: 0.5;
        cursor: not-allowed;
      }

      .hint {
        margin: 0;
        font-size: 0.85rem;
      }

      .notice {
        padding: 0.75rem 1rem;
        border-radius: var(--radius-md);
        background: rgba(196, 40, 40, 0.08);
        color: #7d1f1f;
      }

      .result-card {
        padding: var(--space-3);
        border-radius: var(--radius-lg);
        background: var(--color-accent-soft);
      }

      .result-grid {
        display: grid;
        gap: var(--space-2);
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      }

      .result-label {
        display: block;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: var(--color-muted);
        margin-bottom: 0.25rem;
      }
    `,
  ],
})
export class AccountPageComponent {
  readonly store = inject(AccountStore);

  onNameInput(event: Event) {
    const target = event.target as HTMLInputElement;
    this.store.updateName(target.value);
  }

  onExternalIdInput(event: Event) {
    const target = event.target as HTMLInputElement;
    this.store.updateExternalId(target.value);
  }

  onAdminTokenInput(event: Event) {
    const target = event.target as HTMLInputElement;
    this.store.updateAdminToken(target.value);
  }

  async onSubmit(event: Event) {
    event.preventDefault();
    await this.store.createUser();
  }
}
