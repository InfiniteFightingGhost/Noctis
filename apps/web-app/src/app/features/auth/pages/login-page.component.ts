import { Component, computed, signal } from "@angular/core";
import { ActivatedRoute, Router, RouterLink } from "@angular/router";
import { AuthService } from "../../../core/services/auth.service";
import { getAuthErrorMessage } from "../../../core/utils/auth-errors";
import {
  getPostAuthRedirect,
  isEmailValid,
  isPasswordValid,
} from "../auth-form.utils";

@Component({
  selector: "app-login-page",
  standalone: true,
  imports: [RouterLink],
  template: `
    <section class="auth-page">
      <div class="auth-card">
        <p class="eyebrow">Noctis</p>
        <h1>Log in</h1>
        <p class="lede">Sign in to continue to your sleep dashboard.</p>

        <form class="auth-form" (submit)="onSubmit($event)">
          <label class="form-field">
            <span>Email</span>
            <input
              type="email"
              autocomplete="email"
              [value]="email()"
              (input)="onEmailInput($event)"
              placeholder="you@example.com"
              [attr.aria-invalid]="showEmailError()"
              required
            />
          </label>

          <label class="form-field">
            <span>Password</span>
            <input
              type="password"
              autocomplete="current-password"
              [value]="password()"
              (input)="onPasswordInput($event)"
              placeholder="At least 8 characters"
              [attr.aria-invalid]="showPasswordError()"
              required
            />
          </label>

          @if (showEmailError()) {
            <p class="field-error">Enter a valid email address.</p>
          }

          @if (showPasswordError()) {
            <p class="field-error">Password must be at least 8 characters.</p>
          }

          @if (errorMessage()) {
            <p class="auth-error" role="alert">{{ errorMessage() }}</p>
          }

          <button class="auth-button" type="submit" [disabled]="isSubmitting()">
            {{ submitLabel() }}
          </button>
        </form>

        <p class="auth-link-row">
          New here?
          <a routerLink="/signup">Create an account</a>
        </p>
      </div>
    </section>
  `,
  styles: [
    `
      .auth-page {
        min-height: calc(100vh - 6rem);
        display: grid;
        place-items: center;
      }

      .auth-card {
        width: min(100%, 420px);
        background: var(--color-surface);
        border-radius: var(--radius-lg);
        box-shadow: var(--shadow-soft);
        padding: var(--space-4);
      }

      h1 {
        margin-bottom: var(--space-1);
      }

      .auth-form {
        margin-top: var(--space-3);
        display: grid;
        gap: var(--space-2);
      }

      .form-field {
        display: grid;
        gap: 0.4rem;
        color: var(--color-muted);
      }

      .form-field input {
        border: 1px solid rgba(15, 27, 45, 0.12);
        border-radius: var(--radius-md);
        padding: 0.75rem 0.85rem;
        font-family: var(--font-body);
        font-size: 1rem;
      }

      .form-field input:focus {
        outline: 2px solid rgba(15, 107, 109, 0.35);
        border-color: rgba(15, 107, 109, 0.6);
      }

      .field-error,
      .auth-error {
        margin: 0;
        padding: 0.6rem 0.8rem;
        border-radius: var(--radius-md);
        background: rgba(196, 40, 40, 0.08);
        color: #7d1f1f;
        font-size: 0.9rem;
      }

      .auth-button {
        border: none;
        border-radius: 999px;
        padding: 0.8rem 1rem;
        background: var(--color-accent);
        color: #fff;
        font-family: var(--font-body);
        font-size: 0.95rem;
        cursor: pointer;
      }

      .auth-button[disabled] {
        opacity: 0.6;
        cursor: not-allowed;
      }

      .auth-link-row {
        margin-top: var(--space-3);
        font-size: 0.95rem;
      }

      .auth-link-row a {
        color: var(--color-accent);
        text-decoration: underline;
      }
    `,
  ],
})
export class LoginPageComponent {
  readonly email = signal("");
  readonly password = signal("");
  readonly hasSubmitted = signal(false);
  readonly isSubmitting = signal(false);
  readonly errorMessage = signal<string | null>(null);

  readonly showEmailError = computed(
    () => this.hasSubmitted() && !isEmailValid(this.email()),
  );
  readonly showPasswordError = computed(
    () => this.hasSubmitted() && !isPasswordValid(this.password()),
  );
  readonly submitLabel = computed(() =>
    this.isSubmitting() ? "Signing in..." : "Sign in",
  );

  constructor(
    private readonly auth: AuthService,
    private readonly router: Router,
    private readonly route: ActivatedRoute,
  ) {}

  onEmailInput(event: Event): void {
    const target = event.target as HTMLInputElement;
    this.email.set(target.value);
    this.errorMessage.set(null);
  }

  onPasswordInput(event: Event): void {
    const target = event.target as HTMLInputElement;
    this.password.set(target.value);
    this.errorMessage.set(null);
  }

  async onSubmit(event: Event): Promise<void> {
    event.preventDefault();
    this.hasSubmitted.set(true);
    this.errorMessage.set(null);

    if (this.showEmailError() || this.showPasswordError()) {
      return;
    }

    this.isSubmitting.set(true);

    try {
      await this.auth.login(this.email().trim(), this.password());
      const redirect = getPostAuthRedirect(this.route.snapshot.queryParamMap.get("redirect"));
      await this.router.navigateByUrl(redirect);
    } catch (error) {
      this.errorMessage.set(getAuthErrorMessage(error));
    } finally {
      this.isSubmitting.set(false);
    }
  }
}
