import { Component } from "@angular/core";

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
    </section>
  `,
})
export class AccountPageComponent {}
