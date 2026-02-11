import { Component } from "@angular/core";

@Component({
  selector: "app-home-page",
  standalone: true,
  template: `
    <section class="page">
      <div class="page__intro">
        <p class="eyebrow">Welcome</p>
        <h1>Home</h1>
        <p class="lede">
          This is a starter home page. Replace this content with your real
          experience.
        </p>
      </div>
    </section>
  `,
})
export class HomePageComponent {}
