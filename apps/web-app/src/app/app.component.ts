import { Component } from "@angular/core";
import { AppShellComponent } from "./layout/components/app-shell.component";

@Component({
  selector: "app-root",
  standalone: true,
  imports: [AppShellComponent],
  template: "<app-shell />",
})
export class AppComponent {}
