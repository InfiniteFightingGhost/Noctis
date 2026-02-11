import { bootstrapApplication } from "@angular/platform-browser";
import { AppComponent } from "./app/app.component";
import { APP_PROVIDERS } from "./app/core/core.providers";

bootstrapApplication(AppComponent, {
  providers: [...APP_PROVIDERS],
});
