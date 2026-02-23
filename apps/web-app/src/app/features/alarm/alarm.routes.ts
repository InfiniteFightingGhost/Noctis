import { Routes } from "@angular/router";
import { AlarmPageComponent } from "./pages/alarm-page.component";
import { AlarmSettingsPageComponent } from "./pages/alarm-settings-page.component";

export const ALARM_ROUTES: Routes = [
  {
    path: "",
    component: AlarmPageComponent,
  },
  {
    path: "settings",
    component: AlarmSettingsPageComponent,
  },
];
