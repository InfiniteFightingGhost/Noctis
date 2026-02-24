import { Routes } from "@angular/router";
import { DevicePageComponent } from "./pages/device-page.component";
import { DeviceClaimPageComponent } from "./pages/device-claim-page.component";
import { DeviceHelpPageComponent } from "./pages/device-help-page.component";

export const DEVICE_ROUTES: Routes = [
  {
    path: "",
    component: DevicePageComponent,
  },
  {
    path: "help",
    component: DeviceHelpPageComponent,
  },
  {
    path: "claim",
    component: DeviceClaimPageComponent,
  },
];
