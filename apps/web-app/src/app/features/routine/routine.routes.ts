import { Routes } from "@angular/router";
import { RoutinePageComponent } from "./pages/routine-page.component";
import { RoutineEditPageComponent } from "./pages/routine-edit-page.component";

export const ROUTINE_ROUTES: Routes = [
  {
    path: "",
    component: RoutinePageComponent,
  },
  {
    path: "edit",
    component: RoutineEditPageComponent,
  },
];
