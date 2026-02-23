import { Injectable, signal } from "@angular/core";

@Injectable({ providedIn: "root" })
export class NetworkStore {
  readonly isOnline = signal(true);

  constructor() {
    if (typeof window !== "undefined") {
      this.isOnline.set(navigator.onLine);
      window.addEventListener("online", () => this.isOnline.set(true));
      window.addEventListener("offline", () => this.isOnline.set(false));
    }
  }
}
