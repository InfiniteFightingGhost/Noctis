import { Injectable, computed, signal } from "@angular/core";

export type NotificationTone = "info" | "success" | "warning" | "error";

export type Notification = {
  id: string;
  message: string;
  tone: NotificationTone;
  title?: string;
  createdAt: number;
};

export type NotificationInput = {
  message: string;
  tone?: NotificationTone;
  title?: string;
  id?: string;
  createdAt?: number;
};

const createNotificationId = (): string => {
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) {
    return crypto.randomUUID();
  }

  return `notification-${Date.now()}-${Math.random().toString(16).slice(2)}`;
};

@Injectable({ providedIn: "root" })
export class NotificationsStore {
  readonly notifications = signal<Notification[]>([]);
  readonly count = computed(() => this.notifications().length);
  readonly hasNotifications = computed(() => this.count() > 0);

  add(notification: NotificationInput): Notification {
    const entry: Notification = {
      id: notification.id ?? createNotificationId(),
      message: notification.message,
      tone: notification.tone ?? "info",
      title: notification.title,
      createdAt: notification.createdAt ?? Date.now(),
    };

    this.notifications.update((current) => [...current, entry]);
    return entry;
  }

  remove(id: string): void {
    this.notifications.update((current) => current.filter((item) => item.id !== id));
  }

  clear(): void {
    this.notifications.set([]);
  }
}
