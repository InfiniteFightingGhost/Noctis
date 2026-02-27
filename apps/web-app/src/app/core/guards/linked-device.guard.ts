import { inject } from "@angular/core";
import { CanActivateFn, Router } from "@angular/router";
import { firstValueFrom } from "rxjs";
import { DevicesApi } from "../api/devices.api";

export const linkedDeviceGuard: CanActivateFn = async (_, state) => {
  const api = inject(DevicesApi);
  const router = inject(Router);

  try {
    const devices = await firstValueFrom(api.getDevices());
    const hasLinkedDevice = devices.some((device) => Boolean(device.user_id));
    if (hasLinkedDevice) {
      return true;
    }
  } catch {
    return router.createUrlTree(["/device/claim"], {
      queryParams: {
        redirect: state.url,
      },
    });
  }

  return router.createUrlTree(["/device/claim"], {
    queryParams: {
      redirect: state.url,
    },
  });
};
