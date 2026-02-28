import { expect, test, type Page } from "@playwright/test";

type ViewportCase = {
  name: string;
  width: number;
  height: number;
};

const viewports: ViewportCase[] = [
  { name: "320", width: 320, height: 568 },
  { name: "375", width: 375, height: 812 },
  { name: "390", width: 390, height: 844 },
  { name: "412", width: 412, height: 915 },
  { name: "768", width: 768, height: 1024 },
  { name: "1024", width: 1024, height: 768 },
  { name: "landscape", width: 844, height: 390 },
];

async function setStableClientState(page: Page): Promise<void> {
  await page.addInitScript(() => {
    window.localStorage.setItem("noctis-theme-preference", "light");
  });
}

async function completeSignupAndConnect(page: Page): Promise<void> {
  await page.goto("/signup");
  await page.getByLabel("Full name").fill("Mobile Test User");
  await page.getByLabel("Email").fill("mobile-user@noctis.example");
  await page.getByLabel("Password", { exact: true }).fill("Password123");
  await page.getByLabel("Confirm password").fill("Password123");
  await page.getByRole("button", { name: "Sign up" }).click();
  await expect(page).toHaveURL(/\/connect-device$/);

  await page.getByLabel("Device external ID").fill("noctis-halo-s1-001");
  await page.getByRole("button", { name: "Connect device" }).click();
  await expect(page).toHaveURL(/\/$/);
  await expect(page.getByRole("heading", { name: "Personal sleep tracking." })).toBeVisible();
}

async function openMenuIfCollapsed(page: Page): Promise<void> {
  const menuButton = page.getByRole("button", { name: /open menu|close menu/i });
  if (await menuButton.isVisible()) {
    const expanded = await menuButton.getAttribute("aria-expanded");
    if (expanded !== "true") {
      await menuButton.click();
    }
  }
}

async function gotoPrimaryRoute(page: Page, routeName: "Home" | "Trends" | "Settings"): Promise<void> {
  await openMenuIfCollapsed(page);
  await page.getByRole("link", { name: routeName }).click();
}

for (const viewport of viewports) {
  test(`responsive flow @${viewport.name}`, async ({ page }) => {
    await page.setViewportSize({ width: viewport.width, height: viewport.height });
    await setStableClientState(page);
    await completeSignupAndConnect(page);

    await gotoPrimaryRoute(page, "Trends");
    await expect(page).toHaveURL(/\/trends$/);
    await expect(page.getByText("Sleep Score Longitudinal")).toBeVisible();

    await gotoPrimaryRoute(page, "Settings");
    await expect(page).toHaveURL(/\/settings$/);
    await expect(page.getByRole("tab", { name: "Profile" })).toBeVisible();

    await gotoPrimaryRoute(page, "Home");
    await expect(page).toHaveURL(/\/$/);

    const hasHorizontalOverflow = await page.evaluate(() => {
      const active = document.activeElement;
      if (active instanceof HTMLElement) {
        active.blur();
      }

      const main = document.querySelector("main");
      if (!(main instanceof HTMLElement)) {
        return false;
      }

      return main.scrollWidth - main.clientWidth > 1;
    });
    expect(hasHorizontalOverflow).toBe(false);
  });
}

test("login flow works on mobile", async ({ page }) => {
  await page.setViewportSize({ width: 375, height: 812 });
  await setStableClientState(page);

  await page.goto("/login");
  await page.getByLabel("Email").fill("sample@noctis.example");
  await page.getByLabel("Password").fill("Password123");
  await page.getByRole("button", { name: "Log in" }).click();

  await expect(page).toHaveURL(/\/connect-device$/);
  await expect(page.getByRole("heading", { name: "Connect mountable device" })).toBeVisible();
});
