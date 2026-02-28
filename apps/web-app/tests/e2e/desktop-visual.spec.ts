import { expect, test, type Page } from "@playwright/test";

async function setStableClientState(page: Page): Promise<void> {
  await page.addInitScript(() => {
    window.localStorage.setItem("noctis-theme-preference", "light");
  });
}

async function completeSignupAndConnect(page: Page): Promise<void> {
  await page.goto("/signup");
  await page.getByLabel("Full name").fill("Desktop Snapshot User");
  await page.getByLabel("Email").fill("desktop-user@noctis.example");
  await page.getByLabel("Password", { exact: true }).fill("Password123");
  await page.getByLabel("Confirm password").fill("Password123");
  await page.getByRole("button", { name: "Sign up" }).click();
  await expect(page).toHaveURL(/\/connect-device$/);

  await page.getByLabel("Device external ID").fill("noctis-halo-s1-001");
  await page.getByRole("button", { name: "Connect device" }).click();
  await expect(page).toHaveURL(/\/$/);
}

test.beforeEach(async ({ page }) => {
  await page.setViewportSize({ width: 1280, height: 800 });
  await setStableClientState(page);
  await completeSignupAndConnect(page);
  await page.addStyleTag({
    content: "*,*::before,*::after{animation:none!important;transition:none!important;}",
  });
});

test("dashboard desktop visual baseline", async ({ page }) => {
  await expect(page.locator("main")).toHaveScreenshot("dashboard-desktop.png", {
    animations: "disabled",
    caret: "hide",
    maxDiffPixelRatio: 0.01,
  });
});

test("trends desktop visual baseline", async ({ page }) => {
  await page.getByRole("link", { name: "Trends" }).click();
  await expect(page).toHaveURL(/\/trends$/);
  await expect(page.getByText("Sleep Score Longitudinal")).toBeVisible();

  await expect(page.locator("main")).toHaveScreenshot("trends-desktop.png", {
    animations: "disabled",
    caret: "hide",
    maxDiffPixelRatio: 0.01,
  });
});

test("settings desktop visual baseline", async ({ page }) => {
  await page.getByRole("link", { name: "Settings" }).click();
  await expect(page).toHaveURL(/\/settings$/);
  await expect(page.getByRole("tab", { name: "Profile" })).toBeVisible();

  await expect(page.locator("main")).toHaveScreenshot("settings-desktop.png", {
    animations: "disabled",
    caret: "hide",
    maxDiffPixelRatio: 0.01,
  });
});
