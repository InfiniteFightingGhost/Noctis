import { defineConfig } from "@playwright/test";

export default defineConfig({
  testDir: "./e2e",
  timeout: 120_000,
  expect: {
    timeout: 10_000,
  },
  use: {
    baseURL: "http://localhost:4200",
    headless: true,
  },
  webServer: {
    command: "npm run start -- --host 0.0.0.0 --port 4200",
    url: "http://localhost:4200",
    reuseExistingServer: true,
    cwd: ".",
    timeout: 120_000,
  },
});
