import { defineConfig, devices } from "@playwright/test";

// Smoke tests against the production build (`npm run build` first). Every
// API call is intercepted in e2e/mock-api.ts, so no backend is needed.
const PORT = 3100;

export default defineConfig({
  testDir: "./e2e",
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: 0,
  reporter: process.env.CI ? [["github"], ["list"]] : "list",
  use: {
    baseURL: `http://localhost:${PORT}`,
    trace: "retain-on-failure",
  },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }],
  webServer: {
    command: `npx next start -p ${PORT}`,
    url: `http://localhost:${PORT}/overview`,
    reuseExistingServer: !process.env.CI,
    // The proxy route must never reach a real API from a test run.
    env: { API_URL: "http://127.0.0.1:9" },
  },
});
