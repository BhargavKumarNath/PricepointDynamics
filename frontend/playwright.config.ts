import { defineConfig, devices } from "@playwright/test";

/**
 * Smoke tests for the statically-exported app (project_refactor.md §19
 * Phase 5: "Playwright smoke tests for the Next.js app"). Runs against
 * the real `out/` build via `serve`, not `next dev` -- the point is to
 * verify what actually ships, not the dev server's behavior.
 */
export default defineConfig({
  testDir: "./tests/e2e",
  fullyParallel: true,
  retries: 0,
  reporter: "list",
  use: {
    baseURL: "http://localhost:3000",
    trace: "retain-on-failure",
  },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }],
  webServer: {
    command: "npx serve -s out -l 3000",
    url: "http://localhost:3000",
    reuseExistingServer: true,
    timeout: 30_000,
  },
});
