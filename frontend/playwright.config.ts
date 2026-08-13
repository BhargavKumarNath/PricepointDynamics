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
    // NOT `-s`/`--single`: that flag puts `serve` in SPA-fallback mode,
    // which rewrites every unmatched route to `index.html` -- wrong for
    // a Next.js static export, which produces a genuinely separate HTML
    // file per route (`market-overview.html`, not `market-overview/index.html`).
    // With `-s` on, every non-home page in this suite was silently
    // being served the home page's content (found while debugging 8
    // false-positive-looking failures that were actually this).
    command: "npx serve out -l 3000",
    url: "http://localhost:3000",
    reuseExistingServer: true,
    timeout: 30_000,
  },
});
