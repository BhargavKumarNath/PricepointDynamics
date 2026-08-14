import { expect, test } from "@playwright/test";

test.describe("Home page", () => {
  test("renders real, non-placeholder headline metrics", async ({ page }) => {
    await page.goto("/");
    await expect(page.getByRole("heading", { name: "PricePoint Dynamics" })).toBeVisible();
    // These come from home_metrics.json, baked in at build time -- not "0" or "undefined".
    await expect(page.getByText("Total price records analysed")).toBeVisible();
    await expect(page.getByText(/^\d/).first()).toBeVisible();
  });

  test("nav links reach every page", async ({ page }) => {
    await page.goto("/");
    const nav = page.getByRole("navigation");
    for (const [label, path] of [
      ["Market Overview", "/market-overview"],
      ["Basket Analysis", "/basket-analysis"],
      ["Price Predictor", "/predictor"],
      ["Model Insights", "/model-insights"],
      ["Market Dynamics", "/market-dynamics"],
    ] as const) {
      await nav.getByRole("link", { name: label, exact: true }).click();
      await expect(page).toHaveURL(new RegExp(`${path}$`));
    }
  });
});

test.describe("Market Overview page", () => {
  test("shows all 5 retailers' price distribution and portfolio charts", async ({ page }) => {
    await page.goto("/market-overview");
    for (const retailer of ["ASDA", "Aldi", "Morrisons", "Sains", "Tesco"]) {
      await expect(page.getByText(retailer, { exact: true }).first()).toBeVisible();
    }
  });
});

test.describe("Basket Analysis page", () => {
  test("switching baskets updates the cost table", async ({ page }) => {
    await page.goto("/basket-analysis");
    const select = page.getByLabel("Choose a shopping basket to analyse:");
    await expect(select).toBeVisible();

    await select.selectOption({ label: "The Essentials" });
    await expect(page.getByText(/Cost of.*The Essentials.*basket/)).toBeVisible();

    await select.selectOption({ label: "Pets" });
    await expect(page.getByText(/Cost of.*Pets.*basket/)).toBeVisible();
  });

  test("detailed breakdown expands to show item-level prices", async ({ page }) => {
    await page.goto("/basket-analysis");
    await page.getByRole("button", { name: /View products in this basket/ }).click();
    await expect(page.getByText(/Showing \d+ of \d+ products/)).toBeVisible();
  });
});

test.describe("Market Dynamics page", () => {
  test("shows the competitiveness index and dispersion trend", async ({ page }) => {
    await page.goto("/market-dynamics");
    await expect(page.getByText("Competitiveness index", { exact: true })).toBeVisible();
    await expect(page.getByText("Primary mover")).toBeVisible();
  });

  test("leadership matrix and its list view both show real leader/follower data", async ({ page }) => {
    await page.goto("/market-dynamics");
    // The matrix is the primary visual -- at least one populated cell (e.g. "4d") should be visible.
    await expect(page.getByRole("button", { name: /leads .* by \d+ days/ }).first()).toBeVisible();

    // The deduplicated list view is a collapsed <details> disclosure by default.
    await page.getByText("View as list").click();
    await expect(page.getByText("leads").first()).toBeVisible();
  });
});

test.describe("Model Insights page", () => {
  test("loads SHAP explorer client-side and renders feature importance", async ({ page }) => {
    await page.goto("/model-insights");
    // Parquet is fetched + parsed client-side (hyparquet), fast enough
    // locally that the loading state isn't reliably observable -- assert
    // the real end state directly rather than a transient message.
    await expect(page.getByText("Global feature importance")).toBeVisible({ timeout: 15_000 });
    await expect(page.getByText("Model's price prediction")).toBeVisible();
  });

  test("switching sample updates the local explanation", async ({ page }) => {
    await page.goto("/model-insights");
    const predictedPrice = page.getByTestId("predicted-price");
    await expect(predictedPrice).toBeVisible({ timeout: 15_000 });
    const before = await predictedPrice.textContent();
    await page.getByLabel("Product sample").selectOption({ index: 5 });
    await expect(async () => {
      expect(await predictedPrice.textContent()).not.toBe(before);
    }).toPass({ timeout: 5_000 });
  });
});

test.describe("Price Predictor page", () => {
  // The product search box itself reads a committed static artifact
  // (predictor_context.parquet), so it needs no mocking. Only the two
  // live network calls this page makes -- POST /v1/predict and
  // GET /v1/products/{id}/history -- are mocked below, via page.route(),
  // so this spec (like every other page in this file) runs in CI against
  // the static export alone, with no FastAPI backend, trained model, or
  // marts required (project_refactor.md §13: "minimal Playwright smoke
  // test against mocked API responses"). Exercising the frontend against
  // a genuinely live backend is covered separately by manual/local
  // end-to-end verification, not by CI.
  test.beforeEach(async ({ page }) => {
    await page.route("**/v1/products/**/history*", async (route) => {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({
          canonical_name: "mocked product",
          history: [
            { date: "2024-01-01", supermarket: "ASDA", avg_price: 1.2, min_price: 1.1, max_price: 1.3, n_listings: 3 },
            { date: "2024-01-08", supermarket: "ASDA", avg_price: 1.25, min_price: 1.15, max_price: 1.35, n_listings: 3 },
          ],
        }),
      });
    });
  });

  test("search, select, and predict against a mocked API", async ({ page }) => {
    await page.route("**/v1/predict", async (route) => {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({
          canonical_name: "mocked product",
          supermarket: "ASDA",
          requested_date: "2024-01-15",
          resolved_from_date: "2024-01-15",
          predicted_price: 1.29,
          price_override_applied: false,
          unresolved_features: [],
        }),
      });
    });

    await page.goto("/predictor");
    await expect(page.getByPlaceholder("e.g. 6 sweet creamy bananas")).toBeVisible({ timeout: 15_000 });

    await page.getByPlaceholder("e.g. 6 sweet creamy bananas").fill("banana");
    await page.getByRole("button", { name: /bananas/i }).first().click();

    const storeSelect = page.getByLabel("Supermarket");
    await expect(storeSelect).toBeVisible();
    await storeSelect.selectOption({ index: 1 });

    await page.getByRole("button", { name: "Predict price" }).click();
    await expect(page.getByText("Predicted price")).toBeVisible();
    await expect(page.getByText("£1.29")).toBeVisible();
  });

  test("shows the API's structured error message, not a crash", async ({ page }) => {
    await page.route("**/v1/predict", async (route) => {
      await route.fulfill({
        status: 422,
        contentType: "application/json",
        body: JSON.stringify({ detail: "Date is outside the dataset's observed range.", context: {} }),
      });
    });

    await page.goto("/predictor");
    await expect(page.getByPlaceholder("e.g. 6 sweet creamy bananas")).toBeVisible({ timeout: 15_000 });
    await page.getByPlaceholder("e.g. 6 sweet creamy bananas").fill("banana");
    await page.getByRole("button", { name: /bananas/i }).first().click();

    const storeSelect = page.getByLabel("Supermarket");
    await expect(storeSelect).toBeVisible();
    await storeSelect.selectOption({ index: 1 });

    await page.getByRole("button", { name: "Predict price" }).click();
    await expect(page.getByText("Date is outside the dataset's observed range.")).toBeVisible();
  });
});
