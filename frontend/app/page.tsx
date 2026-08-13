import Link from "next/link";
import { StatTile } from "@/components/StatTile";
import { loadHomeMetrics } from "@/lib/artifacts";
import { formatCompactNumber, formatGBP } from "@/lib/format";

export default async function HomePage() {
  const metrics = await loadHomeMetrics();

  return (
    <div className="space-y-10">
      <div className="text-center">
        <h1 className="text-3xl font-semibold text-text-primary">PricePoint Dynamics</h1>
        <p className="mt-2 text-text-secondary">
          An interactive dashboard for the UK supermarket competitive landscape.
        </p>
        <p className="mx-auto mt-4 max-w-2xl text-sm text-text-secondary">
          This dashboard presents the findings from an end-to-end data science project analysing over{" "}
          {formatCompactNumber(metrics.total_records)} daily price records from 5 major UK supermarkets. The goal is
          to uncover deep competitive structure, model market dynamics, and predict future prices using machine
          learning.
        </p>
      </div>

      <div>
        <h2 className="mb-4 text-lg font-semibold text-text-primary">Project Highlights</h2>
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
          <StatTile label="Total price records analysed" value={formatCompactNumber(metrics.total_records)} />
          <StatTile
            label="Comparable products identified"
            value={formatCompactNumber(metrics.canonical_products)}
          />
          <StatTile
            label="Price prediction model MAE"
            value={formatGBP(metrics.mae)}
            help={`RMSE ${formatGBP(metrics.rmse)} · R² ${metrics.r2.toFixed(3)} · trained on ${formatCompactNumber(
              metrics.n_train_rows,
            )} rows`}
          />
        </div>
      </div>

      <div className="rounded-lg border border-border bg-surface p-6">
        <h2 className="mb-3 text-lg font-semibold text-text-primary">Navigating the dashboard</h2>
        <ul className="space-y-2 text-sm text-text-secondary">
          <li>
            <Link href="/market-overview" className="font-medium text-text-primary hover:underline">
              Market Overview
            </Link>{" "}
            — a high-level view of retailer pricing and product portfolios.
          </li>
          <li>
            <Link href="/basket-analysis" className="font-medium text-text-primary hover:underline">
              Basket Analysis
            </Link>{" "}
            — compare the cost of standardised shopping baskets across stores.
          </li>
          <li>
            <Link href="/predictor" className="font-medium text-text-primary hover:underline">
              Price Predictor
            </Link>{" "}
            — an interactive tool to predict product prices using the trained model.
          </li>
          <li>
            <Link href="/model-insights" className="font-medium text-text-primary hover:underline">
              Model Insights
            </Link>{" "}
            — understand why the model makes its predictions using SHAP.
          </li>
          <li>
            <Link href="/market-dynamics" className="font-medium text-text-primary hover:underline">
              Market Dynamics
            </Link>{" "}
            — explore price leadership and market competitiveness over time.
          </li>
        </ul>
      </div>
    </div>
  );
}
