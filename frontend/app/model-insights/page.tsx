import { ShapExplorerClient } from "@/components/ShapExplorerClient";
import { StatTile } from "@/components/StatTile";
import { loadHomeMetrics, loadShapExplorerMeta } from "@/lib/artifacts";
import { formatCompactNumber, formatGBP } from "@/lib/format";

export const metadata = { title: "Model Insights | PricePoint Dynamics" };

export default async function ModelInsightsPage() {
  const [metrics, meta] = await Promise.all([loadHomeMetrics(), loadShapExplorerMeta()]);

  return (
    <div className="space-y-10">
      <div className="text-center">
        <h1 className="text-2xl font-semibold text-text-primary">Model Insights &amp; Explainable AI (XAI)</h1>
        <p className="mx-auto mt-2 max-w-2xl text-sm text-text-secondary">
          This page delves into the &ldquo;brain&rdquo; of the price prediction model. We use SHAP to understand
          not just what the model predicts, but why.
        </p>
      </div>

      <section className="rounded-lg border border-border bg-surface p-5">
        <h2 className="mb-1 text-lg font-semibold text-text-primary">Model performance</h2>
        <p className="mb-4 text-sm text-text-secondary">
          The LightGBM model was trained on {formatCompactNumber(metrics.n_train_rows)} data points and tested on a
          hold-out set of {formatCompactNumber(metrics.n_test_rows)} records.
        </p>
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
          <StatTile label="Mean absolute error (MAE)" value={formatGBP(metrics.mae)} help="On average, off by this much." />
          <StatTile label="Root mean squared error (RMSE)" value={formatGBP(metrics.rmse)} help="Penalizes larger errors more." />
          <StatTile label="Dataset size" value={formatCompactNumber(metrics.total_records)} help="Total records analysed." />
        </div>
      </section>

      <ShapExplorerClient meta={meta} />
    </div>
  );
}
