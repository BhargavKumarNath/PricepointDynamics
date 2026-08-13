import { BentoGrid } from "@/components/BentoGrid";
import { GlassCard } from "@/components/GlassCard";
import { InfoTooltip } from "@/components/InfoTooltip";
import { MetricCard } from "@/components/MetricCard";
import { ShapExplorerClient } from "@/components/ShapExplorerClient";
import { loadHomeMetrics, loadShapExplorerMeta } from "@/lib/artifacts";
import { formatCompactNumber, formatGBP } from "@/lib/format";

export const metadata = { title: "Model Insights | PricePoint Dynamics" };

export default async function ModelInsightsPage() {
  const [metrics, meta] = await Promise.all([loadHomeMetrics(), loadShapExplorerMeta()]);

  return (
    <div className="space-y-6">
      <div className="text-center">
        <div className="flex items-center justify-center gap-1.5">
          <h1 className="text-2xl font-semibold text-text-primary">Model Insights</h1>
          <InfoTooltip label="What is explainable AI?">
            <p>
              Most machine learning models are a &ldquo;black box&rdquo; — you see the prediction, not why. This page
              uses <strong className="text-text-primary">SHAP</strong> (SHapley Additive exPlanations), a technique
              that shows exactly how much each factor pushed a prediction up or down — like getting an itemised
              receipt for a forecast, rather than a single number and a shrug.
            </p>
          </InfoTooltip>
        </div>
        <p className="mx-auto mt-2 max-w-2xl text-sm text-text-secondary">
          What does the model actually use to predict prices, and why did it predict what it did for one specific
          product?
        </p>
      </div>

      <GlassCard>
        <h2 className="mb-1 text-lg font-semibold text-text-primary">Model performance</h2>
        <p className="mb-4 text-sm text-text-secondary">
          The model was trained on {formatCompactNumber(metrics.n_train_rows)} data points and tested on a hold-out
          set of {formatCompactNumber(metrics.n_test_rows)} records it never saw during training.
        </p>
        <BentoGrid>
          <MetricCard
            span={4}
            tone="brand"
            label="Average prediction error"
            value={formatGBP(metrics.mae)}
            help="On a typical product, the forecast is within this much of the real price (MAE)."
          />
          <MetricCard
            span={4}
            label="Error, penalising big misses"
            value={formatGBP(metrics.rmse)}
            help="Same idea, but weighted so rare large errors count more (RMSE)."
          />
          <MetricCard
            span={4}
            label="Dataset size"
            value={formatCompactNumber(metrics.total_records)}
            help="Total price records analysed."
          />
        </BentoGrid>
      </GlassCard>

      <ShapExplorerClient meta={meta} />
    </div>
  );
}
