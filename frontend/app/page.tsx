import Link from "next/link";
import { BentoGrid } from "@/components/BentoGrid";
import { GlassCard } from "@/components/GlassCard";
import { MetricCard } from "@/components/MetricCard";
import {
  ArrowRightIcon,
  BasketIcon,
  BrainIcon,
  ChartBarIcon,
  DatabaseIcon,
  NetworkIcon,
  SparkleIcon,
  TagIcon,
  TargetIcon,
} from "@/components/icons";
import { loadHomeMetrics } from "@/lib/artifacts";
import { formatCompactNumber, formatGBP } from "@/lib/format";

const NAV_TILES = [
  {
    href: "/market-overview",
    label: "Market Overview",
    icon: ChartBarIcon,
    description: "A high-level view of retailer pricing and product portfolios.",
  },
  {
    href: "/basket-analysis",
    label: "Basket Analysis",
    icon: BasketIcon,
    description: "Compare the cost of standardised shopping baskets across stores.",
  },
  {
    href: "/predictor",
    label: "Price Predictor",
    icon: TargetIcon,
    description: "An interactive tool to predict product prices using the trained model.",
  },
  {
    href: "/model-insights",
    label: "Model Insights",
    icon: BrainIcon,
    description: "Understand why the model makes its predictions.",
  },
  {
    href: "/market-dynamics",
    label: "Market Dynamics",
    icon: NetworkIcon,
    description: "Explore price leadership and market competitiveness over time.",
  },
] as const;

const HOW_IT_WORKS = [
  { icon: DatabaseIcon, label: "Ingest", description: "9.5M+ daily prices from 5 major retailers" },
  { icon: NetworkIcon, label: "Match", description: "AI links the same product across every store" },
  { icon: TargetIcon, label: "Predict & explain", description: "A trained model forecasts prices, transparently" },
] as const;

export default async function HomePage() {
  const metrics = await loadHomeMetrics();

  return (
    <div className="space-y-6">
      <GlassCard variant="strong" accent padding="lg">
        <div className="text-center">
          <h1 className="text-3xl font-semibold text-text-primary">PricePoint Dynamics</h1>
          <p className="mt-2 text-text-secondary">
            An interactive dashboard for the UK supermarket competitive landscape.
          </p>
          <p className="mx-auto mt-4 max-w-2xl text-sm text-text-secondary">
            UK grocery prices shift constantly across five major retailers, with no easy way to compare true costs,
            spot which store moves first, or know what a product will cost tomorrow. This dashboard turns{" "}
            {formatCompactNumber(metrics.total_records)} raw price records into a live intelligence layer that
            answers exactly that — built end-to-end, from data pipeline to a trained, explainable machine learning
            model.
          </p>
        </div>

        <div className="mx-auto mt-8 flex max-w-2xl flex-col items-center gap-2 sm:flex-row sm:justify-between">
          {HOW_IT_WORKS.map((step, i) => (
            <div key={step.label} className="flex items-center gap-2">
              <div className="flex items-center gap-2">
                <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-full bg-brand-soft text-brand">
                  <step.icon size={18} />
                </div>
                <div className="text-left">
                  <div className="text-sm font-medium text-text-primary">{step.label}</div>
                  <div className="text-xs text-text-muted">{step.description}</div>
                </div>
              </div>
              {i < HOW_IT_WORKS.length - 1 && (
                <ArrowRightIcon size={16} className="mx-1 hidden shrink-0 text-text-muted sm:block" />
              )}
            </div>
          ))}
        </div>
      </GlassCard>

      <div>
        <h2 className="mb-4 text-lg font-semibold text-text-primary">Project highlights</h2>
        <BentoGrid>
          <MetricCard
            span={4}
            tone="brand"
            icon={<DatabaseIcon size={18} />}
            label="Total price records analysed"
            value={formatCompactNumber(metrics.total_records)}
          />
          <MetricCard
            span={4}
            icon={<TagIcon size={18} />}
            label="Comparable products identified"
            value={formatCompactNumber(metrics.canonical_products)}
          />
          <MetricCard
            span={4}
            icon={<SparkleIcon size={18} />}
            label="Price prediction model accuracy"
            value={formatGBP(metrics.mae)}
            help={`Average error (MAE) · R² ${metrics.r2.toFixed(3)} · trained on ${formatCompactNumber(
              metrics.n_train_rows,
            )} rows`}
          />
        </BentoGrid>
      </div>

      <div>
        <h2 className="mb-4 text-lg font-semibold text-text-primary">Navigating the dashboard</h2>
        <BentoGrid>
          {NAV_TILES.map((tile) => (
            <GlassCard
              key={tile.href}
              span={4}
              className="group relative transition-shadow hover:shadow-[var(--shadow-glass),var(--glow-brand)]"
            >
              <Link href={tile.href} className="absolute inset-0 rounded-bento" aria-label={tile.label} />
              <div className="flex h-9 w-9 items-center justify-center rounded-full bg-brand-soft text-brand">
                <tile.icon size={18} />
              </div>
              <div className="mt-3 font-medium text-text-primary">{tile.label}</div>
              <div className="mt-1 text-xs text-text-secondary">{tile.description}</div>
            </GlassCard>
          ))}
        </BentoGrid>
      </div>
    </div>
  );
}
