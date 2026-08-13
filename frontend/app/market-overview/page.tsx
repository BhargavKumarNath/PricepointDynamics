import { BentoGrid } from "@/components/BentoGrid";
import { Callout } from "@/components/Callout";
import { GlassCard } from "@/components/GlassCard";
import { InfoTooltip } from "@/components/InfoTooltip";
import { Meter } from "@/components/Meter";
import { PriceDistributionChart } from "@/components/charts/PriceDistributionChart";
import { SimpleBarChart } from "@/components/charts/SimpleBarChart";
import { OwnBrandChart } from "@/components/charts/OwnBrandChart";
import { supermarketColor } from "@/lib/colors";
import { loadMarketOverview } from "@/lib/artifacts";
import { formatPercent } from "@/lib/format";

export const metadata = { title: "Market Overview | PricePoint Dynamics" };

export default async function MarketOverviewPage() {
  const { supermarkets } = await loadMarketOverview();
  const sorted = [...supermarkets].sort((a, b) => a.supermarket.localeCompare(b.supermarket));

  const distributionData = sorted.map((s) => ({
    name: s.supermarket,
    color: supermarketColor(s.supermarket),
    min: s.min_price,
    p25: s.price_p25,
    median: s.price_median,
    p75: s.price_p75,
    max: s.max_price,
  }));

  const portfolioData = sorted
    .map((s) => ({ name: s.supermarket, value: s.portfolio_size, color: supermarketColor(s.supermarket) }))
    .sort((a, b) => b.value - a.value);

  const ownBrandMeterData = [...sorted].sort((a, b) => b.own_brand_pct - a.own_brand_pct);

  const ownBrandChartData = sorted.map((s) => ({
    name: s.supermarket,
    ownBrand: s.own_brand_count,
    branded: s.branded_count,
  }));

  return (
    <div className="space-y-6">
      <div className="text-center">
        <h1 className="text-2xl font-semibold text-text-primary">Market Overview</h1>
        <p className="mx-auto mt-2 max-w-2xl text-sm text-text-secondary">
          A 30,000-foot view of the UK supermarket landscape, exploring each retailer&apos;s pricing strategy,
          product portfolio, and category focus.
        </p>
      </div>

      <section>
        <h2 className="mb-4 text-lg font-semibold text-text-primary">At a glance: pricing &amp; portfolio</h2>
        <BentoGrid>
          <GlassCard span={7}>
            <div className="mb-4 flex items-center gap-1.5">
              <h3 className="text-sm font-medium text-text-secondary">
                Price distribution by supermarket (min · p25 · median · p75 · max)
              </h3>
              <InfoTooltip label="How to read this chart">
                <p>
                  Each row is a five-number summary of one retailer&apos;s prices: a thin line spans the typical
                  range (min to max), and the coloured box marks where the middle 50% of prices sit (25th to 75th
                  percentile). The white tick is the median price.
                </p>
                <p className="mt-2">
                  A handful of extreme outlier prices (e.g. large multipacks) would otherwise stretch the shared
                  scale so far that every retailer&apos;s typical range becomes invisible, so the axis is capped at a
                  standard statistical threshold — hover any row for the true, uncapped values.
                </p>
              </InfoTooltip>
            </div>
            <PriceDistributionChart data={distributionData} format="currency" />
            <div className="mt-4">
              <Callout>
                Aldi operates in a significantly lower price bracket, confirming its hard-discounter model, while the
                &ldquo;Big Four&rdquo; compete in a similar, higher price range.
              </Callout>
            </div>
          </GlassCard>

          <GlassCard span={5}>
            <h3 className="mb-4 text-sm font-medium text-text-secondary">Product portfolio size</h3>
            <SimpleBarChart data={portfolioData} format="number" />
            <div className="mt-4">
              <Callout>
                Sainsbury&apos;s, ASDA, and Tesco offer a vast range, positioning themselves as
                &ldquo;one-stop-shops&rdquo;. Aldi&apos;s curated selection highlights a strategy focused on
                operational efficiency over choice.
              </Callout>
            </div>
          </GlassCard>
        </BentoGrid>
      </section>

      <section>
        <h2 className="mb-4 text-lg font-semibold text-text-primary">Deep dive: own brand strategy</h2>
        <BentoGrid>
          <GlassCard span={4}>
            <h3 className="mb-4 text-sm font-medium text-text-secondary">Percentage of own brand</h3>
            <div className="space-y-4">
              {ownBrandMeterData.map((s) => (
                <Meter
                  key={s.supermarket}
                  label={s.supermarket}
                  value={s.own_brand_pct}
                  color={supermarketColor(s.supermarket)}
                  valueLabel={formatPercent(s.own_brand_pct, 2)}
                />
              ))}
            </div>
            <div className="mt-4">
              <Callout>
                Contrary to common perception, it&apos;s the larger supermarkets like ASDA that have the highest
                proportion of own-brand items, showing their reliance on these lines to compete.
              </Callout>
            </div>
          </GlassCard>

          <GlassCard span={8}>
            <div className="mb-4 flex items-center gap-1.5">
              <h3 className="text-sm font-medium text-text-secondary">
                Product listings: own brand vs. branded (log scale)
              </h3>
              <InfoTooltip label="Why is this axis log-scaled?" align="end">
                <p>
                  Values are log-scaled because branded listings vastly outnumber own-brand ones at some retailers —
                  a linear scale would make the smaller bars invisible. Each step up the axis represents a 10x
                  increase in listings, not an even amount.
                </p>
              </InfoTooltip>
            </div>
            <OwnBrandChart data={ownBrandChartData} />
          </GlassCard>
        </BentoGrid>
      </section>
    </div>
  );
}
