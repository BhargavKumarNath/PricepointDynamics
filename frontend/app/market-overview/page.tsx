import { PriceDistributionChart } from "@/components/charts/PriceDistributionChart";
import { SimpleBarChart } from "@/components/charts/SimpleBarChart";
import { OwnBrandChart } from "@/components/charts/OwnBrandChart";
import { supermarketColor } from "@/lib/colors";
import { loadMarketOverview } from "@/lib/artifacts";
import { formatGBP, formatPercent } from "@/lib/format";

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

  const ownBrandTableData = [...sorted].sort((a, b) => b.own_brand_pct - a.own_brand_pct);

  const ownBrandChartData = sorted.map((s) => ({
    name: s.supermarket,
    ownBrand: s.own_brand_count,
    branded: s.branded_count,
  }));

  return (
    <div className="space-y-10">
      <div className="text-center">
        <h1 className="text-2xl font-semibold text-text-primary">Market Overview</h1>
        <p className="mx-auto mt-2 max-w-2xl text-sm text-text-secondary">
          A 30,000-foot view of the UK supermarket landscape, exploring each retailer&apos;s pricing strategy,
          product portfolio, and category focus.
        </p>
      </div>

      <section>
        <h2 className="mb-4 text-lg font-semibold text-text-primary">At a glance: pricing &amp; portfolio</h2>
        <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
          <div className="rounded-lg border border-border bg-surface p-5">
            <h3 className="mb-4 text-sm font-medium text-text-secondary">
              Price distribution by supermarket (min · p25 · median · p75 · max)
            </h3>
            <PriceDistributionChart data={distributionData} valueFormatter={formatGBP} />
            <p className="mt-4 text-xs text-text-secondary">
              <strong className="text-text-primary">Insight:</strong> Aldi operates in a significantly lower price
              bracket, confirming its hard-discounter model, while the &ldquo;Big Four&rdquo; compete in a similar,
              higher price range.
            </p>
          </div>
          <div className="rounded-lg border border-border bg-surface p-5">
            <h3 className="mb-4 text-sm font-medium text-text-secondary">Product portfolio size</h3>
            <SimpleBarChart data={portfolioData} format="number" />
            <p className="mt-4 text-xs text-text-secondary">
              <strong className="text-text-primary">Insight:</strong> Sainsbury&apos;s, ASDA, and Tesco offer a vast
              range, positioning themselves as &ldquo;one-stop-shops&rdquo;. Aldi&apos;s curated selection highlights
              a strategy focused on operational efficiency over choice.
            </p>
          </div>
        </div>
      </section>

      <section>
        <h2 className="mb-4 text-lg font-semibold text-text-primary">Deep dive: own brand strategy</h2>
        <div className="grid grid-cols-1 gap-4 lg:grid-cols-[1fr_2fr]">
          <div className="rounded-lg border border-border bg-surface p-5">
            <h3 className="mb-4 text-sm font-medium text-text-secondary">Percentage of own brand</h3>
            <table className="w-full text-sm">
              <tbody>
                {ownBrandTableData.map((s) => (
                  <tr key={s.supermarket} className="border-b border-border last:border-0">
                    <td className="py-2 text-text-secondary">{s.supermarket}</td>
                    <td className="py-2 text-right font-medium text-text-primary">
                      {formatPercent(s.own_brand_pct, 2)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            <p className="mt-4 text-xs text-text-secondary">
              <strong className="text-text-primary">Insight:</strong> Contrary to common perception, it&apos;s the
              larger supermarkets like ASDA that have the highest proportion of own-brand items, showing their
              reliance on these lines to compete.
            </p>
          </div>
          <div className="rounded-lg border border-border bg-surface p-5">
            <h3 className="mb-4 text-sm font-medium text-text-secondary">
              Product listings: own brand vs. branded (log scale)
            </h3>
            <OwnBrandChart data={ownBrandChartData} />
          </div>
        </div>
      </section>
    </div>
  );
}
