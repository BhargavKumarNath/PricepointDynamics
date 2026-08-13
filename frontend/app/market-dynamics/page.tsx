import { BentoGrid } from "@/components/BentoGrid";
import { Callout } from "@/components/Callout";
import { GlassCard } from "@/components/GlassCard";
import { InfoTooltip } from "@/components/InfoTooltip";
import { LeadershipList } from "@/components/LeadershipList";
import { LeadershipMatrix } from "@/components/LeadershipMatrix";
import { Meter } from "@/components/Meter";
import { MetricCard } from "@/components/MetricCard";
import { DispersionLineChart } from "@/components/charts/DispersionLineChart";
import { loadMarketDynamics } from "@/lib/artifacts";

export const metadata = { title: "Market Dynamics | PricePoint Dynamics" };

function withRollingAverage(dispersion: { date: string; dispersion: number }[]) {
  return dispersion.map((point, i) => {
    const window = dispersion.slice(Math.max(0, i - 6), i + 1);
    const rollingAvg = window.length >= 7 ? window.reduce((sum, p) => sum + p.dispersion, 0) / window.length : null;
    return { ...point, rollingAvg };
  });
}

export default async function MarketDynamicsPage() {
  const dynamics = await loadMarketDynamics();
  const chartData = withRollingAverage(dynamics.dispersion);

  // Competitiveness index (0-100): 100 = the most competitive day
  // observed in this window, 0 = the least -- always relative to the
  // observed period, never presented as an absolute scale.
  const dispersionValues = dynamics.dispersion.map((d) => d.dispersion);
  const minDispersion = Math.min(...dispersionValues);
  const maxDispersion = Math.max(...dispersionValues);
  const competitivenessIndex =
    maxDispersion === minDispersion
      ? 100
      : 100 * (1 - (dynamics.latest_dispersion - minDispersion) / (maxDispersion - minDispersion));

  return (
    <div className="space-y-6">
      <div className="text-center">
        <h1 className="text-2xl font-semibold text-text-primary">Market Dynamics &amp; Price Leadership</h1>
        <p className="mx-auto mt-2 max-w-2xl text-sm text-text-secondary">
          This section explores the strategic interactions between retailers over time, answering the question: who
          leads, and who follows?
        </p>
      </div>

      <section>
        <div className="mb-4 flex items-center gap-1.5">
          <h2 className="text-lg font-semibold text-text-primary">Market competitiveness over time</h2>
          <InfoTooltip label="What is the competitiveness index?">
            <p>
              A 0-100 score comparing today&apos;s price dispersion (how much the same product&apos;s price varies
              across the 5 stores) against the range observed in this dataset. 100 means today is the most
              competitive day seen; 0 means the least. It&apos;s relative to what&apos;s actually been observed, not
              an absolute industry benchmark.
            </p>
          </InfoTooltip>
        </div>
        <BentoGrid>
          <MetricCard
            span={4}
            tone="brand"
            label="Competitiveness index"
            value={competitivenessIndex.toFixed(0)}
            help="0 = least competitive day observed, 100 = most"
          />
          <GlassCard span={8} padding="sm">
            <Meter
              label="Today vs. the observed range"
              value={competitivenessIndex}
              valueLabel={`Dispersion ${dynamics.latest_dispersion.toFixed(3)} (avg ${dynamics.avg_dispersion.toFixed(3)})`}
            />
            <p className="mt-3 text-xs text-text-secondary">
              Price dispersion is the average variation of prices for the same product across stores — a lower value
              means the market is pricing more uniformly (more competitive).
            </p>
          </GlassCard>
        </BentoGrid>

        <GlassCard className="mt-4">
          <DispersionLineChart data={chartData} />
        </GlassCard>
        <div className="mt-4">
          <Callout>
            After a volatile period in mid-January, the market settled into a stable equilibrium. The level of price
            difference between retailers is not escalating into a price war, nor is it diminishing.
          </Callout>
        </div>
      </section>

      <section>
        <h2 className="mb-4 text-lg font-semibold text-text-primary">Who leads and who follows?</h2>
        <p className="mb-4 text-sm text-text-secondary">
          An entry &ldquo;A leads B&rdquo; means A&apos;s price changes tend to happen before B&apos;s.
        </p>
        {dynamics.top_leader && dynamics.fastest_follower && (
          <BentoGrid className="mb-4">
            <MetricCard
              span={6}
              label="Primary mover"
              value={dynamics.top_leader}
              help="Most frequently leads price changes."
            />
            <MetricCard
              span={6}
              label="Fastest follower"
              value={dynamics.fastest_follower.follower}
              help={`Reacts within ${Math.abs(dynamics.fastest_follower.median_lag_days).toFixed(1)} days of ${dynamics.fastest_follower.leader}`}
            />
          </BentoGrid>
        )}

        <GlassCard>
          <LeadershipMatrix rows={dynamics.leadership} />
        </GlassCard>

        <details className="mt-4">
          <summary className="cursor-pointer text-sm font-medium text-text-primary hover:underline">
            View as list
          </summary>
          <div className="mt-3">
            <LeadershipList rows={dynamics.leadership} />
          </div>
        </details>

        <div className="mt-4">
          <Callout>
            Aldi is a primary price-setter, with the &ldquo;Big Four&rdquo; following its changes with a lag of
            several days. Among the Big Four, the relationships are much faster and more reciprocal, indicating a
            tight, reactive competitive cluster.
          </Callout>
        </div>
      </section>
    </div>
  );
}
