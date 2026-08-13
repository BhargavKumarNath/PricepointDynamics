import { DispersionLineChart } from "@/components/charts/DispersionLineChart";
import { LeadershipList } from "@/components/LeadershipList";
import { StatTile } from "@/components/StatTile";
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

  return (
    <div className="space-y-10">
      <div className="text-center">
        <h1 className="text-2xl font-semibold text-text-primary">Market Dynamics &amp; Price Leadership</h1>
        <p className="mx-auto mt-2 max-w-2xl text-sm text-text-secondary">
          This section explores the strategic interactions between retailers over time, answering the question: who
          leads, and who follows?
        </p>
      </div>

      <section>
        <h2 className="mb-4 text-lg font-semibold text-text-primary">Market competitiveness over time</h2>
        <p className="mb-4 text-sm text-text-secondary">
          This chart measures price dispersion (the average variation of prices for the same product across
          stores). A lower value indicates a more competitive market.
        </p>
        <div className="mb-4 grid grid-cols-1 gap-4 sm:grid-cols-2">
          <StatTile label="Latest market dispersion" value={dynamics.latest_dispersion.toFixed(3)} />
          <StatTile
            label="vs. average"
            value={`${(dynamics.latest_dispersion - dynamics.avg_dispersion >= 0 ? "+" : "") + (dynamics.latest_dispersion - dynamics.avg_dispersion).toFixed(3)}`}
            help={`Average across the observed period: ${dynamics.avg_dispersion.toFixed(3)}`}
          />
        </div>
        <div className="rounded-lg border border-border bg-surface p-5">
          <DispersionLineChart data={chartData} />
        </div>
        <p className="mt-4 text-xs text-text-secondary">
          <strong className="text-text-primary">Insight:</strong> After a volatile period in mid-January, the market
          settled into a stable equilibrium. The level of price difference between retailers is not escalating into
          a price war, nor is it diminishing.
        </p>
      </section>

      <section>
        <h2 className="mb-4 text-lg font-semibold text-text-primary">Who leads and who follows?</h2>
        <p className="mb-4 text-sm text-text-secondary">
          An entry &ldquo;A leads B&rdquo; means A&apos;s price changes tend to happen before B&apos;s.
        </p>
        {dynamics.top_leader && dynamics.fastest_follower && (
          <div className="mb-4 grid grid-cols-1 gap-4 sm:grid-cols-2">
            <StatTile label="Primary mover" value={dynamics.top_leader} help="Most frequently leads price changes." />
            <StatTile
              label="Fastest follower"
              value={dynamics.fastest_follower.follower}
              help={`Reacts within ${Math.abs(dynamics.fastest_follower.median_lag_days).toFixed(1)} days of ${dynamics.fastest_follower.leader}`}
            />
          </div>
        )}
        <LeadershipList rows={dynamics.leadership} />
        <p className="mt-4 text-xs text-text-secondary">
          <strong className="text-text-primary">Insight:</strong> Aldi is a primary price-setter, with the
          &ldquo;Big Four&rdquo; following its changes with a lag of several days. Among the Big Four, the
          relationships are much faster and more reciprocal, indicating a tight, reactive competitive cluster.
        </p>
      </section>
    </div>
  );
}
