import { GlassCard } from "@/components/GlassCard";

export interface MetricCardDelta {
  value: string;
  direction: "up" | "down" | "flat";
  /** Which direction counts as "good," if any. Omit when the metric has
   * no inherent good/bad judgment -- renders neutral rather than
   * inventing one the data doesn't support. */
  goodDirection?: "up" | "down";
}

interface MetricCardProps {
  label: string;
  value: string;
  help?: string;
  icon?: React.ReactNode;
  delta?: MetricCardDelta;
  /** Raw series for a hand-rolled sparkline -- no charting library. */
  trend?: number[];
  span?: 3 | 4 | 6 | 12;
  /** "brand" = the one featured/hero metric on a page. */
  tone?: "default" | "brand";
}

function deltaColor(delta: MetricCardDelta): string {
  if (!delta.goodDirection || delta.direction === "flat") return "text-text-secondary";
  return delta.direction === delta.goodDirection ? "text-status-good" : "text-status-critical";
}

function Sparkline({ values }: { values: number[] }) {
  if (values.length < 2) return null;
  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = max - min || 1;
  const points = values
    .map((v, i) => {
      const x = (i / (values.length - 1)) * 64;
      const y = 20 - ((v - min) / range) * 18 - 1;
      return `${x},${y}`;
    })
    .join(" ");
  const lastX = 64;
  const lastY = 20 - ((values[values.length - 1] - min) / range) * 18 - 1;
  return (
    <svg width={64} height={20} viewBox="0 0 64 20" className="shrink-0">
      <polyline points={points} fill="none" stroke="var(--text-muted)" strokeWidth={1.5} />
      <circle cx={lastX} cy={lastY} r={2.5} fill="var(--brand)" />
    </svg>
  );
}

/** Bento stat tile -- upgrades the earlier `StatTile` with an optional
 * icon, delta badge, and sparkline, per the dataviz skill's stat-tile
 * contract (label / value / delta / trend). */
export function MetricCard({ label, value, help, icon, delta, trend, span = 4, tone = "default" }: MetricCardProps) {
  return (
    <GlassCard span={span} accent={tone === "brand"}>
      <div className="flex items-start justify-between gap-3">
        <div>
          <div className="text-sm text-text-secondary">{label}</div>
          <div className={`mt-1 text-3xl font-semibold ${tone === "brand" ? "text-brand" : "text-text-primary"}`}>
            {value}
          </div>
        </div>
        {icon && (
          <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-full bg-brand-soft text-brand">
            {icon}
          </div>
        )}
      </div>
      <div className="mt-2 flex items-center justify-between gap-2">
        <div>
          {help && <div className="text-xs text-text-muted">{help}</div>}
          {delta && <div className={`text-xs font-medium ${deltaColor(delta)}`}>{delta.value}</div>}
        </div>
        {trend && <Sparkline values={trend} />}
      </div>
    </GlassCard>
  );
}
