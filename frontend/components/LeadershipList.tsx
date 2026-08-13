import { supermarketColor } from "@/lib/colors";
import type { LeadershipRow } from "@/lib/artifacts";

interface LeadershipListProps {
  rows: LeadershipRow[];
}

/**
 * "Who leads, who follows" as a sorted directed list -- a simplification
 * of the legacy dashboard's physics-simulated force-directed graph
 * (`streamlit_agraph`), which has no practical, dependency-light React
 * equivalent. Each row still carries the same information (leader,
 * follower, lag), just as a scannable list rather than a node diagram.
 */
export function LeadershipList({ rows }: LeadershipListProps) {
  const sorted = [...rows].sort((a, b) => Math.abs(b.median_lag_days) - Math.abs(a.median_lag_days));

  return (
    <div className="space-y-2">
      {sorted.map((row) => {
        const [leader, follower] =
          row.median_lag_days >= 0 ? [row.leader, row.follower] : [row.follower, row.leader];
        return (
          <div
            key={`${row.leader}-${row.follower}`}
            className="flex items-center gap-3 rounded border border-border bg-surface px-4 py-2.5 text-sm"
          >
            <span className="flex items-center gap-1.5 font-medium text-text-primary">
              <span className="inline-block h-2.5 w-2.5 rounded-full" style={{ background: supermarketColor(leader) }} />
              {leader}
            </span>
            <span className="text-text-muted">leads</span>
            <span className="flex items-center gap-1.5 font-medium text-text-primary">
              <span className="inline-block h-2.5 w-2.5 rounded-full" style={{ background: supermarketColor(follower) }} />
              {follower}
            </span>
            <span className="ml-auto text-text-secondary">by {Math.abs(row.median_lag_days).toFixed(0)} days</span>
            <span className="text-xs text-text-muted">({row.n_products_analyzed.toLocaleString("en-GB")} products)</span>
          </div>
        );
      })}
    </div>
  );
}
