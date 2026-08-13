import { supermarketColor } from "@/lib/colors";
import type { LeadershipRow } from "@/lib/artifacts";
import { canonicalLeadershipEdges } from "@/lib/leadership";

interface LeadershipListProps {
  rows: LeadershipRow[];
}

/**
 * "Who leads, who follows" as a sorted directed list -- a simplification
 * of the legacy dashboard's physics-simulated force-directed graph
 * (`streamlit_agraph`), which has no practical, dependency-light React
 * equivalent. Each row still carries the same information (leader,
 * follower, lag), just as a scannable list rather than a node diagram.
 *
 * Deduplicated via `canonicalLeadershipEdges` -- see that module's
 * docstring for the duplicate-row bug this fixes.
 */
export function LeadershipList({ rows }: LeadershipListProps) {
  const edges = canonicalLeadershipEdges(rows).sort((a, b) => b.lagDays - a.lagDays);

  return (
    <div className="space-y-2">
      {edges.map((edge) => (
        <div
          key={`${edge.leader}-${edge.follower}`}
          className="flex items-center gap-3 rounded-bento-sm border border-glass-border bg-surface px-4 py-2.5 text-sm"
        >
          <span className="flex items-center gap-1.5 font-medium text-text-primary">
            <span className="inline-block h-2.5 w-2.5 rounded-full" style={{ background: supermarketColor(edge.leader) }} />
            {edge.leader}
          </span>
          <span className="text-text-muted">leads</span>
          <span className="flex items-center gap-1.5 font-medium text-text-primary">
            <span className="inline-block h-2.5 w-2.5 rounded-full" style={{ background: supermarketColor(edge.follower) }} />
            {edge.follower}
          </span>
          <span className="ml-auto text-text-secondary">by {edge.lagDays.toFixed(0)} days</span>
          <span className="text-xs text-text-muted">({edge.nProducts.toLocaleString("en-GB")} products)</span>
        </div>
      ))}
    </div>
  );
}
