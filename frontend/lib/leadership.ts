import type { LeadershipRow } from "@/lib/artifacts";

export interface CanonicalEdge {
  leader: string;
  follower: string;
  lagDays: number;
  nProducts: number;
}

/**
 * One row per unordered retailer pair, direction resolved by sign --
 * deduplicates `price_leadership.parquet`'s two mirrored rows per pair
 * (bug found via direct data inspection: the artifact stores each pair
 * twice, once from each side's perspective, with opposite-signed lag --
 * e.g. {leader: ASDA, follower: Aldi, lag: -3} and {leader: Aldi,
 * follower: ASDA, lag: +3} both describe the same relationship). The
 * previous direction-normalizing display logic collapsed both onto an
 * identical rendered line, producing visible duplicate rows.
 */
export function canonicalLeadershipEdges(rows: LeadershipRow[]): CanonicalEdge[] {
  const seen = new Map<string, CanonicalEdge>();
  for (const row of rows) {
    const pairKey = [row.leader, row.follower].sort().join("::");
    if (seen.has(pairKey)) continue;
    const [leader, follower] = row.median_lag_days >= 0 ? [row.leader, row.follower] : [row.follower, row.leader];
    seen.set(pairKey, { leader, follower, lagDays: Math.abs(row.median_lag_days), nProducts: row.n_products_analyzed });
  }
  return [...seen.values()];
}
