"use client";

import { Fragment, useState } from "react";
import { supermarketColor, SUPERMARKET_ORDER } from "@/lib/colors";
import { canonicalLeadershipEdges, type CanonicalEdge } from "@/lib/leadership";
import type { LeadershipRow } from "@/lib/artifacts";

interface LeadershipMatrixProps {
  rows: LeadershipRow[];
}

/**
 * Row = leader, column = follower -- each unordered pair is placed in
 * exactly ONE cell (via `canonicalLeadershipEdges`, the same
 * deduplication used by `LeadershipList`), never both mirror
 * positions, since the raw artifact's `leader`/`follower` labels don't
 * reliably indicate the true direction on their own (see
 * `lib/leadership.ts`'s docstring).
 *
 * Cell fill encodes *coupling strength* (1/lagDays) on the design
 * system's default sequential hue (blue) -- a short lag (tight,
 * reactive pair) is the strong signal here, so it gets the more
 * saturated step; a long lag fades toward the surface. This is a
 * deliberate inversion of "near-zero fades," because in this specific
 * chart "near zero days" is the strongest relationship, not the
 * weakest.
 */
export function LeadershipMatrix({ rows }: LeadershipMatrixProps) {
  const [hovered, setHovered] = useState<string | null>(null);
  const edges = canonicalLeadershipEdges(rows);
  const cellOf = new Map<string, CanonicalEdge>(edges.map((e) => [`${e.leader}::${e.follower}`, e]));

  const lags = edges.map((e) => e.lagDays).filter((l) => l > 0);
  const minLag = Math.min(...lags);
  const maxLag = Math.max(...lags);

  function fillFor(edge: CanonicalEdge): string {
    if (maxLag === minLag) return "color-mix(in oklab, var(--series-asda) 55%, var(--surface))";
    // Invert + normalize: shortest lag -> strongest fill (~90%), longest -> weakest (~15%).
    const strength = 1 - (edge.lagDays - minLag) / (maxLag - minLag);
    const pct = 15 + strength * 75;
    return `color-mix(in oklab, var(--series-asda) ${pct}%, var(--surface))`;
  }

  return (
    <div>
      <div className="overflow-x-auto">
        <div
          className="grid gap-1"
          style={{ gridTemplateColumns: `88px repeat(${SUPERMARKET_ORDER.length}, minmax(64px, 1fr))` }}
        >
          <div />
          {SUPERMARKET_ORDER.map((follower) => (
            <div key={follower} className="flex flex-col items-center gap-1 pb-1 text-center">
              <span className="h-2 w-2 rounded-full" style={{ background: supermarketColor(follower) }} />
              <span className="text-[11px] text-text-secondary">{follower}</span>
            </div>
          ))}

          {SUPERMARKET_ORDER.map((leader) => (
            <Fragment key={leader}>
              <div className="flex items-center gap-1.5 pr-2 text-[11px] text-text-secondary">
                <span className="h-2 w-2 shrink-0 rounded-full" style={{ background: supermarketColor(leader) }} />
                {leader}
              </div>
              {SUPERMARKET_ORDER.map((follower) => {
                const key = `${leader}::${follower}`;
                const edge = leader === follower ? undefined : cellOf.get(key);
                return (
                  <button
                    key={key}
                    type="button"
                    disabled={!edge}
                    onMouseEnter={() => edge && setHovered(key)}
                    onMouseLeave={() => setHovered((v) => (v === key ? null : v))}
                    onFocus={() => edge && setHovered(key)}
                    onBlur={() => setHovered((v) => (v === key ? null : v))}
                    aria-label={
                      edge
                        ? `${edge.leader} leads ${edge.follower} by ${edge.lagDays} days, based on ${edge.nProducts} products`
                        : "No significant leader-follower relationship measured"
                    }
                    className="flex aspect-square items-center justify-center rounded-bento-sm text-xs font-medium transition-transform hover:scale-105 disabled:cursor-default"
                    style={{
                      background: edge ? fillFor(edge) : "color-mix(in oklab, var(--text-muted) 6%, var(--surface))",
                      color: edge ? "var(--text-primary)" : "var(--text-muted)",
                    }}
                  >
                    {edge ? `${edge.lagDays}d` : "—"}
                  </button>
                );
              })}
            </Fragment>
          ))}
        </div>
      </div>

      {hovered && cellOf.has(hovered) && (
        <p className="mt-3 text-xs text-text-secondary">
          {(() => {
            const e = cellOf.get(hovered)!;
            return `${e.leader} leads ${e.follower} by ${e.lagDays} days, based on ${e.nProducts.toLocaleString("en-GB")} products.`;
          })()}
        </p>
      )}

      <div className="mt-3 flex items-center gap-2 text-[11px] text-text-muted">
        <span
          className="h-2 w-8 rounded-full"
          style={{ background: "linear-gradient(to right, color-mix(in oklab, var(--series-asda) 15%, var(--surface)), color-mix(in oklab, var(--series-asda) 90%, var(--surface)))" }}
        />
        <span>Slower reaction → Tighter coupling</span>
      </div>
      <p className="mt-2 text-xs text-text-muted">
        Read row → column: the row retailer leads; the column retailer follows, that many days later.
      </p>
    </div>
  );
}
