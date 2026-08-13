"use client";

import { useMemo, useState } from "react";
import { BentoGrid } from "@/components/BentoGrid";
import { Callout } from "@/components/Callout";
import { GlassCard } from "@/components/GlassCard";
import { InfoTooltip } from "@/components/InfoTooltip";
import { Meter } from "@/components/Meter";
import { SimpleBarChart } from "@/components/charts/SimpleBarChart";
import { supermarketColor, SUPERMARKET_ORDER } from "@/lib/colors";
import { formatGBP, formatPercent } from "@/lib/format";
import type { BasketDetail } from "@/lib/artifacts";

interface BasketAnalysisClientProps {
  baskets: Record<string, BasketDetail>;
}

function averageCoverage(basket: BasketDetail): number {
  if (basket.rows.length === 0) return 0;
  return basket.rows.reduce((sum, row) => sum + row.coverage_pct, 0) / basket.rows.length;
}

export function BasketAnalysisClient({ baskets }: BasketAnalysisClientProps) {
  // Default to the best-coverage basket, not alphabetically-first --
  // a low-coverage basket as the very first thing a visitor sees reads
  // as "this is broken" rather than "matching is naturally partial".
  const basketNames = useMemo(
    () => Object.keys(baskets).sort((a, b) => averageCoverage(baskets[b]) - averageCoverage(baskets[a])),
    [baskets],
  );
  const [selected, setSelected] = useState(basketNames[0]);
  const [showItems, setShowItems] = useState(false);
  const basket = baskets[selected];

  const sortedRows = [...basket.rows].sort((a, b) => a.basket_cost - b.basket_cost);
  const chartData = sortedRows.map((row) => ({
    name: row.supermarket,
    value: row.basket_cost,
    color: supermarketColor(row.supermarket),
  }));

  const cheapest = sortedRows[0];
  const priciest = sortedRows[sortedRows.length - 1];
  const savingsPct = priciest && cheapest ? ((priciest.basket_cost - cheapest.basket_cost) / priciest.basket_cost) * 100 : 0;

  // canonical_name -> supermarket -> price, for the item-level grid
  const itemGrid = useMemo(() => {
    const grid = new Map<string, Map<string, number>>();
    for (const item of basket.items) {
      if (!grid.has(item.canonical_name)) grid.set(item.canonical_name, new Map());
      grid.get(item.canonical_name)!.set(item.supermarket, item.price);
    }
    return grid;
  }, [basket.items]);

  return (
    <div className="space-y-6">
      <label className="block text-sm">
        <span className="mb-1 block text-text-secondary">Choose a shopping basket to analyse:</span>
        <select
          value={selected}
          onChange={(e) => setSelected(e.target.value)}
          className="w-full max-w-sm rounded-bento-sm border border-glass-border bg-glass-surface px-3 py-2 text-text-primary backdrop-blur-glass-sm"
        >
          {basketNames.map((name) => (
            <option key={name} value={name}>
              {name}
            </option>
          ))}
        </select>
      </label>

      <BentoGrid>
        <GlassCard span={7}>
          <h3 className="mb-4 text-sm font-medium text-text-secondary">Cost of &lsquo;{selected}&rsquo; basket</h3>
          <SimpleBarChart data={chartData} format="currency" />
          {cheapest && priciest && cheapest.supermarket !== priciest.supermarket && (
            <div className="mt-4">
              <Callout>
                {`${selected} is cheapest at ${cheapest.supermarket} (${formatGBP(cheapest.basket_cost)}), ${savingsPct.toFixed(
                  0,
                )}% below the most expensive option (${priciest.supermarket}).`}
              </Callout>
            </div>
          )}
        </GlassCard>

        <GlassCard span={5}>
          <div className="mb-4 flex items-center gap-1.5">
            <h3 className="text-sm font-medium text-text-secondary">Coverage</h3>
            <InfoTooltip label="Why isn't coverage 100%?" align="end">
              <p>
                Coverage reflects how many of this basket&apos;s items an automated product-matching model could
                confidently link to a listing at that retailer. Not every store stocks every specific item, and some
                products don&apos;t have a confident match yet — partial coverage is expected, not a data error.
              </p>
            </InfoTooltip>
          </div>
          <div className="space-y-4">
            {sortedRows.map((row) => (
              <Meter
                key={row.supermarket}
                label={row.supermarket}
                value={row.coverage_pct}
                color={supermarketColor(row.supermarket)}
                valueLabel={`${row.items_found} / ${basket.total_items} (${formatPercent(row.coverage_pct)})`}
              />
            ))}
          </div>
        </GlassCard>
      </BentoGrid>

      <GlassCard variant="flat">
        <h3 className="mb-1 text-sm font-medium text-text-secondary">Detailed basket breakdown</h3>
        <p className="mb-4 text-xs text-text-secondary">
          Total cost and items found for the {basket.total_items}-item basket at each supermarket.
        </p>
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-border text-left text-text-muted">
              <th className="py-2 font-normal">Supermarket</th>
              <th className="py-2 text-right font-normal">Total cost</th>
              <th className="py-2 text-right font-normal">Items found</th>
            </tr>
          </thead>
          <tbody>
            {sortedRows.map((row) => (
              <tr key={row.supermarket} className="border-b border-border last:border-0">
                <td className="py-2 text-text-primary">{row.supermarket}</td>
                <td className="py-2 text-right text-text-primary">{formatGBP(row.basket_cost)}</td>
                <td className="py-2 text-right text-text-secondary">
                  {row.items_found} / {basket.total_items}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </GlassCard>

      <GlassCard variant="flat">
        <button
          type="button"
          onClick={() => setShowItems((v) => !v)}
          className="text-sm font-medium text-text-primary hover:underline"
        >
          {showItems ? "Hide" : "View"} products in this basket and their prices
        </button>
        {showItems && (
          <div className="mt-4 overflow-x-auto">
            <p className="mb-2 text-xs text-text-secondary">
              Showing {itemGrid.size} of {basket.total_items} products found in the database for the latest date.
            </p>
            <table className="w-full min-w-[600px] text-sm">
              <thead>
                <tr className="border-b border-border text-left text-text-muted">
                  <th className="py-2 pr-4 font-normal">Product</th>
                  {SUPERMARKET_ORDER.map((s) => (
                    <th key={s} className="py-2 text-right font-normal">
                      {s}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {[...itemGrid.entries()].map(([name, prices]) => (
                  <tr key={name} className="border-b border-border last:border-0">
                    <td className="py-2 pr-4 text-text-primary">{name}</td>
                    {SUPERMARKET_ORDER.map((s) => {
                      const price = prices.get(s);
                      return (
                        <td key={s} className="py-2 text-right text-text-secondary">
                          {price === undefined ? "—" : formatGBP(price)}
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </GlassCard>
    </div>
  );
}
