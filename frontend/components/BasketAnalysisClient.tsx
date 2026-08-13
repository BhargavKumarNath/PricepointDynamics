"use client";

import { useMemo, useState } from "react";
import { SimpleBarChart } from "@/components/charts/SimpleBarChart";
import { supermarketColor, SUPERMARKET_ORDER } from "@/lib/colors";
import { formatGBP, formatPercent } from "@/lib/format";
import type { BasketDetail } from "@/lib/artifacts";

interface BasketAnalysisClientProps {
  baskets: Record<string, BasketDetail>;
}

export function BasketAnalysisClient({ baskets }: BasketAnalysisClientProps) {
  const basketNames = useMemo(() => Object.keys(baskets).sort(), [baskets]);
  const [selected, setSelected] = useState(basketNames[0]);
  const [showItems, setShowItems] = useState(false);
  const basket = baskets[selected];

  const chartData = [...basket.rows]
    .sort((a, b) => a.basket_cost - b.basket_cost)
    .map((row) => ({ name: row.supermarket, value: row.basket_cost, color: supermarketColor(row.supermarket) }));

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
          className="w-full max-w-sm rounded border border-border bg-surface px-3 py-2 text-text-primary"
        >
          {basketNames.map((name) => (
            <option key={name} value={name}>
              {name}
            </option>
          ))}
        </select>
      </label>

      <div className="rounded-lg border border-border bg-surface p-5">
        <h3 className="mb-4 text-sm font-medium text-text-secondary">Cost of &lsquo;{selected}&rsquo; basket</h3>
        <SimpleBarChart data={chartData} format="currency" />
      </div>

      <div className="rounded-lg border border-border bg-surface p-5">
        <h3 className="mb-1 text-sm font-medium text-text-secondary">Detailed basket breakdown</h3>
        <p className="mb-4 text-xs text-text-secondary">
          Total cost, items found, and coverage of the {basket.total_items}-item basket at each supermarket.
        </p>
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-border text-left text-text-muted">
              <th className="py-2 font-normal">Supermarket</th>
              <th className="py-2 text-right font-normal">Total cost</th>
              <th className="py-2 text-right font-normal">Items found</th>
              <th className="py-2 text-right font-normal">Coverage</th>
            </tr>
          </thead>
          <tbody>
            {[...basket.rows]
              .sort((a, b) => a.basket_cost - b.basket_cost)
              .map((row) => (
                <tr key={row.supermarket} className="border-b border-border last:border-0">
                  <td className="py-2 text-text-primary">{row.supermarket}</td>
                  <td className="py-2 text-right text-text-primary">{formatGBP(row.basket_cost)}</td>
                  <td className="py-2 text-right text-text-secondary">
                    {row.items_found} / {basket.total_items}
                  </td>
                  <td className="py-2 text-right text-text-secondary">{formatPercent(row.coverage_pct)}</td>
                </tr>
              ))}
          </tbody>
        </table>
      </div>

      <div className="rounded-lg border border-border bg-surface p-5">
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
      </div>
    </div>
  );
}
