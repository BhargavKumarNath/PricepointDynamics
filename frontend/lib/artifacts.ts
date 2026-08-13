/**
 * Server-side reads of the small precomputed JSON artifacts
 * (project_refactor.md §25.3) -- called only from Server Components, so
 * these run at `next build` time (static export) and never ship to the
 * browser as JS. Reading `public/data/*.json` directly via `fs` rather
 * than an HTTP fetch to a localhost dev server -- there's no server in
 * the build process to fetch from.
 */
import { readFile } from "node:fs/promises";
import path from "node:path";

const DATA_DIR = path.join(process.cwd(), "public", "data");

async function readJson<T>(filename: string): Promise<T> {
  const raw = await readFile(path.join(DATA_DIR, filename), "utf-8");
  return JSON.parse(raw) as T;
}

export interface HomeMetrics {
  total_records: number;
  canonical_products: number;
  mae: number;
  rmse: number;
  r2: number;
  n_train_rows: number;
  n_test_rows: number;
  n_features: number;
  trained_at: string;
}

export function loadHomeMetrics(): Promise<HomeMetrics> {
  return readJson<HomeMetrics>("home_metrics.json");
}

export interface MarketOverviewRow {
  supermarket: string;
  portfolio_size: number;
  own_brand_pct: number;
  own_brand_count: number;
  branded_count: number;
  min_price: number;
  price_p25: number;
  price_median: number;
  price_p75: number;
  max_price: number;
}

export function loadMarketOverview(): Promise<{ supermarkets: MarketOverviewRow[] }> {
  return readJson("market_overview.json");
}

export interface BasketCostRow {
  supermarket: string;
  basket_cost: number;
  items_found: number;
  coverage_pct: number;
}

export interface BasketItemPrice {
  canonical_name: string;
  supermarket: string;
  price: number;
}

export interface BasketDetail {
  total_items: number;
  rows: BasketCostRow[];
  items: BasketItemPrice[];
}

export function loadBasketAnalysis(): Promise<{ baskets: Record<string, BasketDetail> }> {
  return readJson("basket_analysis.json");
}

export interface DispersionPoint {
  date: string;
  dispersion: number;
}

export interface LeadershipRow {
  leader: string;
  follower: string;
  median_lag_days: number;
  n_products_analyzed: number;
}

export interface MarketDynamics {
  dispersion: DispersionPoint[];
  leadership: LeadershipRow[];
  latest_dispersion: number;
  avg_dispersion: number;
  top_leader: string | null;
  fastest_follower: { follower: string; leader: string; median_lag_days: number } | null;
}

export function loadMarketDynamics(): Promise<MarketDynamics> {
  return readJson("market_dynamics.json");
}

export interface ShapExplorerMeta {
  base_value: number;
  feature_names: string[];
  n_samples: number;
}

export function loadShapExplorerMeta(): Promise<ShapExplorerMeta> {
  return readJson("shap_explorer_meta.json");
}
