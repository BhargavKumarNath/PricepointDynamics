"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { BentoGrid } from "@/components/BentoGrid";
import { GlassCard } from "@/components/GlassCard";
import { MetricCard } from "@/components/MetricCard";
import { TargetIcon } from "@/components/icons";
import { PriceHistoryChart } from "@/components/charts/PriceHistoryChart";
import { ApiError, getProductHistory, predictPrice, type ProductHistoryResponse, type PredictResponse } from "@/lib/api";
import { humanizeFeature } from "@/lib/featureLabels";
import { loadPredictorContext, type PredictorContextRow } from "@/lib/parquet";
import { formatGBP } from "@/lib/format";

// If the request is still pending after this long, it's very likely a
// Cloud Run cold start, not a slow network blip -- show an honest
// "waking up" state rather than a bare spinner (project_refactor.md §25.6).
const COLD_START_HINT_MS = 3000;

type LoadState = "loading" | "ready" | "error";
type HistoryState = "idle" | "loading" | "ready" | "error";

export function PredictorClient() {
  const [context, setContext] = useState<PredictorContextRow[]>([]);
  const [loadState, setLoadState] = useState<LoadState>("loading");
  const [query, setQuery] = useState("");
  const [selectedProduct, setSelectedProduct] = useState<string | null>(null);
  const [selectedStore, setSelectedStore] = useState<string | null>(null);
  const [priceOverride, setPriceOverride] = useState("");
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [predictError, setPredictError] = useState<string | null>(null);
  const [predicting, setPredicting] = useState(false);
  const [showColdStartHint, setShowColdStartHint] = useState(false);
  const coldStartTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  const [history, setHistory] = useState<ProductHistoryResponse | null>(null);
  const [historyState, setHistoryState] = useState<HistoryState>("idle");

  useEffect(() => {
    loadPredictorContext()
      .then((rows) => {
        setContext(rows);
        setLoadState("ready");
      })
      .catch(() => setLoadState("error"));
  }, []);

  const matches = useMemo(() => {
    if (query.trim().length < 2) return [];
    const q = query.toLowerCase();
    const seen = new Set<string>();
    const out: string[] = [];
    for (const row of context) {
      if (row.canonical_name.toLowerCase().includes(q) && !seen.has(row.canonical_name)) {
        seen.add(row.canonical_name);
        out.push(row.canonical_name);
        if (out.length >= 50) break;
      }
    }
    return out;
  }, [context, query]);

  const storesForProduct = useMemo(() => {
    if (!selectedProduct) return [];
    return context.filter((row) => row.canonical_name === selectedProduct);
  }, [context, selectedProduct]);

  const selectedRow = storesForProduct.find((row) => row.supermarket === selectedStore) ?? null;

  function selectProduct(name: string) {
    setSelectedProduct(name);
    setQuery(name);
    setSelectedStore(null);
    setResult(null);
    setPredictError(null);
    setHistory(null);
    setHistoryState("idle");
  }

  // The synchronous "we're now loading" state reset belongs in the event
  // handler that causes it, not in the effect that performs the fetch --
  // avoids a cascading synchronous setState directly in the effect body.
  function selectStore(store: string | null) {
    setSelectedStore(store);
    setResult(null);
    setHistory(null);
    setHistoryState(store ? "loading" : "idle");
  }

  useEffect(() => {
    if (!selectedProduct || !selectedStore) return;
    let cancelled = false;
    getProductHistory(selectedProduct, selectedStore)
      .then((res) => {
        if (cancelled) return;
        setHistory(res);
        setHistoryState("ready");
      })
      .catch(() => {
        if (cancelled) return;
        setHistoryState("error");
      });
    return () => {
      cancelled = true;
    };
  }, [selectedProduct, selectedStore]);

  async function handlePredict() {
    if (!selectedRow) return;
    setPredicting(true);
    setPredictError(null);
    setResult(null);
    coldStartTimer.current = setTimeout(() => setShowColdStartHint(true), COLD_START_HINT_MS);

    try {
      const response = await predictPrice({
        canonical_name: selectedRow.canonical_name,
        supermarket: selectedRow.supermarket,
        date: selectedRow.date,
        ...(priceOverride ? { price_override: Number(priceOverride) } : {}),
      });
      setResult(response);
    } catch (err) {
      if (err instanceof ApiError) {
        setPredictError(err.message);
      } else {
        setPredictError("Could not reach the prediction service. It may be waking up -- please try again shortly.");
      }
    } finally {
      if (coldStartTimer.current) clearTimeout(coldStartTimer.current);
      setShowColdStartHint(false);
      setPredicting(false);
    }
  }

  if (loadState === "loading") {
    return <p className="text-sm text-text-secondary">Loading product catalogue…</p>;
  }
  if (loadState === "error") {
    return <p className="text-sm text-status-critical">Could not load the product catalogue. Refresh to try again.</p>;
  }

  const chartHistory =
    history?.history.map((h) => ({
      date: h.date,
      avgPrice: h.avg_price,
      minPrice: h.min_price,
      maxPrice: h.max_price,
    })) ?? [];

  const predictedPoint = result ? { date: result.requested_date, price: result.predicted_price } : null;

  const lastKnownPrice = history?.history[history.history.length - 1]?.avg_price ?? null;
  const delta = result && lastKnownPrice != null ? result.predicted_price - lastKnownPrice : null;

  return (
    <BentoGrid>
      <GlassCard span={5} className="h-fit">
        <label className="block text-sm">
          <span className="mb-1 block text-text-secondary">Search for a product</span>
          <input
            type="text"
            value={query}
            onChange={(e) => {
              setQuery(e.target.value);
              setSelectedProduct(null);
              setSelectedStore(null);
              setResult(null);
              setHistory(null);
              setHistoryState("idle");
            }}
            placeholder="e.g. 6 sweet creamy bananas"
            className="w-full rounded-bento-sm border border-glass-border bg-glass-surface px-3 py-2 text-text-primary backdrop-blur-glass-sm"
          />
        </label>
        {matches.length > 0 && !selectedProduct && (
          <ul className="mt-2 max-h-56 divide-y divide-border overflow-y-auto rounded-bento-sm border border-glass-border">
            {matches.map((name) => (
              <li key={name}>
                <button
                  type="button"
                  onClick={() => selectProduct(name)}
                  className="w-full px-3 py-2 text-left text-sm text-text-primary hover:bg-brand-soft"
                >
                  {name}
                </button>
              </li>
            ))}
          </ul>
        )}

        {selectedProduct && (
          <div className="mt-4 space-y-4">
            <label className="block text-sm">
              <span className="mb-1 block text-text-secondary">Supermarket</span>
              <select
                value={selectedStore ?? ""}
                onChange={(e) => selectStore(e.target.value || null)}
                className="w-full max-w-xs rounded-bento-sm border border-glass-border bg-glass-surface px-3 py-2 text-text-primary backdrop-blur-glass-sm"
              >
                <option value="">Choose a store…</option>
                {storesForProduct.map((row) => (
                  <option key={row.supermarket} value={row.supermarket}>
                    {row.supermarket}
                  </option>
                ))}
              </select>
            </label>

            {selectedRow && (
              <p className="text-xs text-text-secondary">
                Latest known price at {selectedRow.supermarket} ({selectedRow.date}):{" "}
                <strong className="text-text-primary">{formatGBP(selectedRow.prices)}</strong>
              </p>
            )}

            <label className="block text-sm">
              <span className="mb-1 block text-text-secondary">
                Override yesterday&apos;s price (optional -- &ldquo;what if&rdquo; scenario)
              </span>
              <input
                type="number"
                step="0.01"
                min="0"
                value={priceOverride}
                onChange={(e) => setPriceOverride(e.target.value)}
                placeholder={selectedRow?.price_lag_1d != null ? formatGBP(selectedRow.price_lag_1d) : "£"}
                className="w-full max-w-xs rounded-bento-sm border border-glass-border bg-glass-surface px-3 py-2 text-text-primary backdrop-blur-glass-sm"
              />
            </label>

            <button
              type="button"
              disabled={!selectedRow || predicting}
              onClick={handlePredict}
              className="rounded-bento-sm bg-brand px-4 py-2 text-sm font-medium text-page disabled:opacity-50"
            >
              {predicting ? "Predicting…" : "Predict price"}
            </button>
            {showColdStartHint && (
              <p className="text-xs text-text-secondary">
                The prediction service is waking up (cold start) -- this can take up to ~15s. Hang tight…
              </p>
            )}
          </div>
        )}
      </GlassCard>

      <GlassCard span={7}>
        {!selectedProduct && (
          <div className="flex h-full min-h-[280px] flex-col items-center justify-center gap-3 text-center">
            <div className="flex h-12 w-12 items-center justify-center rounded-full bg-brand-soft text-brand">
              <TargetIcon size={22} />
            </div>
            <p className="max-w-xs text-sm text-text-secondary">
              Search a product on the left to see its price history and get a live prediction.
            </p>
          </div>
        )}

        {selectedProduct && !selectedStore && (
          <div className="flex h-full min-h-[280px] items-center justify-center text-center text-sm text-text-secondary">
            Choose a store to load {selectedProduct}&apos;s price history.
          </div>
        )}

        {selectedStore && historyState === "loading" && (
          <div className="flex h-full min-h-[280px] items-center justify-center text-sm text-text-secondary">
            Loading price history…
          </div>
        )}

        {selectedStore && historyState === "error" && (
          <p className="text-sm text-status-critical">Could not load price history for this product/store.</p>
        )}

        {selectedStore && historyState === "ready" && (
          <>
            <h3 className="mb-4 text-sm font-medium text-text-secondary">
              Price history at {selectedStore}
              {result ? " + prediction" : ""}
            </h3>
            <PriceHistoryChart history={chartHistory} predicted={predictedPoint} />
          </>
        )}

        {predictError && (
          <div className="mt-4 rounded-bento-sm border border-status-critical/30 bg-status-critical/10 p-4 text-sm text-status-critical">
            {predictError}
          </div>
        )}

        {result && (
          <div className="mt-4">
            <BentoGrid columns={6}>
              <MetricCard
                span={3}
                tone="brand"
                label="Predicted price"
                value={formatGBP(result.predicted_price)}
                help={`Resolved from ${result.resolved_from_date}`}
              />
              {delta != null && (
                <MetricCard
                  span={3}
                  label="Vs. last known price"
                  value={`${delta >= 0 ? "+" : ""}${formatGBP(delta)}`}
                  help={lastKnownPrice != null ? `Last known: ${formatGBP(lastKnownPrice)}` : undefined}
                />
              )}
            </BentoGrid>
            {result.unresolved_features.length > 0 && (
              <p className="mt-3 text-xs text-text-secondary">
                Some contextual signals weren&apos;t available for this product/date (
                {result.unresolved_features.map((f) => humanizeFeature(f)).join(", ")}) -- the model handles these
                as missing values natively, nothing was fabricated or zero-filled.
              </p>
            )}
          </div>
        )}
      </GlassCard>
    </BentoGrid>
  );
}
