"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { ApiError, predictPrice, type PredictResponse } from "@/lib/api";
import { loadPredictorContext, type PredictorContextRow } from "@/lib/parquet";
import { formatGBP } from "@/lib/format";

// If the request is still pending after this long, it's very likely a
// Cloud Run cold start, not a slow network blip -- show an honest
// "waking up" state rather than a bare spinner (project_refactor.md §25.6).
const COLD_START_HINT_MS = 3000;

type LoadState = "loading" | "ready" | "error";

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
  }

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
    return (
      <p className="text-sm text-status-critical">
        Could not load the product catalogue. Refresh to try again.
      </p>
    );
  }

  return (
    <div className="space-y-6">
      <div className="rounded-lg border border-border bg-surface p-5">
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
            }}
            placeholder="e.g. 6 sweet creamy bananas"
            className="w-full rounded border border-border bg-surface px-3 py-2 text-text-primary"
          />
        </label>
        {matches.length > 0 && !selectedProduct && (
          <ul className="mt-2 max-h-56 divide-y divide-border overflow-y-auto rounded border border-border">
            {matches.map((name) => (
              <li key={name}>
                <button
                  type="button"
                  onClick={() => selectProduct(name)}
                  className="w-full px-3 py-2 text-left text-sm text-text-primary hover:bg-page"
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
                onChange={(e) => {
                  setSelectedStore(e.target.value || null);
                  setResult(null);
                }}
                className="w-full max-w-xs rounded border border-border bg-surface px-3 py-2 text-text-primary"
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
                className="w-full max-w-xs rounded border border-border bg-surface px-3 py-2 text-text-primary"
              />
            </label>

            <button
              type="button"
              disabled={!selectedRow || predicting}
              onClick={handlePredict}
              className="rounded bg-text-primary px-4 py-2 text-sm font-medium text-page disabled:opacity-50"
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
      </div>

      {predictError && (
        <div className="rounded-lg border border-status-critical/30 bg-surface p-4 text-sm text-status-critical">
          {predictError}
        </div>
      )}

      {result && (
        <div className="rounded-lg border border-border bg-surface p-5">
          <div className="text-sm text-text-secondary">Predicted price</div>
          <div className="mt-1 text-3xl font-semibold text-text-primary">{formatGBP(result.predicted_price)}</div>
          <dl className="mt-4 grid grid-cols-2 gap-2 text-xs text-text-secondary sm:grid-cols-4">
            <div>
              <dt className="text-text-muted">Requested date</dt>
              <dd className="text-text-primary">{result.requested_date}</dd>
            </div>
            <div>
              <dt className="text-text-muted">Resolved from</dt>
              <dd className="text-text-primary">{result.resolved_from_date}</dd>
            </div>
            <div>
              <dt className="text-text-muted">Override applied</dt>
              <dd className="text-text-primary">{result.price_override_applied ? "Yes" : "No"}</dd>
            </div>
            <div>
              <dt className="text-text-muted">Unresolved features</dt>
              <dd className="text-text-primary">{result.unresolved_features.length}</dd>
            </div>
          </dl>
          {result.unresolved_features.length > 0 && (
            <p className="mt-3 text-xs text-text-secondary">
              No real history was available for: {result.unresolved_features.join(", ")}. LightGBM handles these as
              missing values natively -- nothing was fabricated or zero-filled.
            </p>
          )}
        </div>
      )}
    </div>
  );
}
