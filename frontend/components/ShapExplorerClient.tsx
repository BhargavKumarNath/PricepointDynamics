"use client";

import { useEffect, useMemo, useState } from "react";
import { ContributionChart } from "@/components/charts/ContributionChart";
import { FeatureImportanceChart } from "@/components/charts/FeatureImportanceChart";
import { loadShapExplorer } from "@/lib/parquet";
import { formatGBP } from "@/lib/format";
import type { ShapExplorerMeta } from "@/lib/artifacts";

interface ShapExplorerClientProps {
  meta: ShapExplorerMeta;
}

export function ShapExplorerClient({ meta }: ShapExplorerClientProps) {
  const [data, setData] = useState<{ features: Record<string, number>[]; values: Record<string, number>[] } | null>(
    null,
  );
  const [error, setError] = useState<string | null>(null);
  const [sampleIndex, setSampleIndex] = useState(0);

  useEffect(() => {
    loadShapExplorer()
      .then(setData)
      .catch((err: unknown) => setError(err instanceof Error ? err.message : "Failed to load SHAP data."));
  }, []);

  const globalImportance = useMemo(() => {
    if (!data) return [];
    const sums = new Map<string, number>();
    for (const row of data.values) {
      for (const feature of meta.feature_names) {
        sums.set(feature, (sums.get(feature) ?? 0) + Math.abs(row[feature] ?? 0));
      }
    }
    return meta.feature_names
      .map((feature) => ({ feature, meanAbsShap: (sums.get(feature) ?? 0) / data.values.length }))
      .sort((a, b) => b.meanAbsShap - a.meanAbsShap)
      .slice(0, 15);
  }, [data, meta.feature_names]);

  const localExplanation = useMemo(() => {
    if (!data) return null;
    const valuesRow = data.values[sampleIndex];
    const featuresRow = data.features[sampleIndex];
    const contributions = meta.feature_names
      .map((feature) => ({ feature, shapValue: valuesRow[feature] ?? 0, featureValue: featuresRow[feature] }))
      .sort((a, b) => Math.abs(b.shapValue) - Math.abs(a.shapValue))
      .slice(0, 10);
    const prediction = meta.base_value + Object.values(valuesRow).reduce((sum, v) => sum + v, 0);
    return { contributions, prediction };
  }, [data, sampleIndex, meta.feature_names, meta.base_value]);

  if (error) {
    return <p className="text-sm text-status-critical">{error}</p>;
  }

  if (!data) {
    return <p className="text-sm text-text-secondary">Loading precomputed SHAP analysis…</p>;
  }

  return (
    <div className="space-y-10">
      <section className="rounded-lg border border-border bg-surface p-5">
        <h2 className="mb-1 text-lg font-semibold text-text-primary">
          Global feature importance: what drives prices overall?
        </h2>
        <p className="mb-4 text-sm text-text-secondary">
          Ranked by average impact across {meta.n_samples.toLocaleString("en-GB")} representative product samples.
        </p>
        <FeatureImportanceChart data={globalImportance} />
      </section>

      <section className="rounded-lg border border-border bg-surface p-5">
        <h2 className="mb-1 text-lg font-semibold text-text-primary">Local prediction explanations</h2>
        <p className="mb-4 text-sm text-text-secondary">
          Select a product instance to see how the model arrived at its forecast.
        </p>
        <label className="mb-4 block text-sm">
          <span className="mb-1 block text-text-secondary">Product sample</span>
          <select
            value={sampleIndex}
            onChange={(e) => setSampleIndex(Number(e.target.value))}
            className="w-full max-w-xs rounded border border-border bg-surface px-3 py-2 text-text-primary"
          >
            {Array.from({ length: data.features.length }, (_, i) => (
              <option key={i} value={i}>
                Product sample #{i + 1}
              </option>
            ))}
          </select>
        </label>

        {localExplanation && (
          <>
            <div className="mb-4 grid grid-cols-1 gap-4 sm:grid-cols-3">
              <div className="rounded border border-border p-4">
                <div className="text-xs text-text-secondary">Model&apos;s price prediction</div>
                <div data-testid="predicted-price" className="mt-1 text-xl font-semibold text-text-primary">
                  {formatGBP(localExplanation.prediction)}
                </div>
              </div>
              <div className="rounded border border-border p-4">
                <div className="text-xs text-text-secondary">Base value (average)</div>
                <div className="mt-1 text-xl font-semibold text-text-primary">{formatGBP(meta.base_value)}</div>
              </div>
              <div className="rounded border border-border p-4">
                <div className="text-xs text-text-secondary">Prediction difference</div>
                <div className="mt-1 text-xl font-semibold text-text-primary">
                  {localExplanation.prediction - meta.base_value >= 0 ? "+" : ""}
                  {formatGBP(localExplanation.prediction - meta.base_value)}
                </div>
              </div>
            </div>
            <h3 className="mb-2 text-sm font-medium text-text-secondary">
              Top contributing features (red pushes the price up, blue pushes it down)
            </h3>
            <ContributionChart data={localExplanation.contributions} />
          </>
        )}
      </section>
    </div>
  );
}
