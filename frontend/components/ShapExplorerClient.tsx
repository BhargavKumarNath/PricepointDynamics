"use client";

import { useEffect, useMemo, useState } from "react";
import { GlassCard } from "@/components/GlassCard";
import { ContributionChart } from "@/components/charts/ContributionChart";
import { FeatureImportanceChart } from "@/components/charts/FeatureImportanceChart";
import { FEATURE_GROUPS, groupContributions, humanizeFeature } from "@/lib/featureLabels";
import { loadShapExplorer } from "@/lib/parquet";
import { formatGBP } from "@/lib/format";
import type { ShapExplorerMeta } from "@/lib/artifacts";

interface ShapExplorerClientProps {
  meta: ShapExplorerMeta;
}

type DetailLevel = "simplified" | "detailed";

export function ShapExplorerClient({ meta }: ShapExplorerClientProps) {
  const [data, setData] = useState<{ features: Record<string, number>[]; values: Record<string, number>[] } | null>(
    null,
  );
  const [error, setError] = useState<string | null>(null);
  const [sampleIndex, setSampleIndex] = useState(0);
  const [detailLevel, setDetailLevel] = useState<DetailLevel>("simplified");

  useEffect(() => {
    loadShapExplorer()
      .then(setData)
      .catch((err: unknown) => setError(err instanceof Error ? err.message : "Failed to load SHAP data."));
  }, []);

  // Detailed: mean(|shap|) per raw feature, independent of grouping.
  const detailedGlobalImportance = useMemo(() => {
    if (!data) return [];
    const sums = new Map<string, number>();
    for (const row of data.values) {
      for (const feature of meta.feature_names) {
        sums.set(feature, (sums.get(feature) ?? 0) + Math.abs(row[feature] ?? 0));
      }
    }
    return meta.feature_names
      .map((feature) => ({ feature: humanizeFeature(feature), meanAbsShap: (sums.get(feature) ?? 0) / data.values.length }))
      .sort((a, b) => b.meanAbsShap - a.meanAbsShap)
      .slice(0, 15);
  }, [data, meta.feature_names]);

  // Simplified: group per-sample first (signed sum within a group, so
  // offsetting cyclical/one-hot members don't inflate the group), THEN
  // take mean(|grouped value|) across samples -- not the same as
  // summing each member's own mean(|shap|), which would overstate a
  // group where members partially cancel out per sample.
  const simplifiedGlobalImportance = useMemo(() => {
    if (!data) return [];
    const groupSums = new Map<string, number>();
    for (let i = 0; i < data.values.length; i++) {
      const valuesRow = data.values[i];
      const featuresRow = data.features[i];
      const rows = meta.feature_names.map((feature) => ({
        feature,
        shapValue: valuesRow[feature] ?? 0,
        featureValue: featuresRow[feature],
      }));
      for (const g of groupContributions(rows)) {
        groupSums.set(g.id, (groupSums.get(g.id) ?? 0) + Math.abs(g.shapValue));
      }
    }
    return [...groupSums.entries()]
      .map(([id, total]) => {
        const groupDef = FEATURE_GROUPS.find((g) => g.id === id);
        return { feature: groupDef?.label ?? humanizeFeature(id), meanAbsShap: total / data.values.length };
      })
      .sort((a, b) => b.meanAbsShap - a.meanAbsShap);
  }, [data, meta.feature_names]);

  const globalImportance = detailLevel === "simplified" ? simplifiedGlobalImportance : detailedGlobalImportance;

  const localExplanation = useMemo(() => {
    if (!data) return null;
    const valuesRow = data.values[sampleIndex];
    const featuresRow = data.features[sampleIndex];
    const rawRows = meta.feature_names.map((feature) => ({
      feature,
      shapValue: valuesRow[feature] ?? 0,
      featureValue: featuresRow[feature],
    }));
    const contributions =
      detailLevel === "simplified"
        ? groupContributions(rawRows)
            .slice(0, 10)
            .map((g) => ({ feature: g.label, shapValue: g.shapValue }))
        : [...rawRows]
            .sort((a, b) => Math.abs(b.shapValue) - Math.abs(a.shapValue))
            .slice(0, 10)
            .map((r) => ({ feature: humanizeFeature(r.feature), shapValue: r.shapValue }));
    const prediction = meta.base_value + Object.values(valuesRow).reduce((sum, v) => sum + v, 0);
    return { contributions, prediction };
  }, [data, sampleIndex, meta.feature_names, meta.base_value, detailLevel]);

  const summarySentence = useMemo(() => {
    if (!localExplanation || localExplanation.contributions.length === 0) return null;
    const top = localExplanation.contributions[0];
    const second = localExplanation.contributions[1];
    const direction = (v: number) => (v >= 0 ? "pushed up" : "pushed down");
    let sentence = `This prediction was mainly ${direction(top.shapValue)} by ${top.feature.toLowerCase()}`;
    if (second) {
      sentence += `, and ${direction(second.shapValue)} by ${second.feature.toLowerCase()}`;
    }
    return sentence + ".";
  }, [localExplanation]);

  if (error) {
    return <p className="text-sm text-status-critical">{error}</p>;
  }

  if (!data) {
    return <p className="text-sm text-text-secondary">Loading precomputed SHAP analysis…</p>;
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-1 rounded-bento-sm border border-glass-border bg-glass-surface p-1 text-sm backdrop-blur-glass-sm w-fit">
        {(["simplified", "detailed"] as const).map((level) => (
          <button
            key={level}
            type="button"
            onClick={() => setDetailLevel(level)}
            className={`rounded-bento-sm px-3 py-1.5 capitalize transition-colors ${
              detailLevel === level ? "bg-brand text-page" : "text-text-secondary hover:text-text-primary"
            }`}
          >
            {level}
          </button>
        ))}
      </div>

      <GlassCard>
        <h2 className="mb-1 text-lg font-semibold text-text-primary">
          Global feature importance: what drives prices overall?
        </h2>
        <p className="mb-4 text-sm text-text-secondary">
          {detailLevel === "simplified"
            ? "Grouped into human-scale categories, ranked by average impact"
            : "Every individual model input, ranked by average impact"}{" "}
          across {meta.n_samples.toLocaleString("en-GB")} representative product samples.
        </p>
        <FeatureImportanceChart data={globalImportance} />
      </GlassCard>

      <GlassCard>
        <h2 className="mb-1 text-lg font-semibold text-text-primary">Local prediction explanations</h2>
        <p className="mb-4 text-sm text-text-secondary">
          Select a product instance to see how the model arrived at its forecast.
        </p>
        <label className="mb-4 block text-sm">
          <span className="mb-1 block text-text-secondary">Product sample</span>
          <select
            value={sampleIndex}
            onChange={(e) => setSampleIndex(Number(e.target.value))}
            className="w-full max-w-xs rounded-bento-sm border border-glass-border bg-glass-surface px-3 py-2 text-text-primary backdrop-blur-glass-sm"
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
              <div className="rounded-bento-sm border border-glass-border p-4">
                <div className="text-xs text-text-secondary">Model&apos;s price prediction</div>
                <div data-testid="predicted-price" className="mt-1 text-xl font-semibold text-text-primary">
                  {formatGBP(localExplanation.prediction)}
                </div>
              </div>
              <div className="rounded-bento-sm border border-glass-border p-4">
                <div className="text-xs text-text-secondary">Base value (average)</div>
                <div className="mt-1 text-xl font-semibold text-text-primary">{formatGBP(meta.base_value)}</div>
              </div>
              <div className="rounded-bento-sm border border-glass-border p-4">
                <div className="text-xs text-text-secondary">Prediction difference</div>
                <div className="mt-1 text-xl font-semibold text-text-primary">
                  {localExplanation.prediction - meta.base_value >= 0 ? "+" : ""}
                  {formatGBP(localExplanation.prediction - meta.base_value)}
                </div>
              </div>
            </div>
            {summarySentence && <p className="mb-3 text-sm text-text-secondary">{summarySentence}</p>}
            <h3 className="mb-2 text-sm font-medium text-text-secondary">
              Top contributing features (red pushes the price up, blue pushes it down)
            </h3>
            <ContributionChart data={localExplanation.contributions} />
          </>
        )}
      </GlassCard>
    </div>
  );
}
