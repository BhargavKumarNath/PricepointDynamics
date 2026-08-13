"use client";

import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

export interface FeatureImportanceDatum {
  feature: string;
  meanAbsShap: number;
}

interface FeatureImportanceChartProps {
  data: FeatureImportanceDatum[];
}

/** Ranked magnitude comparison -- sequential single-hue treatment (the
 * safe default per the dataviz skill), since bar order already conveys
 * the ranking and there's no identity/polarity job here. */
export function FeatureImportanceChart({ data }: FeatureImportanceChartProps) {
  const height = Math.max(280, data.length * 28);
  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart data={data} layout="vertical" margin={{ top: 8, right: 24, left: 8, bottom: 8 }}>
        <CartesianGrid strokeDasharray="0" stroke="var(--gridline)" horizontal={false} />
        <XAxis type="number" tick={{ fill: "var(--text-muted)", fontSize: 11 }} axisLine={{ stroke: "var(--axis)" }} tickLine={false} />
        <YAxis
          type="category"
          dataKey="feature"
          width={160}
          tick={{ fill: "var(--text-secondary)", fontSize: 11 }}
          axisLine={{ stroke: "var(--axis)" }}
          tickLine={false}
        />
        <Tooltip
          contentStyle={{ background: "var(--surface)", border: "1px solid var(--border)", borderRadius: 8 }}
          labelStyle={{ color: "var(--text-primary)" }}
          formatter={(value) => Number(value).toFixed(4)}
        />
        <Bar dataKey="meanAbsShap" name="mean(|SHAP value|)" fill="var(--series-asda)" maxBarSize={16} radius={[0, 4, 4, 0]} />
      </BarChart>
    </ResponsiveContainer>
  );
}
