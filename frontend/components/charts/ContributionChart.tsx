"use client";

import { Bar, BarChart, CartesianGrid, Cell, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

export interface ContributionDatum {
  feature: string;
  shapValue: number;
}

interface ContributionChartProps {
  data: ContributionDatum[];
}

const POSITIVE_COLOR = "var(--diverging-positive)";
const NEGATIVE_COLOR = "var(--diverging-negative)";

/**
 * Diverging bar breakdown of a single prediction's top feature
 * contributions -- above/below a zero baseline, so this is a genuinely
 * diverging job (not categorical). Uses the palette's dedicated
 * diverging tokens, not the per-retailer identity colors used
 * elsewhere -- this chart has nothing to do with retailer identity, and
 * reusing e.g. "series-asda" here would misleadingly imply it does. A
 * reimplementation of the legacy dashboard's `shap.js` force plot (no
 * practical port of that library into this stack); same story -- which
 * features pushed the prediction up vs down -- as a bar chart instead
 * of an arrow diagram.
 */
export function ContributionChart({ data }: ContributionChartProps) {
  const height = Math.max(220, data.length * 32);
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
        <ReferenceLine x={0} stroke="var(--axis)" />
        <Tooltip
          contentStyle={{ background: "var(--surface)", border: "1px solid var(--border)", borderRadius: 8 }}
          labelStyle={{ color: "var(--text-primary)" }}
          formatter={(value) => Number(value).toFixed(4)}
        />
        <Bar dataKey="shapValue" name="Impact on this prediction (£)" radius={[0, 4, 4, 0]} maxBarSize={18}>
          {data.map((entry) => (
            <Cell key={entry.feature} fill={entry.shapValue >= 0 ? POSITIVE_COLOR : NEGATIVE_COLOR} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}
