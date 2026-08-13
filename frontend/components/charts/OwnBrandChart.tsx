"use client";

import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

export interface OwnBrandDatum {
  name: string;
  ownBrand: number;
  branded: number;
}

interface OwnBrandChartProps {
  data: OwnBrandDatum[];
}

const OWN_BRAND_COLOR = "#008300"; // categorical slot 6, unused by retailer identity elsewhere
const BRANDED_COLOR = "var(--text-muted)"; // emphasis pattern: own-brand is the story, branded is context

/** Own-brand vs branded listing counts per retailer, log-scaled -- an
 * "emphasis" chart (own-brand is the point, branded is context), not a
 * retailer-identity chart, so it deliberately does NOT reuse the
 * per-retailer color mapping used elsewhere. */
export function OwnBrandChart({ data }: OwnBrandChartProps) {
  return (
    <ResponsiveContainer width="100%" height={280}>
      <BarChart data={data} margin={{ top: 16, right: 16, left: 0, bottom: 0 }}>
        <CartesianGrid strokeDasharray="0" stroke="var(--gridline)" vertical={false} />
        <XAxis dataKey="name" tick={{ fill: "var(--text-muted)", fontSize: 12 }} axisLine={{ stroke: "var(--axis)" }} tickLine={false} />
        <YAxis
          scale="log"
          domain={["auto", "auto"]}
          tick={{ fill: "var(--text-muted)", fontSize: 12 }}
          axisLine={{ stroke: "var(--axis)" }}
          tickLine={false}
          allowDataOverflow
        />
        <Tooltip contentStyle={{ background: "var(--surface)", border: "1px solid var(--border)", borderRadius: 8 }} labelStyle={{ color: "var(--text-primary)" }} />
        <Legend wrapperStyle={{ fontSize: 12, color: "var(--text-secondary)" }} />
        <Bar dataKey="ownBrand" name="Own brand" fill={OWN_BRAND_COLOR} maxBarSize={28} radius={[4, 4, 0, 0]} />
        <Bar dataKey="branded" name="Branded" fill={BRANDED_COLOR} maxBarSize={28} radius={[4, 4, 0, 0]} />
      </BarChart>
    </ResponsiveContainer>
  );
}
