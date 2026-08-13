"use client";

import { Bar, BarChart, CartesianGrid, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { type ValueFormat, formatByKind } from "@/lib/format";

export interface SimpleBarDatum {
  name: string;
  value: number;
  color: string;
}

interface SimpleBarChartProps {
  data: SimpleBarDatum[];
  format?: ValueFormat;
  yAxisWidth?: number;
}

/** Single-series categorical bar chart -- one color per category (identity),
 * mark spec per the dataviz skill: capped bar thickness, rounded data-end,
 * hairline recessive gridlines, value at the tip. */
export function SimpleBarChart({ data, format = "number", yAxisWidth = 48 }: SimpleBarChartProps) {
  const fmt = (value: number) => formatByKind(value, format);
  return (
    <ResponsiveContainer width="100%" height={280}>
      <BarChart data={data} margin={{ top: 16, right: 16, left: 0, bottom: 0 }}>
        <CartesianGrid strokeDasharray="0" stroke="var(--gridline)" vertical={false} />
        <XAxis dataKey="name" tick={{ fill: "var(--text-muted)", fontSize: 12 }} axisLine={{ stroke: "var(--axis)" }} tickLine={false} />
        <YAxis
          width={yAxisWidth}
          tick={{ fill: "var(--text-muted)", fontSize: 12 }}
          axisLine={{ stroke: "var(--axis)" }}
          tickLine={false}
          tickFormatter={fmt}
        />
        <Tooltip
          contentStyle={{ background: "var(--surface)", border: "1px solid var(--border)", borderRadius: 8 }}
          labelStyle={{ color: "var(--text-primary)" }}
          formatter={(value) => fmt(Number(value))}
        />
        <Bar
          dataKey="value"
          maxBarSize={40}
          radius={[4, 4, 0, 0]}
          label={{ position: "top", fill: "var(--text-secondary)", fontSize: 11, formatter: (v) => fmt(Number(v)) }}
        >
          {data.map((entry) => (
            <Cell key={entry.name} fill={entry.color} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}
