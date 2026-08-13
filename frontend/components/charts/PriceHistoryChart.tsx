"use client";

import { Area, CartesianGrid, ComposedChart, Line, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { formatGBP } from "@/lib/format";

export interface PriceHistoryPoint {
  date: string;
  avgPrice: number;
  minPrice: number;
  maxPrice: number;
}

export interface PredictedPoint {
  date: string;
  price: number;
}

interface PriceHistoryChartProps {
  history: PriceHistoryPoint[];
  predicted?: PredictedPoint | null;
}

interface ChartRow {
  date: string;
  avgPrice: number | null;
  minPrice: number | null;
  bandHeight: number | null;
  predictedSegment: number | null;
  isPrediction?: boolean;
}

function PredictionDot(props: { cx?: number; cy?: number; payload?: ChartRow }) {
  const { cx, cy, payload } = props;
  if (!payload?.isPrediction || cx === undefined || cy === undefined) return null;
  const size = 5;
  return (
    <path
      d={`M ${cx} ${cy - size} L ${cx + size} ${cy} L ${cx} ${cy + size} L ${cx - size} ${cy} Z`}
      fill="var(--color-model-output)"
      stroke="var(--surface)"
      strokeWidth={1.5}
    />
  );
}

/** Historical price line (solid) with a shaded min/max band, plus an
 * optional predicted point (diamond marker in the model-output token,
 * connected to the last historical observation by a dashed guide line)
 * -- the "distinct new semantic category" exception to "brand color
 * never appears as a data mark" (see globals.css's --color-model-output). */
export function PriceHistoryChart({ history, predicted }: PriceHistoryChartProps) {
  const rows: ChartRow[] = history.map((h) => ({
    date: h.date,
    avgPrice: h.avgPrice,
    minPrice: h.minPrice,
    bandHeight: h.maxPrice - h.minPrice,
    predictedSegment: null,
  }));

  if (predicted && rows.length > 0) {
    rows[rows.length - 1] = { ...rows[rows.length - 1], predictedSegment: rows[rows.length - 1].avgPrice };
    rows.push({
      date: predicted.date,
      avgPrice: null,
      minPrice: null,
      bandHeight: null,
      predictedSegment: predicted.price,
      isPrediction: true,
    });
  }

  return (
    <ResponsiveContainer width="100%" height={280}>
      <ComposedChart data={rows} margin={{ top: 16, right: 16, left: 0, bottom: 0 }}>
        <CartesianGrid strokeDasharray="0" stroke="var(--gridline)" vertical={false} />
        <XAxis
          dataKey="date"
          tick={{ fill: "var(--text-muted)", fontSize: 11 }}
          axisLine={{ stroke: "var(--axis)" }}
          tickLine={false}
          minTickGap={32}
        />
        <YAxis
          tick={{ fill: "var(--text-muted)", fontSize: 12 }}
          axisLine={{ stroke: "var(--axis)" }}
          tickLine={false}
          width={56}
          tickFormatter={(v: number) => formatGBP(v)}
        />
        <Tooltip
          contentStyle={{ background: "var(--surface)", border: "1px solid var(--border)", borderRadius: 8 }}
          labelStyle={{ color: "var(--text-primary)" }}
          formatter={(value, name) => [typeof value === "number" ? formatGBP(value) : String(value), String(name)]}
        />
        <Area dataKey="minPrice" stackId="band" stroke="none" fill="transparent" isAnimationActive={false} legendType="none" />
        <Area
          dataKey="bandHeight"
          stackId="band"
          name="Range"
          stroke="none"
          fill="var(--series-asda)"
          fillOpacity={0.1}
          isAnimationActive={false}
        />
        <Line type="monotone" dataKey="avgPrice" name="Observed price" stroke="var(--series-asda)" strokeWidth={2} dot={false} />
        <Line
          type="monotone"
          dataKey="predictedSegment"
          name="Predicted"
          stroke="var(--color-model-output)"
          strokeWidth={2}
          strokeDasharray="5 4"
          dot={<PredictionDot />}
          activeDot={false}
        />
      </ComposedChart>
    </ResponsiveContainer>
  );
}
