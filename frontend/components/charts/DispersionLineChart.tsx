"use client";

import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

export interface DispersionDatum {
  date: string;
  dispersion: number;
  rollingAvg: number | null;
}

interface DispersionLineChartProps {
  data: DispersionDatum[];
}

/** Trend-over-time, single series is the point (daily dispersion), the
 * 7-day rolling average is a smoothed companion, not a second competing
 * hue -- solid blue for the signal, dashed neutral gray for the trend. */
export function DispersionLineChart({ data }: DispersionLineChartProps) {
  return (
    <ResponsiveContainer width="100%" height={320}>
      <LineChart data={data} margin={{ top: 16, right: 16, left: 0, bottom: 0 }}>
        <CartesianGrid strokeDasharray="0" stroke="var(--gridline)" vertical={false} />
        <XAxis
          dataKey="date"
          tick={{ fill: "var(--text-muted)", fontSize: 11 }}
          axisLine={{ stroke: "var(--axis)" }}
          tickLine={false}
          minTickGap={40}
        />
        <YAxis tick={{ fill: "var(--text-muted)", fontSize: 12 }} axisLine={{ stroke: "var(--axis)" }} tickLine={false} width={56} />
        <Tooltip
          contentStyle={{ background: "var(--surface)", border: "1px solid var(--border)", borderRadius: 8 }}
          labelStyle={{ color: "var(--text-primary)" }}
        />
        <Line
          type="monotone"
          dataKey="dispersion"
          name="Daily dispersion"
          stroke="var(--series-asda)"
          strokeWidth={2}
          dot={false}
        />
        <Line
          type="monotone"
          dataKey="rollingAvg"
          name="7-day rolling average"
          stroke="var(--text-muted)"
          strokeWidth={2}
          strokeDasharray="5 4"
          dot={false}
        />
      </LineChart>
    </ResponsiveContainer>
  );
}
