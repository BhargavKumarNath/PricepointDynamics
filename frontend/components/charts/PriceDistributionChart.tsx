"use client";

import { useState } from "react";
import { type ValueFormat, formatByKind } from "@/lib/format";

export interface PriceDistributionDatum {
  name: string;
  color: string;
  min: number;
  p25: number;
  median: number;
  p75: number;
  max: number;
}

interface PriceDistributionChartProps {
  data: PriceDistributionDatum[];
  format?: ValueFormat;
}

const ROW_HEIGHT = 44;
const TRACK_LEFT = 90;
const TRACK_RIGHT = 16;

/** Round up to the nearest multiple of 5 -- keeps the capped axis on a
 * clean, chart-friendly tick value rather than an arbitrary decimal. */
function roundUpToFive(value: number): number {
  return Math.ceil(value / 5) * 5;
}

/**
 * A 5-number-summary "box plot" (min/p25/median/p75/max) per retailer --
 * not a full box-and-whisker with individual outlier points, since
 * shipping row-level data to the browser is exactly what §25.3 rejects.
 * Hand-built (not Recharts, which has no native box-plot primitive):
 * a thin whisker line (min-max) with a thicker IQR box (p25-p75) and a
 * median tick, per the dataviz skill's mark specs.
 *
 * Bug fix (UI_refactor.md): the shared axis used to span every
 * retailer's raw min/max, so one retailer's outlier max price (e.g. a
 * bulk multipack) compressed every real IQR box to a few pixels. The
 * axis is now capped at the Tukey upper fence (p75 + 1.5*IQR, a
 * standard statistical convention, not an arbitrary cut) -- any
 * retailer whose true max exceeds the cap gets a chevron-capped
 * whisker plus a disclosed caption and a full-value hover tooltip, so
 * no data is hidden, only the visual scale is disclosed-and-capped.
 */
export function PriceDistributionChart({ data, format = "currency" }: PriceDistributionChartProps) {
  const [hovered, setHovered] = useState<number | null>(null);
  const valueFormatter = (value: number) => formatByKind(value, format);

  const globalMin = Math.min(...data.map((d) => d.min));
  const upperFence = Math.max(...data.map((d) => d.p75 + 1.5 * (d.p75 - d.p25)));
  const rawMax = Math.max(...data.map((d) => d.max));
  const axisCap = Math.min(roundUpToFive(upperFence), roundUpToFive(rawMax));
  const excluded = data.filter((d) => d.max > axisCap);

  const width = 640;
  const trackWidth = width - TRACK_LEFT - TRACK_RIGHT;
  const scale = (value: number) =>
    TRACK_LEFT + (Math.min(value, axisCap) - globalMin) / (axisCap - globalMin) * trackWidth;
  const height = data.length * ROW_HEIGHT;

  return (
    <div className="relative">
      <svg viewBox={`0 0 ${width} ${height}`} width="100%" role="img" aria-label="Price distribution by supermarket">
        {data.map((d, i) => {
          const y = i * ROW_HEIGHT + ROW_HEIGHT / 2;
          const isCapped = d.max > axisCap;
          const whiskerEndX = scale(d.max);
          return (
            <g
              key={d.name}
              onMouseEnter={() => setHovered(i)}
              onMouseLeave={() => setHovered((v) => (v === i ? null : v))}
              onFocus={() => setHovered(i)}
              onBlur={() => setHovered((v) => (v === i ? null : v))}
              tabIndex={0}
              style={{ cursor: "pointer" }}
            >
              <rect x={0} y={i * ROW_HEIGHT} width={width} height={ROW_HEIGHT} fill="transparent" />
              <text x={0} y={y} dy={4} fontSize={12} fill="var(--text-secondary)">
                {d.name}
              </text>
              {/* whisker: min -> max (or capped) */}
              <line x1={scale(d.min)} x2={whiskerEndX} y1={y} y2={y} stroke="var(--axis)" strokeWidth={2} strokeLinecap="round" />
              {isCapped && (
                // truncated-continues chevron, not a plain clipped end
                <path
                  d={`M ${whiskerEndX - 5} ${y - 5} L ${whiskerEndX} ${y} L ${whiskerEndX - 5} ${y + 5}`}
                  fill="none"
                  stroke="var(--axis)"
                  strokeWidth={2}
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
              )}
              {/* IQR box: p25 -> p75 */}
              <rect x={scale(d.p25)} y={y - 10} width={Math.max(scale(d.p75) - scale(d.p25), 1)} height={20} rx={4} fill={d.color} />
              {/* median tick */}
              <line x1={scale(d.median)} x2={scale(d.median)} y1={y - 12} y2={y + 12} stroke="var(--surface)" strokeWidth={2} />
            </g>
          );
        })}
      </svg>

      {hovered !== null && (
        <div
          className="pointer-events-none absolute z-10 rounded-bento-sm border border-glass-border bg-glass-surface-strong px-3 py-2 text-xs text-text-secondary backdrop-blur-glass-sm"
          style={{
            top: hovered * ROW_HEIGHT * (100 / height) + "%",
            left: "12%",
            boxShadow: "var(--shadow-glass)",
          }}
        >
          <div className="font-medium text-text-primary">{data[hovered].name}</div>
          <div>min {valueFormatter(data[hovered].min)}</div>
          <div>p25 {valueFormatter(data[hovered].p25)}</div>
          <div>median {valueFormatter(data[hovered].median)}</div>
          <div>p75 {valueFormatter(data[hovered].p75)}</div>
          <div>max {valueFormatter(data[hovered].max)}</div>
        </div>
      )}

      {excluded.length > 0 && (
        <p className="mt-2 text-xs text-text-muted">
          {`Axis capped at ${valueFormatter(axisCap)} for legibility — ${excluded.length} retailer${
            excluded.length === 1 ? "" : "s"
          } ${excluded.length === 1 ? "has" : "have"} a maximum above this. Hover a row for the exact value.`}
        </p>
      )}
    </div>
  );
}
