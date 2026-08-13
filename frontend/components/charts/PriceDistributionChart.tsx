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
  valueFormatter: (value: number) => string;
}

const ROW_HEIGHT = 44;
const TRACK_LEFT = 90;
const TRACK_RIGHT = 16;

/**
 * A 5-number-summary "box plot" (min/p25/median/p75/max) per retailer --
 * not a full box-and-whisker with individual outlier points, since
 * shipping row-level data to the browser is exactly what §25.3 rejects.
 * Hand-built (not Recharts, which has no native box-plot primitive):
 * a thin whisker line (min-max) with a thicker IQR box (p25-p75) and a
 * median tick, per the dataviz skill's mark specs.
 */
export function PriceDistributionChart({ data, valueFormatter }: PriceDistributionChartProps) {
  const globalMin = Math.min(...data.map((d) => d.min));
  const globalMax = Math.max(...data.map((d) => d.max));
  const width = 640;
  const trackWidth = width - TRACK_LEFT - TRACK_RIGHT;
  const scale = (value: number) => TRACK_LEFT + ((value - globalMin) / (globalMax - globalMin)) * trackWidth;
  const height = data.length * ROW_HEIGHT;

  return (
    <svg viewBox={`0 0 ${width} ${height}`} width="100%" role="img" aria-label="Price distribution by supermarket">
      {data.map((d, i) => {
        const y = i * ROW_HEIGHT + ROW_HEIGHT / 2;
        return (
          <g key={d.name}>
            <text x={0} y={y} dy={4} fontSize={12} fill="var(--text-secondary)">
              {d.name}
            </text>
            {/* whisker: min -> max */}
            <line x1={scale(d.min)} x2={scale(d.max)} y1={y} y2={y} stroke="var(--axis)" strokeWidth={2} strokeLinecap="round" />
            {/* IQR box: p25 -> p75 */}
            <rect x={scale(d.p25)} y={y - 10} width={Math.max(scale(d.p75) - scale(d.p25), 1)} height={20} rx={4} fill={d.color} />
            {/* median tick */}
            <line x1={scale(d.median)} x2={scale(d.median)} y1={y - 12} y2={y + 12} stroke="var(--surface)" strokeWidth={2} />
            <title>
              {d.name}: min {valueFormatter(d.min)}, p25 {valueFormatter(d.p25)}, median {valueFormatter(d.median)}, p75{" "}
              {valueFormatter(d.p75)}, max {valueFormatter(d.max)}
            </title>
          </g>
        );
      })}
    </svg>
  );
}
