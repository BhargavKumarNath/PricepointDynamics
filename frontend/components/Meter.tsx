interface MeterProps {
  label: string;
  /** 0-100. */
  value: number;
  color?: string;
  valueLabel: string;
}

/** Progress-bar style % display (own-brand %, basket coverage %,
 * competitiveness index). Track is a lighter step of the fill's own
 * ramp, per the dataviz skill's meter spec, so state reads across the
 * whole bar rather than only at the filled tip. */
export function Meter({ label, value, color = "var(--brand)", valueLabel }: MeterProps) {
  const clamped = Math.max(0, Math.min(100, value));
  return (
    <div>
      <div className="mb-1 flex items-baseline justify-between text-xs">
        <span className="text-text-secondary">{label}</span>
        <span className="font-medium text-text-primary">{valueLabel}</span>
      </div>
      <div className="h-2 overflow-hidden rounded-full" style={{ background: `color-mix(in oklab, ${color} 18%, var(--surface))` }}>
        <div className="h-full rounded-full" style={{ width: `${clamped}%`, background: color }} />
      </div>
    </div>
  );
}
