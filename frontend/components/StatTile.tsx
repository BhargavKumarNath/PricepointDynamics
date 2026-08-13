interface StatTileProps {
  label: string;
  value: string;
  help?: string;
}

/** Stat tile contract (dataviz skill): label (sentence case, no trailing
 * colon), value (semibold, auto-compact), optional supporting text. */
export function StatTile({ label, value, help }: StatTileProps) {
  return (
    <div className="rounded-lg border border-border bg-surface p-5">
      <div className="text-sm text-text-secondary">{label}</div>
      <div className="mt-1 text-3xl font-semibold text-text-primary">{value}</div>
      {help && <div className="mt-1 text-xs text-text-muted">{help}</div>}
    </div>
  );
}
