export function formatGBP(value: number): string {
  return new Intl.NumberFormat("en-GB", { style: "currency", currency: "GBP" }).format(value);
}

export function formatCompactNumber(value: number): string {
  return new Intl.NumberFormat("en-GB", { notation: "compact", maximumFractionDigits: 1 }).format(value);
}

export function formatPercent(value: number, digits = 1): string {
  return `${value.toFixed(digits)}%`;
}

/** A named format kind, not a function prop -- functions can't cross
 * the Server -> Client Component boundary, so chart components that
 * are rendered from Server Component pages take this instead and
 * resolve formatting internally via `formatByKind`. */
export type ValueFormat = "currency" | "number";

export function formatByKind(value: number, format: ValueFormat): string {
  return format === "currency" ? formatGBP(value) : value.toLocaleString("en-GB");
}
