/**
 * Categorical color assignment -- fixed order, never cycled (dataviz
 * skill's core rule: color follows the entity, never generated per
 * render). One retailer, one slot, everywhere in the app.
 */
export const SUPERMARKET_ORDER = ["ASDA", "Aldi", "Morrisons", "Sains", "Tesco"] as const;

export type Supermarket = (typeof SUPERMARKET_ORDER)[number];

const SUPERMARKET_COLOR_VAR: Record<Supermarket, string> = {
  ASDA: "var(--series-asda)",
  Aldi: "var(--series-aldi)",
  Morrisons: "var(--series-morrisons)",
  Sains: "var(--series-sains)",
  Tesco: "var(--series-tesco)",
};

export function supermarketColor(name: string): string {
  return SUPERMARKET_COLOR_VAR[name as Supermarket] ?? "var(--text-muted)";
}
