/**
 * Human-readable labels for the model's 42 raw ML feature/column names
 * (project_refactor.md UI_refactor.md §Design System Foundation). No raw
 * snake_case name should ever reach rendered UI text after this module
 * is wired in -- consumed by `FeatureImportanceChart`, `ContributionChart`,
 * and the Predictor's `unresolved_features` display.
 */

export type FeatureGroupId =
  | "recent_price_trend"
  | "market_position"
  | "product_traits"
  | "calendar_day_of_week"
  | "calendar_day_of_month"
  | "calendar_week_of_year"
  | "retailer"
  | "unit_type"
  | "category";

export interface FeatureLabel {
  raw: string;
  label: string;
  description: string;
  group: FeatureGroupId | null;
}

function entry(raw: string, label: string, description: string, group: FeatureGroupId | null = null): FeatureLabel {
  return { raw, label, description, group };
}

export const FEATURE_LABELS: Record<string, FeatureLabel> = {
  own_brand: entry("own_brand", "Own-brand product", "Whether this is a retailer's own-label product rather than a branded one.", "product_traits"),

  price_rol_mean_7d: entry("price_rol_mean_7d", "7-day average price", "The mean price over the trailing 7 days.", "recent_price_trend"),
  price_rol_std_7d: entry("price_rol_std_7d", "7-day price volatility", "How much the price fluctuated over the trailing 7 days.", "recent_price_trend"),
  price_rol_max_7d: entry("price_rol_max_7d", "7-day highest price", "The highest price seen over the trailing 7 days.", "recent_price_trend"),
  price_rol_min_7d: entry("price_rol_min_7d", "7-day lowest price", "The lowest price seen over the trailing 7 days.", "recent_price_trend"),

  price_rol_mean_14d: entry("price_rol_mean_14d", "14-day average price", "The mean price over the trailing 14 days.", "recent_price_trend"),
  price_rol_std_14d: entry("price_rol_std_14d", "14-day price volatility", "How much the price fluctuated over the trailing 14 days.", "recent_price_trend"),
  price_rol_max_14d: entry("price_rol_max_14d", "14-day highest price", "The highest price seen over the trailing 14 days.", "recent_price_trend"),
  price_rol_min_14d: entry("price_rol_min_14d", "14-day lowest price", "The lowest price seen over the trailing 14 days.", "recent_price_trend"),

  price_rol_mean_30d: entry("price_rol_mean_30d", "30-day average price", "The mean price over the trailing 30 days.", "recent_price_trend"),
  price_rol_std_30d: entry("price_rol_std_30d", "30-day price volatility", "How much the price fluctuated over the trailing 30 days.", "recent_price_trend"),
  price_rol_max_30d: entry("price_rol_max_30d", "30-day highest price", "The highest price seen over the trailing 30 days.", "recent_price_trend"),
  price_rol_min_30d: entry("price_rol_min_30d", "30-day lowest price", "The lowest price seen over the trailing 30 days.", "recent_price_trend"),

  price_lag_1d: entry("price_lag_1d", "Yesterday's price", "The price recorded one day before the prediction date.", "recent_price_trend"),
  price_lag_7d: entry("price_lag_7d", "Price one week ago", "The price recorded seven days before the prediction date.", "recent_price_trend"),
  price_diff_1d: entry("price_diff_1d", "Day-over-day price change", "How much the price moved since the previous day.", "recent_price_trend"),

  price_vs_market_avg: entry("price_vs_market_avg", "Price vs. market average", "How this product's price compares to the average across all 5 retailers.", "market_position"),
  price_rank: entry("price_rank", "Price rank among retailers", "Where this retailer's price for the product ranks (1 = cheapest) among the 5.", "market_position"),
  is_cheapest_in_market: entry("is_cheapest_in_market", "Is the cheapest option", "Whether this retailer currently has the lowest price for the product.", "market_position"),

  day_of_week_sin: entry("day_of_week_sin", "Day-of-week pattern", "Encodes which day of the week it is, so the model can learn weekly pricing patterns.", "calendar_day_of_week"),
  day_of_week_cos: entry("day_of_week_cos", "Day-of-week pattern", "Encodes which day of the week it is, so the model can learn weekly pricing patterns.", "calendar_day_of_week"),
  day_of_month_sin: entry("day_of_month_sin", "Day-of-month pattern", "Encodes where in the month it is, so the model can learn monthly pricing patterns.", "calendar_day_of_month"),
  day_of_month_cos: entry("day_of_month_cos", "Day-of-month pattern", "Encodes where in the month it is, so the model can learn monthly pricing patterns.", "calendar_day_of_month"),
  week_of_year_sin: entry("week_of_year_sin", "Seasonality (week of year)", "Encodes the time of year, so the model can learn seasonal pricing patterns.", "calendar_week_of_year"),
  week_of_year_cos: entry("week_of_year_cos", "Seasonality (week of year)", "Encodes the time of year, so the model can learn seasonal pricing patterns.", "calendar_week_of_year"),

  supermarket_Aldi: entry("supermarket_Aldi", "Retailer: Aldi", "Flag indicating the retailer is Aldi.", "retailer"),
  supermarket_Morrisons: entry("supermarket_Morrisons", "Retailer: Morrisons", "Flag indicating the retailer is Morrisons.", "retailer"),
  supermarket_Sains: entry("supermarket_Sains", "Retailer: Sainsbury's", "Flag indicating the retailer is Sainsbury's.", "retailer"),
  supermarket_Tesco: entry("supermarket_Tesco", "Retailer: Tesco", "Flag indicating the retailer is Tesco.", "retailer"),

  unit_l: entry("unit_l", "Sold by volume (litres)", "Flag indicating the product is sold by liquid volume.", "unit_type"),
  unit_m: entry("unit_m", "Sold by weight (kg)", "Flag indicating the product is sold by weight.", "unit_type"),
  unit_unit: entry("unit_unit", "Sold as a single item", "Flag indicating the product is sold per item rather than by weight or volume.", "unit_type"),

  category_bakery: entry("category_bakery", "Category: Bakery", "Flag indicating the product is in the Bakery category.", "category"),
  category_drinks: entry("category_drinks", "Category: Drinks", "Flag indicating the product is in the Drinks category.", "category"),
  category_food_cupboard: entry("category_food_cupboard", "Category: Food cupboard", "Flag indicating the product is in the Food Cupboard category.", "category"),
  "category_free-from": entry("category_free-from", "Category: Free-from", "Flag indicating the product is in the Free-From category.", "category"),
  category_fresh_food: entry("category_fresh_food", "Category: Fresh food", "Flag indicating the product is in the Fresh Food category.", "category"),
  category_frozen: entry("category_frozen", "Category: Frozen", "Flag indicating the product is in the Frozen category.", "category"),
  category_health_products: entry("category_health_products", "Category: Health products", "Flag indicating the product is in the Health Products category.", "category"),
  category_home: entry("category_home", "Category: Home", "Flag indicating the product is in the Home category.", "category"),
  category_household: entry("category_household", "Category: Household", "Flag indicating the product is in the Household category.", "category"),
  category_pets: entry("category_pets", "Category: Pets", "Flag indicating the product is in the Pets category.", "category"),
};

/** Human label for a raw feature name, falling back to a title-cased
 * version of the raw name itself if it's ever missing from the
 * dictionary above (should not happen for the 42 known columns, but a
 * safety net beats a raw snake_case string leaking through). */
export function humanizeFeature(raw: string): string {
  const known = FEATURE_LABELS[raw];
  if (known) return known.label;
  return raw
    .replace(/_/g, " ")
    .replace(/\b\w/g, (c) => c.toUpperCase());
}

export function featureDescription(raw: string): string | null {
  return FEATURE_LABELS[raw]?.description ?? null;
}

export interface FeatureGroupDef {
  id: FeatureGroupId;
  label: string;
  description: string;
  members: string[];
}

// Grouping buckets the 6 cyclical calendar columns and the one-hot
// retailer/unit/category dummies into human-scale concepts, for the
// Model Insights page's default "Simplified" view.
export const FEATURE_GROUPS: FeatureGroupDef[] = [
  {
    id: "calendar_day_of_week",
    label: "Day-of-week pattern",
    description: "Combined effect of which day of the week it is.",
    members: ["day_of_week_sin", "day_of_week_cos"],
  },
  {
    id: "calendar_day_of_month",
    label: "Day-of-month pattern",
    description: "Combined effect of where in the month it is.",
    members: ["day_of_month_sin", "day_of_month_cos"],
  },
  {
    id: "calendar_week_of_year",
    label: "Seasonality (week of year)",
    description: "Combined effect of the time of year.",
    members: ["week_of_year_sin", "week_of_year_cos"],
  },
  {
    id: "retailer",
    label: "Which supermarket",
    description:
      "Combined effect of which retailer this is (ASDA has no separate flag -- it's the baseline every other retailer is compared against).",
    members: ["supermarket_Aldi", "supermarket_Morrisons", "supermarket_Sains", "supermarket_Tesco"],
  },
  {
    id: "unit_type",
    label: "Product unit type",
    description: "Combined effect of whether the product is sold by volume, weight, or item.",
    members: ["unit_l", "unit_m", "unit_unit"],
  },
  {
    id: "category",
    label: "Product category",
    description: "Combined effect of the product's category.",
    members: [
      "category_bakery",
      "category_drinks",
      "category_food_cupboard",
      "category_free-from",
      "category_fresh_food",
      "category_frozen",
      "category_health_products",
      "category_home",
      "category_household",
      "category_pets",
    ],
  },
];

const GROUP_OF_FEATURE = new Map<string, FeatureGroupDef>(
  FEATURE_GROUPS.flatMap((g) => g.members.map((m) => [m, g] as const)),
);

export interface GroupedContribution {
  id: string;
  label: string;
  shapValue: number;
  isGroup: boolean;
  /** The active one-hot member, if this is a resolved single-value
   * group (e.g. "Category: Fresh food" instead of just "Category"). */
  activeMember?: string;
}

/**
 * Collapses raw per-feature SHAP contributions into human-scale groups
 * for the "Simplified" view. Sums *signed* SHAP values within a group
 * -- never `Math.abs()` before summing -- because SHAP values are
 * additive by construction (sum(shap) + base_value === prediction);
 * summing absolute values would overstate group magnitude and break
 * that identity.
 */
export function groupContributions(
  raw: { feature: string; shapValue: number; featureValue?: number }[],
): GroupedContribution[] {
  const buckets = new Map<string, { def: FeatureGroupDef; shapSum: number; active: string | null }>();
  const out: GroupedContribution[] = [];

  for (const row of raw) {
    const group = GROUP_OF_FEATURE.get(row.feature);
    if (!group) {
      out.push({ id: row.feature, label: humanizeFeature(row.feature), shapValue: row.shapValue, isGroup: false });
      continue;
    }
    const bucket = buckets.get(group.id) ?? { def: group, shapSum: 0, active: null };
    bucket.shapSum += row.shapValue;
    if (row.featureValue !== undefined && row.featureValue > 0) bucket.active = row.feature;
    buckets.set(group.id, bucket);
  }

  for (const [id, bucket] of buckets) {
    const label =
      bucket.active && FEATURE_LABELS[bucket.active] ? FEATURE_LABELS[bucket.active].label : bucket.def.label;
    out.push({
      id,
      label,
      shapValue: bucket.shapSum,
      isGroup: true,
      activeMember: bucket.active ?? undefined,
    });
  }

  return out.sort((a, b) => Math.abs(b.shapValue) - Math.abs(a.shapValue));
}
