/**
 * Client-side Parquet loading for the two larger artifacts (project_refactor.md
 * §25.3) -- predictor_context.parquet (product/store selector) and the
 * SHAP explorer's feature/value grids. Read directly in the browser via
 * hyparquet (pure JS, no WASM) rather than round-tripping through a
 * server, since these are static files served from a CDN/R2 either way.
 *
 * `NEXT_PUBLIC_PARQUET_BASE_URL` defaults to the local `/data-r2` stand-in
 * (see `WebArtifactsConfig`'s docstring); a deployed build points it at
 * the real Cloudflare R2 bucket URL instead.
 */
import { asyncBufferFromUrl, parquetReadObjects } from "hyparquet";

const PARQUET_BASE_URL = process.env.NEXT_PUBLIC_PARQUET_BASE_URL ?? "/data-r2";

export interface PredictorContextRow {
  canonical_name: string;
  supermarket: string;
  category: string;
  date: string;
  prices: number;
  price_lag_1d: number | null;
}

let predictorContextCache: Promise<PredictorContextRow[]> | null = null;

// hyparquet parses the source TIMESTAMP column (feature_engineered_data.parquet's
// `date`, carried through unchanged by web_artifacts.py::export_predictor_context)
// into a native JS Date, not a string -- normalize to YYYY-MM-DD immediately so
// every downstream consumer (JSX rendering, the /v1/predict request body, which
// expects a plain ISO date) gets the string this module's own types promise.
function toIsoDate(value: unknown): string {
  return value instanceof Date ? value.toISOString().slice(0, 10) : String(value);
}

export function loadPredictorContext(): Promise<PredictorContextRow[]> {
  predictorContextCache ??= (async () => {
    const file = await asyncBufferFromUrl({ url: `${PARQUET_BASE_URL}/predictor_context.parquet` });
    const rows = (await parquetReadObjects({ file })) as unknown as PredictorContextRow[];
    return rows.map((row) => ({ ...row, date: toIsoDate(row.date) }));
  })();
  return predictorContextCache;
}

export interface ShapExplorerData {
  features: Record<string, number>[];
  values: Record<string, number>[];
}

let shapExplorerCache: Promise<ShapExplorerData> | null = null;

export function loadShapExplorer(): Promise<ShapExplorerData> {
  shapExplorerCache ??= (async () => {
    const [featuresFile, valuesFile] = await Promise.all([
      asyncBufferFromUrl({ url: `${PARQUET_BASE_URL}/shap_explorer_features.parquet` }),
      asyncBufferFromUrl({ url: `${PARQUET_BASE_URL}/shap_explorer_values.parquet` }),
    ]);
    const [features, values] = await Promise.all([
      parquetReadObjects({ file: featuresFile }),
      parquetReadObjects({ file: valuesFile }),
    ]);
    return {
      features: features as unknown as Record<string, number>[],
      values: values as unknown as Record<string, number>[],
    };
  })();
  return shapExplorerCache;
}
