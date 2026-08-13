/**
 * The live network calls this app makes (project_refactor.md §25.4):
 * `POST /v1/predict` (fired by an explicit user click, never on page
 * load) and `GET /v1/products/{id}/history` (fetched once a product +
 * store is selected, to show the historical trend alongside the
 * prediction -- UI_refactor.md's promotion of this endpoint). Every
 * other page reads a precomputed static artifact instead.
 */
const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

export interface PredictRequest {
  canonical_name: string;
  supermarket: string;
  date: string;
  price_override?: number;
}

export interface PredictResponse {
  canonical_name: string;
  supermarket: string;
  requested_date: string;
  resolved_from_date: string;
  predicted_price: number;
  price_override_applied: boolean;
  unresolved_features: string[];
}

export interface ProductHistoryPoint {
  date: string;
  supermarket: string;
  avg_price: number;
  min_price: number;
  max_price: number;
  n_listings: number;
}

export interface ProductHistoryResponse {
  canonical_name: string;
  history: ProductHistoryPoint[];
}

export interface ApiErrorBody {
  detail: string;
  context: Record<string, unknown>;
}

export class ApiError extends Error {
  status: number;
  context: Record<string, unknown>;

  constructor(status: number, body: ApiErrorBody) {
    super(body.detail);
    this.status = status;
    this.context = body.context;
  }
}

async function parseOrThrow<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const body = (await response.json().catch(() => ({ detail: response.statusText, context: {} }))) as ApiErrorBody;
    throw new ApiError(response.status, body);
  }
  return (await response.json()) as T;
}

export async function predictPrice(request: PredictRequest, signal?: AbortSignal): Promise<PredictResponse> {
  const response = await fetch(`${API_BASE_URL}/v1/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request),
    signal,
  });
  return parseOrThrow<PredictResponse>(response);
}

export async function getProductHistory(
  canonicalName: string,
  supermarket: string,
  signal?: AbortSignal,
): Promise<ProductHistoryResponse> {
  const url = `${API_BASE_URL}/v1/products/${encodeURIComponent(canonicalName)}/history?supermarket=${encodeURIComponent(
    supermarket,
  )}`;
  const response = await fetch(url, { signal });
  return parseOrThrow<ProductHistoryResponse>(response);
}
