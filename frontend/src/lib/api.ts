const API_BASE = process.env.NEXT_PUBLIC_API_URL || "https://aparna-property-api.onrender.com";

async function apiFetch<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json", ...options?.headers },
    ...options,
  });
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`API ${res.status}: ${body}`);
  }
  return res.json();
}

export interface PredictionRequest {
  pid?: string;
  address?: string;
  latitude?: number;
  longitude?: number;
  property_type?: string;
  overrides?: Record<string, number | string>;
}

export interface ConfidenceInterval {
  lower: number;
  upper: number;
  level: number;
}

export interface ShapFeature {
  feature_name: string;
  feature_value: number | string | null;
  shap_value: number;
  direction: "up" | "down";
  description: string;
}

export interface ComparableDTO {
  pid: string;
  address: string;
  assessed_value: number;
  distance_m: number;
  similarity_score: number;
  year_built: number | null;
  zoning: string | null;
  neighbourhood: string | null;
}

export interface AdjustmentDTO {
  name: string;
  percentage: number;
  dollar_amount: number;
  explanation: string;
}

export interface RiskFlag {
  category: string;
  severity: "low" | "medium" | "high";
  description: string;
}

export interface MarketContext {
  neighbourhood_code: string;
  neighbourhood_name: string;
  median_assessed_value: number;
  yoy_change_pct: number | null;
  interest_rate_5yr: number | null;
  property_count: number;
  assessment_year: number;
}

export interface PredictionMetadata {
  model_segment: string;
  model_version: string;
  prediction_timestamp: string;
  data_freshness: string | null;
  mls_available: boolean;
}

export interface PredictionResponse {
  point_estimate: number;
  confidence_interval: ConfidenceInterval;
  confidence_grade: "A" | "B" | "C";
  comparables: ComparableDTO[];
  shap_features: ShapFeature[];
  adjustments: AdjustmentDTO[];
  market_context: MarketContext;
  risk_flags: RiskFlag[];
  metadata: PredictionMetadata;
  assessed_value?: number;
  market_estimate?: number;
  market_model_info?: string;
}

export interface MarketSummary {
  neighbourhood_code: string;
  neighbourhood_name: string;
  property_counts: Record<string, number>;
  median_values: Record<string, number>;
  yoy_changes: Record<string, number | null>;
  interest_rate: number | null;
}

export interface PropertyDetail {
  pid: string;
  address: string;
  neighbourhood_code: string;
  neighbourhood_name: string;
  property_type: string;
  zoning: string | null;
  year_built: number | null;
  land_value: number;
  improvement_value: number;
  total_assessed_value: number;
  estimated_living_area_sqft: number | null;
  latitude: number;
  longitude: number;
}

export interface SearchResult {
  pid: string;
  address: string;
  property_type: string;
  neighbourhood: string;
  assessed_value: number;
}

export interface BuildingUnit {
  pid: string;
  unit_number: number | null;
  property_type: string;
  assessed_value: number;
}

export interface TrendPoint {
  year: number;
  median_value: number;
  count: number;
}

export interface NeighbourhoodTrend {
  neighbourhood_code: string;
  neighbourhood_name: string;
  trends: TrendPoint[];
}

export interface Neighbourhood {
  code: string;
  name: string;
}

export interface HealthResponse {
  status: string;
  model_count: number;
  data_freshness: Record<string, string>;
  version: string;
}

// CMA Types
export interface CMARequest {
  pid?: string;
  address?: string;
  latitude?: number;
  longitude?: number;
  property_type?: string;
  bedrooms?: number;
  bathrooms?: number;
  floor_area?: number;
  year_built?: number;
  max_comps?: number;
  max_radius_m?: number;
  max_age_days?: number;
}

export interface CMAComparable {
  mls_number: string;
  address: string;
  sold_price: number;
  list_price: number | null;
  sold_date: string;
  dom: number | null;
  bedrooms: number | null;
  bathrooms: number | null;
  floor_area: number | null;
  year_built: number | null;
  property_type: string;
  distance_m: number;
  similarity_score: number;
  assessed_value: number | null;
  sar: number | null;
  list_to_sold: number | null;
  adjusted_price: number;
  adjustments: Array<{
    name: string;
    detail: string;
    percentage: number;
    dollar: number;
  }>;
}

export interface CMAResponse {
  subject: {
    address: string;
    property_type: string;
    bedrooms: number | null;
    bathrooms: number | null;
    floor_area: number | null;
    year_built: number | null;
    assessed_value: number | null;
  };
  comparables: CMAComparable[];
  comparable_count: number;
  cma_estimate: number | null;
  cma_range: {
    low: number;
    median: number;
    high: number;
    mean: number;
  } | null;
  sar_estimate: number | null;
  assessed_value: number | null;
  market_stats: {
    median_sold_price: number;
    avg_sar: number | null;
    avg_list_to_sold: number | null;
    avg_dom: number | null;
    avg_distance_m: number;
  } | null;
  recommendation: {
    estimated_value: number | null;
    estimated_range: { low: number; high: number } | null;
    confidence: string;
    method: string | null;
    note: string | null;
  };
}

export const api = {
  predict: (req: PredictionRequest) =>
    apiFetch<PredictionResponse>("/api/predict", {
      method: "POST",
      body: JSON.stringify(req),
    }),

  getProperty: (pid: string) =>
    apiFetch<PropertyDetail>(`/api/property/${encodeURIComponent(pid)}`),

  getMarketAll: () => apiFetch<MarketSummary[]>("/api/market/all"),

  getMarket: (code: string) =>
    apiFetch<MarketSummary>(`/api/market/${encodeURIComponent(code)}`),

  getNeighbourhoods: () => apiFetch<Neighbourhood[]>("/api/neighbourhoods"),

  searchProperties: (q: string, limit = 10) =>
    apiFetch<SearchResult[]>(
      `/api/search?q=${encodeURIComponent(q)}&limit=${limit}`,
    ),

  getMarketTrends: (propertyType?: string) =>
    apiFetch<NeighbourhoodTrend[]>(
      `/api/market/trends${propertyType ? `?property_type=${encodeURIComponent(propertyType)}` : ""}`,
    ),

  getHealth: () => apiFetch<HealthResponse>("/api/health"),

  getBuildingUnits: (streetNumber: number, streetName: string) =>
    apiFetch<BuildingUnit[]>(
      `/api/building-units?street_number=${streetNumber}&street_name=${encodeURIComponent(streetName)}`,
    ),

  getCMA: (req: CMARequest) =>
    apiFetch<CMAResponse>("/api/cma", {
      method: "POST",
      body: JSON.stringify(req),
    }),
};
