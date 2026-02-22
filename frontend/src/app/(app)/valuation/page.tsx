"use client";

import { useState } from "react";
import { api, type PredictionRequest, type PredictionResponse } from "@/lib/api";
import {
  formatCurrencyFull,
  formatCurrency,
  formatPercent,
  formatDistance,
  gradeColor,
  gradeBg,
  severityColor,
} from "@/lib/format";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Cell,
  ReferenceLine,
} from "recharts";

export default function ValuationPage() {
  const [mode, setMode] = useState<"pid" | "address" | "coordinates">("pid");
  const [pid, setPid] = useState("");
  const [address, setAddress] = useState("");
  const [lat, setLat] = useState("");
  const [lon, setLon] = useState("");
  const [propertyType, setPropertyType] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);

    const req: PredictionRequest = {};
    if (mode === "pid" && pid) req.pid = pid;
    if (mode === "address" && address) req.address = address;
    if (mode === "coordinates" && lat && lon) {
      req.latitude = parseFloat(lat);
      req.longitude = parseFloat(lon);
    }
    if (propertyType) req.property_type = propertyType;

    try {
      const res = await api.predict(req);
      setResult(res);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Prediction failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="space-y-8">
      {/* Header */}
      <div>
        <h1
          className="text-3xl text-sand-900 tracking-tight"
          style={{ fontFamily: "var(--font-display)" }}
        >
          Property Valuation
        </h1>
        <p className="text-sand-500 text-sm mt-1">
          Generate an ML-powered property valuation with confidence intervals and comparables
        </p>
      </div>

      {/* Search Form */}
      <form onSubmit={handleSubmit} className="card-hairline p-6 space-y-5">
        {/* Mode tabs */}
        <div className="flex gap-1 p-1 bg-sand-100 rounded-lg w-fit">
          {(["pid", "address", "coordinates"] as const).map((m) => (
            <button
              key={m}
              type="button"
              onClick={() => setMode(m)}
              className={`px-4 py-1.5 text-xs font-medium rounded-md transition ${
                mode === m
                  ? "bg-white text-sand-900 shadow-sm"
                  : "text-sand-500 hover:text-sand-700"
              }`}
            >
              {m === "pid" ? "PID" : m === "address" ? "Address" : "Coordinates"}
            </button>
          ))}
        </div>

        <div className="flex gap-4 items-end">
          {/* Dynamic input */}
          <div className="flex-1">
            {mode === "pid" && (
              <div>
                <label className="block text-xs font-medium text-sand-500 mb-1.5">
                  BC Assessment PID
                </label>
                <input
                  type="text"
                  value={pid}
                  onChange={(e) => setPid(e.target.value)}
                  placeholder="e.g. 012-345-678"
                  className="w-full px-4 py-2.5 rounded-lg border border-sand-200 bg-white text-sand-900 text-sm placeholder:text-sand-300 focus:outline-none focus:border-teal-400 focus:ring-1 focus:ring-teal-200 transition"
                />
              </div>
            )}
            {mode === "address" && (
              <div>
                <label className="block text-xs font-medium text-sand-500 mb-1.5">
                  Street Address
                </label>
                <input
                  type="text"
                  value={address}
                  onChange={(e) => setAddress(e.target.value)}
                  placeholder="e.g. 1234 Main St, Vancouver, BC"
                  className="w-full px-4 py-2.5 rounded-lg border border-sand-200 bg-white text-sand-900 text-sm placeholder:text-sand-300 focus:outline-none focus:border-teal-400 focus:ring-1 focus:ring-teal-200 transition"
                />
              </div>
            )}
            {mode === "coordinates" && (
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="block text-xs font-medium text-sand-500 mb-1.5">
                    Latitude
                  </label>
                  <input
                    type="text"
                    value={lat}
                    onChange={(e) => setLat(e.target.value)}
                    placeholder="49.2827"
                    className="w-full px-4 py-2.5 rounded-lg border border-sand-200 bg-white text-sand-900 text-sm placeholder:text-sand-300 focus:outline-none focus:border-teal-400 focus:ring-1 focus:ring-teal-200 transition"
                  />
                </div>
                <div>
                  <label className="block text-xs font-medium text-sand-500 mb-1.5">
                    Longitude
                  </label>
                  <input
                    type="text"
                    value={lon}
                    onChange={(e) => setLon(e.target.value)}
                    placeholder="-123.1207"
                    className="w-full px-4 py-2.5 rounded-lg border border-sand-200 bg-white text-sand-900 text-sm placeholder:text-sand-300 focus:outline-none focus:border-teal-400 focus:ring-1 focus:ring-teal-200 transition"
                  />
                </div>
              </div>
            )}
          </div>

          {/* Property type */}
          <div className="w-[200px] shrink-0">
            <label className="block text-xs font-medium text-sand-500 mb-1.5">
              Property Type
            </label>
            <select
              value={propertyType}
              onChange={(e) => setPropertyType(e.target.value)}
              className="w-full px-4 py-2.5 rounded-lg border border-sand-200 bg-white text-sand-900 text-sm focus:outline-none focus:border-teal-400 focus:ring-1 focus:ring-teal-200 transition"
            >
              <option value="">Auto-detect</option>
              <option value="condo">Condo</option>
              <option value="townhome">Townhome</option>
              <option value="detached">Detached</option>
              <option value="development_land">Development Land</option>
            </select>
          </div>

          {/* Submit */}
          <button
            type="submit"
            disabled={loading}
            className="shrink-0 px-6 py-2.5 text-sm font-semibold text-white bg-gradient-to-r from-teal-600 to-teal-700 rounded-lg hover:from-teal-700 hover:to-teal-800 transition shadow-sm disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {loading ? (
              <span className="flex items-center gap-2">
                <svg className="animate-spin w-4 h-4" viewBox="0 0 24 24" fill="none">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                </svg>
                Valuating...
              </span>
            ) : (
              "Get Valuation"
            )}
          </button>
        </div>
      </form>

      {/* Error */}
      {error && (
        <div className="card-hairline p-4 border-rose-200 bg-rose-50">
          <p className="text-sm text-rose-600">{error}</p>
        </div>
      )}

      {/* Results */}
      {result && <ValuationResult result={result} />}
    </div>
  );
}

function ValuationResult({ result }: { result: PredictionResponse }) {
  return (
    <div className="space-y-6 animate-fade-in-up" style={{ animationFillMode: "forwards" }}>
      {/* Hero Estimate */}
      <div className="card-hairline p-8">
        <div className="flex items-start justify-between">
          <div>
            <div className="text-xs font-medium text-sand-400 uppercase tracking-wider mb-2">
              Estimated Value
            </div>
            <div className="hero-number">{formatCurrencyFull(result.point_estimate)}</div>
            <div className="mt-3 flex items-center gap-4">
              <span className="text-sm text-sand-500">
                {formatCurrencyFull(result.confidence_interval.lower)} &ndash;{" "}
                {formatCurrencyFull(result.confidence_interval.upper)}
              </span>
              <span className="text-xs text-sand-400">
                {(result.confidence_interval.level * 100).toFixed(0)}% confidence interval
              </span>
            </div>
          </div>
          <div className="text-right">
            <div
              className={`inline-flex items-center gap-2 px-4 py-2 rounded-xl border ${gradeBg(
                result.confidence_grade,
              )}`}
            >
              <span className={`text-2xl font-semibold ${gradeColor(result.confidence_grade)}`} style={{ fontFamily: "var(--font-display)" }}>
                {result.confidence_grade}
              </span>
              <span className="text-xs text-sand-500">
                Confidence<br />Grade
              </span>
            </div>
          </div>
        </div>

        {/* Confidence bar */}
        <div className="mt-6 relative">
          <div className="h-2 bg-sand-100 rounded-full overflow-hidden">
            <div
              className="h-full rounded-full bg-gradient-to-r from-teal-400 to-teal-600 transition-all duration-1000"
              style={{
                width: `${Math.min(
                  100,
                  ((result.point_estimate - result.confidence_interval.lower) /
                    (result.confidence_interval.upper - result.confidence_interval.lower)) *
                    100,
                )}%`,
              }}
            />
          </div>
          <div className="flex justify-between mt-1.5 text-[10px] text-sand-400">
            <span>{formatCurrency(result.confidence_interval.lower)}</span>
            <span>{formatCurrency(result.confidence_interval.upper)}</span>
          </div>
        </div>
      </div>

      {/* Grid: SHAP + Adjustments + Market Context */}
      <div className="grid lg:grid-cols-2 gap-6">
        {/* SHAP Features */}
        <div className="card-hairline p-6">
          <h3 className="text-sm font-semibold text-sand-800 mb-4">Value Drivers</h3>
          {result.shap_features.length > 0 ? (
            <div className="h-[280px]">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart
                  layout="vertical"
                  data={result.shap_features
                    .sort((a, b) => Math.abs(b.shap_value) - Math.abs(a.shap_value))
                    .slice(0, 8)
                    .map((f) => ({
                      name: f.feature_name.replace(/_/g, " "),
                      value: f.shap_value,
                    }))}
                  margin={{ top: 0, right: 20, left: 0, bottom: 0 }}
                >
                  <XAxis type="number" tick={{ fontSize: 10, fill: "#9a9080" }} />
                  <YAxis
                    type="category"
                    dataKey="name"
                    width={120}
                    tick={{ fontSize: 11, fill: "#5f574c" }}
                  />
                  <Tooltip
                    contentStyle={{
                      background: "white",
                      border: "1px solid #e8e4dd",
                      borderRadius: 8,
                      fontSize: 12,
                    }}
                  />
                  <ReferenceLine x={0} stroke="#d5cfc5" />
                  <Bar dataKey="value" radius={[0, 4, 4, 0]} barSize={16}>
                    {result.shap_features
                      .sort((a, b) => Math.abs(b.shap_value) - Math.abs(a.shap_value))
                      .slice(0, 8)
                      .map((f, i) => (
                        <Cell
                          key={i}
                          fill={f.shap_value >= 0 ? "#06c2ae" : "#f43f5e"}
                        />
                      ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          ) : (
            <p className="text-sm text-sand-400">No SHAP features available</p>
          )}
        </div>

        {/* Adjustments */}
        <div className="card-hairline p-6">
          <h3 className="text-sm font-semibold text-sand-800 mb-4">Adjustments Applied</h3>
          {result.adjustments.length > 0 ? (
            <div className="space-y-3">
              {result.adjustments.map((adj, i) => (
                <div
                  key={i}
                  className="flex items-center justify-between py-2 border-b border-sand-100 last:border-0"
                >
                  <div className="flex-1">
                    <div className="text-sm text-sand-800">{adj.name}</div>
                    <div className="text-xs text-sand-400 mt-0.5">{adj.explanation}</div>
                  </div>
                  <div className="text-right ml-4">
                    <div
                      className={`text-sm font-medium ${
                        adj.percentage >= 0 ? "text-emerald-600" : "text-rose-500"
                      }`}
                    >
                      {adj.percentage >= 0 ? "+" : ""}
                      {adj.percentage.toFixed(1)}%
                    </div>
                    <div className="text-[11px] text-sand-400">
                      {adj.dollar_amount >= 0 ? "+" : ""}
                      {formatCurrency(adj.dollar_amount)}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <p className="text-sm text-sand-400">No adjustments applied</p>
          )}
        </div>
      </div>

      {/* Market Context */}
      <div className="card-hairline p-6">
        <h3 className="text-sm font-semibold text-sand-800 mb-4">Market Context</h3>
        <div className="grid grid-cols-2 md:grid-cols-5 gap-6">
          {[
            { label: "Neighbourhood", value: result.market_context.neighbourhood_name },
            { label: "Median Value", value: formatCurrency(result.market_context.median_assessed_value) },
            { label: "YoY Change", value: formatPercent(result.market_context.yoy_change_pct) },
            { label: "Properties", value: result.market_context.property_count.toLocaleString() },
            { label: "Assessment Year", value: result.market_context.assessment_year.toString() },
          ].map((item) => (
            <div key={item.label}>
              <div className="text-[11px] text-sand-400 uppercase tracking-wider">{item.label}</div>
              <div className="text-sm font-medium text-sand-800 mt-1">{item.value}</div>
            </div>
          ))}
        </div>
      </div>

      {/* Comparables */}
      <div className="card-hairline p-6">
        <h3 className="text-sm font-semibold text-sand-800 mb-4">
          Comparable Properties ({result.comparables.length})
        </h3>
        {result.comparables.length > 0 ? (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-sand-200">
                  {["Address", "Assessed Value", "Distance", "Similarity", "Year Built", "Zoning"].map(
                    (h) => (
                      <th
                        key={h}
                        className="text-left py-2 px-3 text-[11px] font-medium text-sand-400 uppercase tracking-wider"
                      >
                        {h}
                      </th>
                    ),
                  )}
                </tr>
              </thead>
              <tbody>
                {result.comparables.map((comp, i) => (
                  <tr
                    key={i}
                    className="border-b border-sand-100 last:border-0 hover:bg-sand-50 transition"
                  >
                    <td className="py-2.5 px-3 text-sand-800">{comp.address}</td>
                    <td className="py-2.5 px-3 text-sand-700">
                      {formatCurrencyFull(comp.assessed_value)}
                    </td>
                    <td className="py-2.5 px-3 text-sand-500">
                      {formatDistance(comp.distance_m)}
                    </td>
                    <td className="py-2.5 px-3">
                      <div className="flex items-center gap-2">
                        <div className="w-16 h-1.5 bg-sand-100 rounded-full overflow-hidden">
                          <div
                            className="h-full bg-teal-500 rounded-full"
                            style={{ width: `${Math.max(5, (1 - comp.similarity_score) * 100)}%` }}
                          />
                        </div>
                        <span className="text-[11px] text-sand-400">
                          {(comp.similarity_score * 100).toFixed(0)}
                        </span>
                      </div>
                    </td>
                    <td className="py-2.5 px-3 text-sand-500">{comp.year_built || "—"}</td>
                    <td className="py-2.5 px-3 text-sand-500">{comp.zoning || "—"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <p className="text-sm text-sand-400">No comparable properties found</p>
        )}
      </div>

      {/* Risk Flags */}
      {result.risk_flags.length > 0 && (
        <div className="card-hairline p-6">
          <h3 className="text-sm font-semibold text-sand-800 mb-4">Risk Flags</h3>
          <div className="space-y-3">
            {result.risk_flags.map((flag, i) => (
              <div
                key={i}
                className={`flex items-start gap-3 p-3 rounded-lg border ${severityColor(
                  flag.severity,
                )}`}
              >
                <svg className="w-4 h-4 mt-0.5 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2">
                  <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L3.27 16.5c-.77.833.192 2.5 1.732 2.5z" />
                </svg>
                <div>
                  <div className="text-sm font-medium capitalize">{flag.category.replace(/_/g, " ")}</div>
                  <div className="text-xs mt-0.5 opacity-80">{flag.description}</div>
                </div>
                <span className="ml-auto text-[10px] uppercase font-semibold tracking-wider">
                  {flag.severity}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Metadata footer */}
      <div className="flex items-center justify-between text-[11px] text-sand-400 px-1">
        <span>
          Model: {result.metadata.model_segment} &middot; v{result.metadata.model_version}
        </span>
        <span>
          {new Date(result.metadata.prediction_timestamp).toLocaleString()}
        </span>
      </div>
    </div>
  );
}
