"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import { api, type PredictionRequest, type PredictionResponse, type SearchResult } from "@/lib/api";
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

const GOOGLE_MAPS_KEY = process.env.NEXT_PUBLIC_GOOGLE_MAPS_KEY;

function useDebounce<T>(value: T, delay: number): T {
  const [debounced, setDebounced] = useState(value);
  useEffect(() => {
    const timer = setTimeout(() => setDebounced(value), delay);
    return () => clearTimeout(timer);
  }, [value, delay]);
  return debounced;
}

// Load Google Maps script once
let googleMapsLoaded = false;
function loadGoogleMaps(): Promise<void> {
  if (googleMapsLoaded || typeof window === "undefined") return Promise.resolve();
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  if ((window as any).google?.maps?.places) {
    googleMapsLoaded = true;
    return Promise.resolve();
  }
  return new Promise((resolve, reject) => {
    const existing = document.querySelector('script[src*="maps.googleapis.com"]');
    if (existing) {
      existing.addEventListener("load", () => { googleMapsLoaded = true; resolve(); });
      return;
    }
    const script = document.createElement("script");
    script.src = `https://maps.googleapis.com/maps/api/js?key=${GOOGLE_MAPS_KEY}&libraries=places`;
    script.async = true;
    script.onload = () => { googleMapsLoaded = true; resolve(); };
    script.onerror = reject;
    document.head.appendChild(script);
  });
}

export default function ValuationPage() {
  const [mode, setMode] = useState<"pid" | "address" | "coordinates">("address");
  const [pid, setPid] = useState("");
  const [address, setAddress] = useState("");
  const [lat, setLat] = useState("");
  const [lon, setLon] = useState("");
  const [propertyType, setPropertyType] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  // PID autocomplete state (internal DB search)
  const [pidQuery, setPidQuery] = useState("");
  const [pidSuggestions, setPidSuggestions] = useState<SearchResult[]>([]);
  const [showPidDropdown, setShowPidDropdown] = useState(false);
  const [pidHighlight, setPidHighlight] = useState(-1);
  const pidDropdownRef = useRef<HTMLDivElement>(null);

  // Google Places autocomplete state
  const addressInputRef = useRef<HTMLInputElement>(null);
  const autocompleteRef = useRef<google.maps.places.Autocomplete | null>(null);
  const [selectedCoords, setSelectedCoords] = useState<{ lat: number; lng: number } | null>(null);
  const [selectedAddress, setSelectedAddress] = useState("");

  // Track pending auto-submit
  // pendingSubmitRef removed — no auto-submit on address selection

  const debouncedPidQuery = useDebounce(pidQuery, 250);

  // Core prediction logic (separated from form handler so we can call it programmatically)
  const runPrediction = useCallback(async (overrideReq?: Partial<PredictionRequest>) => {
    setLoading(true);
    setError(null);
    setResult(null);

    const req: PredictionRequest = {};
    if (overrideReq) {
      Object.assign(req, overrideReq);
    } else if (mode === "pid" && pid) {
      req.pid = pid;
    } else if (mode === "address") {
      if (selectedCoords) {
        req.latitude = selectedCoords.lat;
        req.longitude = selectedCoords.lng;
      }
      if (address) req.address = address;
    } else if (mode === "coordinates" && lat && lon) {
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
  }, [mode, pid, address, selectedCoords, lat, lon, propertyType]);

  // PID mode: fetch from internal DB
  useEffect(() => {
    if (mode !== "pid" || debouncedPidQuery.length < 2) {
      setPidSuggestions([]);
      return;
    }
    let cancelled = false;
    api.searchProperties(debouncedPidQuery, 8).then((results) => {
      if (!cancelled) {
        setPidSuggestions(results);
        setShowPidDropdown(results.length > 0);
        setPidHighlight(-1);
      }
    }).catch(() => {
      if (!cancelled) setPidSuggestions([]);
    });
    return () => { cancelled = true; };
  }, [debouncedPidQuery, mode]);

  // Address mode: init Google Places autocomplete
  useEffect(() => {
    if (mode !== "address" || !GOOGLE_MAPS_KEY) return;

    let mounted = true;
    loadGoogleMaps().then(() => {
      if (!mounted || !addressInputRef.current) return;
      if (autocompleteRef.current) return; // Already initialized

      const ac = new google.maps.places.Autocomplete(addressInputRef.current, {
        componentRestrictions: { country: "ca" },
        fields: ["formatted_address", "geometry", "address_components"],
        types: ["address"],
      });

      // Bias to Vancouver area
      const vancouverBounds = new google.maps.LatLngBounds(
        new google.maps.LatLng(49.19, -123.27),
        new google.maps.LatLng(49.32, -123.02),
      );
      ac.setBounds(vancouverBounds);

      ac.addListener("place_changed", () => {
        const place = ac.getPlace();
        if (!place.geometry?.location) return;

        const formattedAddr = place.formatted_address || "";
        const coords = {
          lat: place.geometry.location.lat(),
          lng: place.geometry.location.lng(),
        };

        setAddress(formattedAddr);
        setSelectedAddress(formattedAddr);
        setSelectedCoords(coords);
        setLat(coords.lat.toString());
        setLon(coords.lng.toString());

        // Address selected — user must click "Get Valuation" to proceed
      });

      autocompleteRef.current = ac;
    });

    return () => {
      mounted = false;
    };
  }, [mode]);

  // Address selection populates fields but does NOT auto-submit.
  // User must click "Get Valuation" to trigger the prediction.

  // Clean up autocomplete when leaving address mode
  useEffect(() => {
    if (mode !== "address") {
      autocompleteRef.current = null;
    }
  }, [mode]);

  // Close PID dropdown on outside click
  useEffect(() => {
    function handleClickOutside(e: MouseEvent) {
      if (pidDropdownRef.current && !pidDropdownRef.current.contains(e.target as Node)) {
        setShowPidDropdown(false);
      }
    }
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  const selectPidSuggestion = useCallback((s: SearchResult) => {
    setPid(s.pid);
    setPidQuery(s.address + " (PID: " + s.pid + ")");
    if (s.property_type) setPropertyType(s.property_type);
    setShowPidDropdown(false);
  }, []);

  function handlePidKeyDown(e: React.KeyboardEvent) {
    if (!showPidDropdown || pidSuggestions.length === 0) return;
    if (e.key === "ArrowDown") {
      e.preventDefault();
      setPidHighlight((i) => Math.min(i + 1, pidSuggestions.length - 1));
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      setPidHighlight((i) => Math.max(i - 1, 0));
    } else if (e.key === "Enter" && pidHighlight >= 0) {
      e.preventDefault();
      selectPidSuggestion(pidSuggestions[pidHighlight]);
    } else if (e.key === "Escape") {
      setShowPidDropdown(false);
    }
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setShowPidDropdown(false);
    runPrediction();
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
            {/* PID Mode: Internal DB search */}
            {mode === "pid" && (
              <div className="relative" ref={pidDropdownRef}>
                <label className="block text-xs font-medium text-sand-500 mb-1.5">
                  Search by PID or Street Name
                </label>
                <div className="relative">
                  <svg className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-sand-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2">
                    <path strokeLinecap="round" strokeLinejoin="round" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                  </svg>
                  <input
                    type="text"
                    value={pidQuery}
                    onChange={(e) => { setPidQuery(e.target.value); setPid(e.target.value); }}
                    onFocus={() => pidSuggestions.length > 0 && setShowPidDropdown(true)}
                    onKeyDown={handlePidKeyDown}
                    placeholder="e.g. 012-345-678 or Main St..."
                    autoComplete="off"
                    className="w-full pl-10 pr-4 py-2.5 rounded-lg border border-sand-200 bg-white text-sand-900 text-sm placeholder:text-sand-300 focus:outline-none focus:border-teal-400 focus:ring-1 focus:ring-teal-200 transition"
                  />
                </div>
                {showPidDropdown && pidSuggestions.length > 0 && (
                  <div className="absolute z-50 top-full left-0 right-0 mt-1 bg-white border border-sand-200 rounded-xl shadow-lg overflow-hidden max-h-[400px] overflow-y-auto">
                    {pidSuggestions.map((s, i) => (
                      <button
                        key={s.pid}
                        type="button"
                        onClick={() => selectPidSuggestion(s)}
                        className={`w-full text-left px-4 py-3 flex items-center justify-between gap-3 transition ${
                          i === pidHighlight
                            ? "bg-teal-50"
                            : "hover:bg-sand-50"
                        } ${i > 0 ? "border-t border-sand-100" : ""}`}
                      >
                        <div className="min-w-0 flex-1">
                          <div className="text-sm text-sand-900 font-medium truncate">
                            {s.address}
                          </div>
                          <div className="text-xs text-sand-400 mt-0.5 flex items-center gap-2">
                            <span className="font-mono">{s.pid}</span>
                            <span>&middot;</span>
                            <span className="capitalize">{s.property_type}</span>
                            <span>&middot;</span>
                            <span>{s.neighbourhood}</span>
                          </div>
                        </div>
                        <div className="text-right shrink-0">
                          <div className="text-sm font-medium text-sand-700">
                            {formatCurrency(s.assessed_value)}
                          </div>
                          <div className="text-[10px] text-sand-400">assessed</div>
                        </div>
                      </button>
                    ))}
                  </div>
                )}
              </div>
            )}
            {/* Address Mode: Google Places autocomplete */}
            {mode === "address" && (
              <div>
                <label className="block text-xs font-medium text-sand-500 mb-1.5">
                  Street Address
                  <span className="ml-2 text-[10px] text-teal-500 font-normal">Powered by Google</span>
                </label>
                <div className="relative">
                  <svg className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-sand-400 z-10 pointer-events-none" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2">
                    <path strokeLinecap="round" strokeLinejoin="round" d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z" />
                    <path strokeLinecap="round" strokeLinejoin="round" d="M15 11a3 3 0 11-6 0 3 3 0 016 0z" />
                  </svg>
                  <input
                    ref={addressInputRef}
                    type="text"
                    placeholder="Start typing an address... e.g. 6149 Fremlin Street"
                    autoComplete="off"
                    className="google-pac-input w-full pl-10 pr-4 py-2.5 rounded-lg border border-sand-200 bg-white text-sand-900 text-sm placeholder:text-sand-300 focus:outline-none focus:border-teal-400 focus:ring-1 focus:ring-teal-200 transition"
                  />
                </div>
                {selectedCoords && (
                  <div className="flex items-center gap-2 mt-2">
                    <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full bg-teal-50 border border-teal-200 text-[11px] text-teal-700">
                      <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2"><path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" /></svg>
                      Address found
                    </span>
                    <span className="text-[10px] text-sand-400">
                      {selectedCoords.lat.toFixed(5)}, {selectedCoords.lng.toFixed(5)}
                    </span>
                    {loading && (
                      <span className="text-[10px] text-teal-600 flex items-center gap-1">
                        <svg className="animate-spin w-3 h-3" viewBox="0 0 24 24" fill="none">
                          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                        </svg>
                        Generating valuation...
                      </span>
                    )}
                  </div>
                )}
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
