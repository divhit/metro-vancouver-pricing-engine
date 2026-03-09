"use client";

import { Suspense, useEffect, useState, useMemo, useRef } from "react";
import { useSearchParams } from "next/navigation";
import { api, type MarketSummary, type NeighbourhoodTrend } from "@/lib/api";
import { formatCurrency, formatPercent, formatNumber } from "@/lib/format";
import {
  BarChart,
  Bar,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
  Cell,
  Legend,
} from "recharts";

const DEMO_SUMMARIES: MarketSummary[] = [
  { neighbourhood_code: "DOWNTOWN", neighbourhood_name: "Downtown", property_counts: { condo: 14200, townhome: 320, detached: 45 }, median_values: { condo: 685000, townhome: 1250000, detached: 2800000 }, yoy_changes: { condo: 3.2, townhome: 1.8, detached: -0.5 }, interest_rate: 4.79 },
  { neighbourhood_code: "KITSILANO", neighbourhood_name: "Kitsilano", property_counts: { condo: 4800, townhome: 1100, detached: 2900 }, median_values: { condo: 725000, townhome: 1450000, detached: 3200000 }, yoy_changes: { condo: 4.1, townhome: 2.3, detached: 1.7 }, interest_rate: 4.79 },
  { neighbourhood_code: "MOUNT-PLEASANT", neighbourhood_name: "Mount Pleasant", property_counts: { condo: 5200, townhome: 800, detached: 1600 }, median_values: { condo: 620000, townhome: 1180000, detached: 1950000 }, yoy_changes: { condo: 5.8, townhome: 3.1, detached: 2.4 }, interest_rate: 4.79 },
  { neighbourhood_code: "KERRISDALE", neighbourhood_name: "Kerrisdale", property_counts: { condo: 2100, townhome: 450, detached: 3200 }, median_values: { condo: 780000, townhome: 1600000, detached: 4100000 }, yoy_changes: { condo: 2.1, townhome: 0.8, detached: -1.2 }, interest_rate: 4.79 },
  { neighbourhood_code: "DUNBAR-SOUTHLANDS", neighbourhood_name: "Dunbar-Southlands", property_counts: { condo: 900, townhome: 280, detached: 4100 }, median_values: { condo: 690000, townhome: 1350000, detached: 3800000 }, yoy_changes: { condo: 3.5, townhome: 1.9, detached: 0.3 }, interest_rate: 4.79 },
  { neighbourhood_code: "FAIRVIEW", neighbourhood_name: "Fairview", property_counts: { condo: 7800, townhome: 600, detached: 800 }, median_values: { condo: 710000, townhome: 1380000, detached: 2600000 }, yoy_changes: { condo: 4.4, townhome: 2.7, detached: 1.1 }, interest_rate: 4.79 },
  { neighbourhood_code: "WEST-END", neighbourhood_name: "West End", property_counts: { condo: 18500, townhome: 120, detached: 20 }, median_values: { condo: 595000, townhome: 1100000, detached: 2200000 }, yoy_changes: { condo: 3.9, townhome: 2.0, detached: 0.0 }, interest_rate: 4.79 },
  { neighbourhood_code: "HASTINGS-SUNRISE", neighbourhood_name: "Hastings-Sunrise", property_counts: { condo: 2400, townhome: 950, detached: 5200 }, median_values: { condo: 520000, townhome: 980000, detached: 1650000 }, yoy_changes: { condo: 6.2, townhome: 4.5, detached: 3.8 }, interest_rate: 4.79 },
  { neighbourhood_code: "GRANDVIEW-WOODLAND", neighbourhood_name: "Grandview-Woodland", property_counts: { condo: 3100, townhome: 600, detached: 2800 }, median_values: { condo: 570000, townhome: 1050000, detached: 1800000 }, yoy_changes: { condo: 5.1, townhome: 3.4, detached: 2.9 }, interest_rate: 4.79 },
  { neighbourhood_code: "RENFREW-COLLINGWOOD", neighbourhood_name: "Renfrew-Collingwood", property_counts: { condo: 4500, townhome: 1200, detached: 6800 }, median_values: { condo: 510000, townhome: 920000, detached: 1550000 }, yoy_changes: { condo: 5.5, townhome: 4.1, detached: 3.2 }, interest_rate: 4.79 },
  { neighbourhood_code: "SOUTH-CAMBIE", neighbourhood_name: "South Cambie", property_counts: { condo: 1800, townhome: 350, detached: 1200 }, median_values: { condo: 750000, townhome: 1480000, detached: 3500000 }, yoy_changes: { condo: 3.8, townhome: 2.5, detached: 1.0 }, interest_rate: 4.79 },
  { neighbourhood_code: "MARPOLE", neighbourhood_name: "Marpole", property_counts: { condo: 3600, townhome: 700, detached: 2200 }, median_values: { condo: 560000, townhome: 1100000, detached: 1900000 }, yoy_changes: { condo: 4.8, townhome: 3.0, detached: 2.1 }, interest_rate: 4.79 },
];

const PROPERTY_TYPES = ["all", "condo", "townhome", "detached"];
const SORT_OPTIONS = [
  { value: "median_high", label: "Median (High to Low)" },
  { value: "median_low", label: "Median (Low to High)" },
  { value: "yoy_high", label: "YoY Growth (Highest)" },
  { value: "name", label: "Alphabetical" },
  { value: "count", label: "Property Count" },
];

export default function MarketPage() {
  return (
    <Suspense fallback={<div className="p-8 text-sand-400">Loading market data...</div>}>
      <MarketContent />
    </Suspense>
  );
}

function MarketContent() {
  const searchParams = useSearchParams();
  const highlightCode = searchParams.get("neighbourhood");
  const trendRef = useRef<HTMLDivElement>(null);

  const [summaries, setSummaries] = useState<MarketSummary[]>([]);
  const [trends, setTrends] = useState<NeighbourhoodTrend[]>([]);
  const [loading, setLoading] = useState(true);
  const [usingDemo, setUsingDemo] = useState(false);
  const [selectedType, setSelectedType] = useState("all");
  const [sortBy, setSortBy] = useState("median_high");
  const [selected, setSelected] = useState<string | null>(highlightCode);

  useEffect(() => {
    async function load() {
      try {
        const [s, t] = await Promise.all([
          api.getMarketAll(),
          api.getMarketTrends(),
        ]);
        setSummaries(s);
        setTrends(t);
      } catch {
        setSummaries(DEMO_SUMMARIES);
        setUsingDemo(true);
      } finally {
        setLoading(false);
      }
    }
    load();
  }, []);

  useEffect(() => {
    if (usingDemo) return;
    api.getMarketTrends(selectedType === "all" ? undefined : selectedType)
      .then(setTrends)
      .catch(() => {});
  }, [selectedType, usingDemo]);

  const getMedian = (s: MarketSummary) => {
    if (selectedType === "all") {
      const vals = Object.values(s.median_values);
      return vals.length > 0 ? vals.reduce((a, b) => a + b, 0) / vals.length : 0;
    }
    return s.median_values[selectedType] || 0;
  };

  const getYoy = (s: MarketSummary): number | null => {
    if (selectedType === "all") {
      const vals = Object.values(s.yoy_changes).filter((v): v is number => v != null);
      return vals.length > 0 ? vals.reduce((a, b) => a + b, 0) / vals.length : null;
    }
    return s.yoy_changes[selectedType] ?? null;
  };

  const getCount = (s: MarketSummary) => {
    if (selectedType === "all") {
      return Object.values(s.property_counts).reduce((a, b) => a + b, 0);
    }
    return s.property_counts[selectedType] || 0;
  };

  const sorted = useMemo(() => {
    const arr = [...summaries];
    switch (sortBy) {
      case "name":
        arr.sort((a, b) => a.neighbourhood_name.localeCompare(b.neighbourhood_name));
        break;
      case "median_high":
        arr.sort((a, b) => getMedian(b) - getMedian(a));
        break;
      case "median_low":
        arr.sort((a, b) => getMedian(a) - getMedian(b));
        break;
      case "yoy_high":
        arr.sort((a, b) => (getYoy(b) ?? -999) - (getYoy(a) ?? -999));
        break;
      case "count":
        arr.sort((a, b) => getCount(b) - getCount(a));
        break;
    }
    return arr;
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [summaries, sortBy, selectedType]);

  // Sort trends in the same order as the sorted summaries
  const sortedTrends = useMemo(() => {
    const orderMap = new Map(sorted.map((s, i) => [s.neighbourhood_code, i]));
    return [...trends]
      .filter((t) => t.trends.length >= 2)
      .sort((a, b) => {
        const aIdx = orderMap.get(a.neighbourhood_code) ?? 999;
        const bIdx = orderMap.get(b.neighbourhood_code) ?? 999;
        return aIdx - bIdx;
      });
  }, [trends, sorted]);

  const barChartData = useMemo(() => {
    return sortedTrends.map((t) => {
      const y2025 = t.trends.find((p) => p.year === 2025);
      const y2026 = t.trends.find((p) => p.year === 2026);
      return {
        name: t.neighbourhood_name.length > 12
          ? t.neighbourhood_name.slice(0, 10) + "\u2026"
          : t.neighbourhood_name,
        fullName: t.neighbourhood_name,
        "2025": y2025?.median_value ?? 0,
        "2026": y2026?.median_value ?? 0,
        code: t.neighbourhood_code,
      };
    });
  }, [sortedTrends]);

  const selectedSummary = selected
    ? summaries.find((s) => s.neighbourhood_code === selected)
    : null;

  const selectedTrend = selected
    ? trends.find((t) => t.neighbourhood_code === selected)
    : null;

  function handleSelect(code: string) {
    const next = selected === code ? null : code;
    setSelected(next);
    if (next && trendRef.current) {
      setTimeout(() => {
        trendRef.current?.scrollIntoView({ behavior: "smooth", block: "nearest" });
      }, 50);
    }
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-end justify-between">
        <div>
          <h1
            className="text-3xl text-sand-900 tracking-tight font-medium"
            style={{ fontFamily: "var(--font-display)" }}
          >
            Market Explorer
          </h1>
          <p className="text-sand-500 text-sm mt-1">
            Compare neighbourhoods across Metro Vancouver
          </p>
        </div>
        {usingDemo && (
          <span className="text-xs px-3 py-1 rounded-full bg-amber-50 text-amber-600 border border-amber-200">
            Demo Data
          </span>
        )}
      </div>

      {/* Filters */}
      <div className="flex items-center justify-between">
        <div className="flex gap-1 p-1 bg-sand-100 rounded-lg">
          {PROPERTY_TYPES.map((t) => (
            <button
              key={t}
              onClick={() => setSelectedType(t)}
              className={`px-3 py-1.5 text-xs font-medium rounded-md transition capitalize ${
                selectedType === t
                  ? "bg-white text-sand-900 shadow-sm"
                  : "text-sand-500 hover:text-sand-700"
              }`}
            >
              {t === "all" ? "All Types" : t}
            </button>
          ))}
        </div>
        <div className="flex items-center gap-2">
          <span className="text-[11px] text-sand-400 uppercase tracking-wider font-medium">Sort</span>
          <select
            value={sortBy}
            onChange={(e) => setSortBy(e.target.value)}
            className="appearance-none px-3 py-1.5 pr-8 text-xs font-medium rounded-lg border border-sand-200 bg-white text-sand-700 focus:outline-none focus:ring-2 focus:ring-teal-400/30 focus:border-teal-400 cursor-pointer"
            style={{
              backgroundImage: `url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='12' viewBox='0 0 12 12' fill='none'%3E%3Cpath d='M3 4.5L6 7.5L9 4.5' stroke='%239a9080' stroke-width='1.5' stroke-linecap='round' stroke-linejoin='round'/%3E%3C/svg%3E")`,
              backgroundRepeat: "no-repeat",
              backgroundPosition: "right 8px center",
            }}
          >
            {SORT_OPTIONS.map((o) => (
              <option key={o.value} value={o.value}>{o.label}</option>
            ))}
          </select>
        </div>
      </div>

      {loading ? (
        <div className="card-hairline p-12 flex items-center justify-center">
          <svg className="animate-spin w-6 h-6 text-teal-500" viewBox="0 0 24 24" fill="none">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
          </svg>
        </div>
      ) : (
        <>
          {/* 2025 vs 2026 Comparison Chart */}
          <div className="card-hairline p-6">
            <div className="flex items-center justify-between mb-4">
              <div>
                <h3 className="text-sm font-semibold text-sand-800">
                  2025 vs 2026 Assessment Values
                </h3>
                <p className="text-xs text-sand-400 mt-0.5">
                  Click any bar to view that neighbourhood&apos;s full trend history
                </p>
              </div>
            </div>
            <div className="h-[380px]">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart
                  data={barChartData}
                  margin={{ top: 10, right: 10, left: 10, bottom: 50 }}
                >
                  <CartesianGrid strokeDasharray="3 3" stroke="#e8e4dd" vertical={false} />
                  <XAxis
                    dataKey="name"
                    tick={{ fontSize: 9, fill: "#7d7365" }}
                    angle={-45}
                    textAnchor="end"
                    height={60}
                  />
                  <YAxis
                    tick={{ fontSize: 10, fill: "#9a9080" }}
                    tickFormatter={(v) => `$${(v / 1000000).toFixed(1)}M`}
                  />
                  <Tooltip
                    contentStyle={{
                      background: "white",
                      border: "1px solid #e8e4dd",
                      borderRadius: 10,
                      fontSize: 12,
                      padding: "10px 14px",
                    }}
                    formatter={(value) => [formatCurrency(value as number), ""]}
                    labelFormatter={(label, payload) => {
                      const fullName = (payload as Array<{ payload?: { fullName?: string } }>)?.[0]?.payload?.fullName;
                      return fullName || String(label);
                    }}
                  />
                  <Legend wrapperStyle={{ fontSize: 11, paddingTop: 8 }} />
                  <Bar dataKey="2025" fill="#d5cfc5" radius={[3, 3, 0, 0]} barSize={12} name="2025 Assessment" />
                  <Bar
                    dataKey="2026"
                    radius={[3, 3, 0, 0]}
                    barSize={12}
                    name="2026 Assessment"
                    cursor="pointer"
                    onClick={(_data: unknown, index: number) => {
                      if (index != null && barChartData[index]) {
                        handleSelect(barChartData[index].code);
                      }
                    }}
                  >
                    {barChartData.map((d) => (
                      <Cell
                        key={d.code}
                        fill={selected === d.code ? "#059e92" : "#06c2ae"}
                        stroke={selected === d.code ? "#047d73" : "none"}
                        strokeWidth={selected === d.code ? 2 : 0}
                      />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Selected Neighbourhood Trend — appears when a neighbourhood is clicked */}
          {selected && selectedTrend && selectedTrend.trends.length >= 2 && (
            <div ref={trendRef} className="card-hairline p-6 border-l-4 border-l-teal-500 animate-fade-in-up">
              <div className="flex items-start justify-between mb-1">
                <div>
                  <div className="flex items-center gap-3">
                    <h3
                      className="text-xl text-sand-900 font-medium"
                      style={{ fontFamily: "var(--font-display)" }}
                    >
                      {selectedTrend.neighbourhood_name}
                    </h3>
                    {selectedSummary && (() => {
                      const yoy = getYoy(selectedSummary);
                      return yoy != null ? (
                        <span className={`text-xs font-medium px-2 py-0.5 rounded-full ${
                          yoy >= 0 ? "bg-emerald-50 text-emerald-600" : "bg-rose-50 text-rose-500"
                        }`}>
                          {formatPercent(yoy)} YoY
                        </span>
                      ) : null;
                    })()}
                  </div>
                  <p className="text-xs text-sand-400 mt-0.5">
                    Median assessed value trend &mdash; {selectedTrend.trends.length} years of data
                  </p>
                </div>
                <button
                  onClick={() => setSelected(null)}
                  className="text-sand-400 hover:text-sand-600 transition p-1"
                >
                  <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round">
                    <path d="M4 4l8 8M12 4l-8 8" />
                  </svg>
                </button>
              </div>

              {/* Property type breakdown inline */}
              {selectedSummary && (
                <div className="flex gap-4 mb-4 mt-3">
                  {Object.entries(selectedSummary.median_values).map(([type, val]) => (
                    <div key={type} className="flex items-center gap-2 text-xs">
                      <span className="text-sand-400 capitalize">{type}:</span>
                      <span className="font-medium text-sand-700">{formatCurrency(val)}</span>
                      <span className={`${
                        (selectedSummary.yoy_changes[type] ?? 0) >= 0 ? "text-emerald-600" : "text-rose-500"
                      }`}>
                        {formatPercent(selectedSummary.yoy_changes[type])}
                      </span>
                      <span className="text-sand-300">|</span>
                      <span className="text-sand-400">{formatNumber(selectedSummary.property_counts[type] || 0)}</span>
                    </div>
                  ))}
                </div>
              )}

              <div className="h-[280px]">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={selectedTrend.trends} margin={{ top: 10, right: 20, left: 10, bottom: 10 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#e8e4dd" />
                    <XAxis
                      dataKey="year"
                      tick={{ fontSize: 11, fill: "#7d7365", fontWeight: 500 }}
                      tickLine={false}
                    />
                    <YAxis
                      tick={{ fontSize: 10, fill: "#9a9080" }}
                      tickFormatter={(v) => `$${(v / 1000000).toFixed(1)}M`}
                      width={60}
                      tickLine={false}
                      axisLine={false}
                    />
                    <Tooltip
                      contentStyle={{
                        background: "white",
                        border: "1px solid #e8e4dd",
                        borderRadius: 10,
                        fontSize: 12,
                        padding: "10px 14px",
                        boxShadow: "0 4px 12px rgba(0,0,0,0.08)",
                      }}
                      formatter={(value) => [formatCurrency(value as number), "Median Value"]}
                      labelFormatter={(label) => `Assessment Year ${label}`}
                    />
                    <Line
                      type="monotone"
                      dataKey="median_value"
                      stroke="#059e92"
                      strokeWidth={2.5}
                      dot={{ r: 5, fill: "#059e92", strokeWidth: 2, stroke: "white" }}
                      activeDot={{ r: 7, fill: "#047d73", strokeWidth: 2, stroke: "white" }}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>

              {/* Year-by-year values row */}
              <div className="flex justify-between mt-3 pt-3 border-t border-sand-100">
                {selectedTrend.trends.map((pt) => (
                  <div key={pt.year} className="text-center">
                    <div className="text-[11px] text-sand-400">{pt.year}</div>
                    <div className="text-sm font-medium text-sand-800" style={{ fontFamily: "var(--font-display)" }}>
                      {formatCurrency(pt.median_value)}
                    </div>
                    <div className="text-[10px] text-sand-400">{formatNumber(pt.count)} props</div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* YoY Change Rankings */}
          {sortedTrends.length > 0 && (
            <div className="grid lg:grid-cols-2 gap-6">
              {/* Biggest Gainers */}
              <div className="card-hairline p-6">
                <h3 className="text-sm font-semibold text-sand-800 mb-4 flex items-center gap-2">
                  <span className="w-2 h-2 rounded-full bg-emerald-500" />
                  Top Gainers (YoY)
                </h3>
                <div className="space-y-1">
                  {sortedTrends
                    .map((t) => {
                      const y2025 = t.trends.find((p) => p.year === 2025);
                      const y2026 = t.trends.find((p) => p.year === 2026);
                      const change = y2025 && y2026 && y2025.median_value > 0
                        ? ((y2026.median_value - y2025.median_value) / y2025.median_value) * 100
                        : null;
                      return { ...t, change, val2026: y2026?.median_value ?? 0 };
                    })
                    .filter((t) => t.change != null && t.change > 0)
                    .sort((a, b) => (b.change ?? 0) - (a.change ?? 0))
                    .slice(0, 8)
                    .map((t, i) => (
                      <div
                        key={t.neighbourhood_code}
                        onClick={() => handleSelect(t.neighbourhood_code)}
                        className={`flex items-center justify-between py-2.5 px-3 rounded-lg cursor-pointer transition ${
                          selected === t.neighbourhood_code
                            ? "bg-teal-50 border border-teal-200/60"
                            : "hover:bg-sand-50 border border-transparent"
                        }`}
                      >
                        <div className="flex items-center gap-3">
                          <span className="text-xs text-sand-400 w-4 tabular-nums">{i + 1}</span>
                          <span className="text-sm text-sand-800">{t.neighbourhood_name}</span>
                        </div>
                        <div className="text-right flex items-center gap-3">
                          <span className="text-xs text-sand-500 tabular-nums">{formatCurrency(t.val2026)}</span>
                          <span className="text-xs font-medium text-emerald-600 bg-emerald-50 px-2 py-0.5 rounded-full tabular-nums">
                            +{t.change?.toFixed(1)}%
                          </span>
                        </div>
                      </div>
                    ))}
                </div>
              </div>
              {/* Biggest Decliners */}
              <div className="card-hairline p-6">
                <h3 className="text-sm font-semibold text-sand-800 mb-4 flex items-center gap-2">
                  <span className="w-2 h-2 rounded-full bg-rose-500" />
                  Largest Declines (YoY)
                </h3>
                <div className="space-y-1">
                  {sortedTrends
                    .map((t) => {
                      const y2025 = t.trends.find((p) => p.year === 2025);
                      const y2026 = t.trends.find((p) => p.year === 2026);
                      const change = y2025 && y2026 && y2025.median_value > 0
                        ? ((y2026.median_value - y2025.median_value) / y2025.median_value) * 100
                        : null;
                      return { ...t, change, val2026: y2026?.median_value ?? 0 };
                    })
                    .filter((t) => t.change != null && t.change <= 0)
                    .sort((a, b) => (a.change ?? 0) - (b.change ?? 0))
                    .slice(0, 8)
                    .map((t, i) => (
                      <div
                        key={t.neighbourhood_code}
                        onClick={() => handleSelect(t.neighbourhood_code)}
                        className={`flex items-center justify-between py-2.5 px-3 rounded-lg cursor-pointer transition ${
                          selected === t.neighbourhood_code
                            ? "bg-teal-50 border border-teal-200/60"
                            : "hover:bg-sand-50 border border-transparent"
                        }`}
                      >
                        <div className="flex items-center gap-3">
                          <span className="text-xs text-sand-400 w-4 tabular-nums">{i + 1}</span>
                          <span className="text-sm text-sand-800">{t.neighbourhood_name}</span>
                        </div>
                        <div className="text-right flex items-center gap-3">
                          <span className="text-xs text-sand-500 tabular-nums">{formatCurrency(t.val2026)}</span>
                          <span className="text-xs font-medium text-rose-500 bg-rose-50 px-2 py-0.5 rounded-full tabular-nums">
                            {t.change?.toFixed(1)}%
                          </span>
                        </div>
                      </div>
                    ))}
                </div>
              </div>
            </div>
          )}

          {/* Neighbourhood Table */}
          <div className="card-hairline overflow-hidden">
            <div className="px-4 py-3 border-b border-sand-200 bg-sand-50/30">
              <h3 className="text-sm font-semibold text-sand-800">
                All Neighbourhoods
              </h3>
              <p className="text-[11px] text-sand-400 mt-0.5">
                Click a row to view assessed value history
              </p>
            </div>
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-sand-200 bg-sand-50/50">
                  {["Neighbourhood", "Median Value", "YoY Change", "Properties"].map((h) => (
                    <th
                      key={h}
                      className="text-left py-3 px-4 text-[11px] font-medium text-sand-400 uppercase tracking-wider"
                    >
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {sorted.map((s) => {
                  const isActive = selected === s.neighbourhood_code;
                  const yoy = getYoy(s);
                  return (
                    <tr
                      key={s.neighbourhood_code}
                      onClick={() => handleSelect(s.neighbourhood_code)}
                      className={`border-b border-sand-100 cursor-pointer transition ${
                        isActive
                          ? "bg-teal-50/60 border-l-[3px] border-l-teal-500"
                          : "hover:bg-sand-50"
                      }`}
                    >
                      <td className="py-3 px-4 text-sand-800 font-medium">
                        {s.neighbourhood_name}
                      </td>
                      <td className="py-3 px-4 text-sand-700 tabular-nums" style={{ fontFamily: "var(--font-display)" }}>
                        {formatCurrency(getMedian(s))}
                      </td>
                      <td className="py-3 px-4">
                        <span
                          className={`text-xs font-medium px-2 py-0.5 rounded-full tabular-nums ${
                            yoy != null && yoy >= 0
                              ? "bg-emerald-50 text-emerald-600"
                              : yoy != null
                              ? "bg-rose-50 text-rose-500"
                              : "text-sand-400"
                          }`}
                        >
                          {formatPercent(yoy)}
                        </span>
                      </td>
                      <td className="py-3 px-4 text-sand-500 tabular-nums">
                        {formatNumber(getCount(s))}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </>
      )}
    </div>
  );
}
