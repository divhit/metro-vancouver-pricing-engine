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
  { value: "median_high", label: "MEDIAN (HIGH)" },
  { value: "median_low", label: "MEDIAN (LOW)" },
  { value: "yoy_high", label: "YOY GROWTH" },
  { value: "name", label: "A-Z" },
  { value: "count", label: "COUNT" },
];

export default function MarketPage() {
  return (
    <Suspense fallback={<div className="p-8 text-concrete-400 text-[11px] tracking-wider">LOADING MARKET DATA...</div>}>
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

  const TOOLTIP_STYLE = {
    background: "white",
    border: "2px solid #0a0a0a",
    borderRadius: 0,
    fontSize: 11,
    fontFamily: "JetBrains Mono",
    padding: "8px 12px",
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-end justify-between gap-2 border-b-2 border-concrete-900 pb-4">
        <div>
          <h1
            className="text-3xl sm:text-4xl text-concrete-900 tracking-[0.08em]"
            style={{ fontFamily: "var(--font-display)" }}
          >
            MARKET EXPLORER
          </h1>
          <p className="text-concrete-500 text-[11px] mt-1 tracking-wider">
            COMPARE NEIGHBOURHOODS // METRO VANCOUVER
          </p>
        </div>
        {usingDemo && (
          <span className="text-[10px] px-3 py-1 bg-signal text-white font-bold tracking-wider self-start sm:self-auto">
            DEMO DATA
          </span>
        )}
      </div>

      {/* Filters */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
        <div className="flex gap-0 border-2 border-concrete-900 overflow-x-auto">
          {PROPERTY_TYPES.map((t) => (
            <button
              key={t}
              onClick={() => setSelectedType(t)}
              className={`px-4 py-2 text-[10px] font-bold tracking-[0.15em] transition whitespace-nowrap uppercase border-r-2 border-concrete-900 last:border-r-0 ${
                selectedType === t
                  ? "bg-concrete-900 text-white"
                  : "bg-white text-concrete-500 hover:bg-concrete-100"
              }`}
            >
              {t === "all" ? "ALL TYPES" : t}
            </button>
          ))}
        </div>
        <div className="flex items-center gap-2">
          <span className="text-[9px] text-concrete-400 tracking-[0.2em] font-bold">SORT //</span>
          <select
            value={sortBy}
            onChange={(e) => setSortBy(e.target.value)}
            className="appearance-none px-3 py-2 pr-8 text-[10px] font-bold tracking-wider border-2 border-concrete-900 bg-white text-concrete-700 focus:outline-none focus:border-signal cursor-pointer"
            style={{
              backgroundImage: `url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='12' viewBox='0 0 12 12' fill='none'%3E%3Cpath d='M3 4.5L6 7.5L9 4.5' stroke='%230a0a0a' stroke-width='2' stroke-linecap='square'/%3E%3C/svg%3E")`,
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
          <div className="text-[10px] text-concrete-400 tracking-[0.3em] font-bold animate-pulse">LOADING...</div>
        </div>
      ) : (
        <>
          {/* 2025 vs 2026 Comparison Chart */}
          <div className="card-hairline p-4 sm:p-6">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between mb-4 gap-2">
              <div>
                <h3
                  className="text-lg sm:text-xl text-concrete-900 tracking-wide"
                  style={{ fontFamily: "var(--font-display)" }}
                >
                  2025 VS 2026 ASSESSMENT VALUES
                </h3>
                <p className="text-[10px] text-concrete-400 mt-0.5 tracking-wider">
                  CLICK ANY BAR TO VIEW TREND HISTORY
                </p>
              </div>
            </div>
            <div className="h-[280px] sm:h-[380px]">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart
                  data={barChartData}
                  margin={{ top: 10, right: 10, left: 10, bottom: 50 }}
                >
                  <CartesianGrid strokeDasharray="0" stroke="#dddbd5" vertical={false} />
                  <XAxis
                    dataKey="name"
                    tick={{ fontSize: 9, fill: "#5c574c", fontFamily: "JetBrains Mono" }}
                    angle={-45}
                    textAnchor="end"
                    height={60}
                  />
                  <YAxis
                    tick={{ fontSize: 10, fill: "#7a7568", fontFamily: "JetBrains Mono" }}
                    tickFormatter={(v) => `$${(v / 1000000).toFixed(1)}M`}
                  />
                  <Tooltip
                    contentStyle={TOOLTIP_STYLE}
                    formatter={(value) => [formatCurrency(value as number), ""]}
                    labelFormatter={(label, payload) => {
                      const fullName = (payload as Array<{ payload?: { fullName?: string } }>)?.[0]?.payload?.fullName;
                      return fullName || String(label);
                    }}
                  />
                  <Legend wrapperStyle={{ fontSize: 10, paddingTop: 8, fontFamily: "JetBrains Mono" }} />
                  <Bar dataKey="2025" fill="#c4c1b8" radius={0} barSize={12} name="2025" />
                  <Bar
                    dataKey="2026"
                    radius={0}
                    barSize={12}
                    name="2026"
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
                        fill={selected === d.code ? "#CC3100" : "#FF3D00"}
                        stroke={selected === d.code ? "#0a0a0a" : "none"}
                        strokeWidth={selected === d.code ? 2 : 0}
                      />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Selected Neighbourhood Trend */}
          {selected && selectedTrend && selectedTrend.trends.length >= 2 && (
            <div ref={trendRef} className="card-hairline p-4 sm:p-6 border-l-4 border-l-signal animate-fade-in-up">
              <div className="flex items-start justify-between mb-1">
                <div>
                  <div className="flex flex-wrap items-center gap-3">
                    <h3
                      className="text-2xl text-concrete-900 tracking-wide"
                      style={{ fontFamily: "var(--font-display)" }}
                    >
                      {selectedTrend.neighbourhood_name.toUpperCase()}
                    </h3>
                    {selectedSummary && (() => {
                      const yoy = getYoy(selectedSummary);
                      return yoy != null ? (
                        <span className={`text-[10px] font-bold px-2 py-0.5 tracking-wider ${
                          yoy >= 0 ? "bg-pos/10 text-pos" : "bg-neg/10 text-neg"
                        }`}>
                          {formatPercent(yoy)} YOY
                        </span>
                      ) : null;
                    })()}
                  </div>
                  <p className="text-[10px] text-concrete-400 mt-0.5 tracking-wider">
                    MEDIAN ASSESSED VALUE TREND // {selectedTrend.trends.length} YEARS
                  </p>
                </div>
                <button
                  onClick={() => setSelected(null)}
                  className="text-concrete-400 hover:text-signal transition p-1"
                >
                  <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="square">
                    <path d="M4 4l8 8M12 4l-8 8" />
                  </svg>
                </button>
              </div>

              {selectedSummary && (
                <div className="flex flex-wrap gap-3 sm:gap-4 mb-4 mt-3">
                  {Object.entries(selectedSummary.median_values).map(([type, val]) => (
                    <div key={type} className="flex items-center gap-2 text-[11px]">
                      <span className="text-concrete-400 uppercase font-bold tracking-wider">{type}:</span>
                      <span className="font-bold text-concrete-700">{formatCurrency(val)}</span>
                      <span className={`font-bold ${
                        (selectedSummary.yoy_changes[type] ?? 0) >= 0 ? "text-pos" : "text-neg"
                      }`}>
                        {formatPercent(selectedSummary.yoy_changes[type])}
                      </span>
                      <span className="text-concrete-300">|</span>
                      <span className="text-concrete-400">{formatNumber(selectedSummary.property_counts[type] || 0)}</span>
                    </div>
                  ))}
                </div>
              )}

              <div className="h-[250px] sm:h-[280px]">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={selectedTrend.trends} margin={{ top: 10, right: 20, left: 10, bottom: 10 }}>
                    <CartesianGrid strokeDasharray="0" stroke="#dddbd5" />
                    <XAxis
                      dataKey="year"
                      tick={{ fontSize: 11, fill: "#5c574c", fontWeight: 700, fontFamily: "JetBrains Mono" }}
                      tickLine={false}
                    />
                    <YAxis
                      tick={{ fontSize: 10, fill: "#7a7568", fontFamily: "JetBrains Mono" }}
                      tickFormatter={(v) => `$${(v / 1000000).toFixed(1)}M`}
                      width={60}
                      tickLine={false}
                      axisLine={false}
                    />
                    <Tooltip
                      contentStyle={TOOLTIP_STYLE}
                      formatter={(value) => [formatCurrency(value as number), "MEDIAN VALUE"]}
                      labelFormatter={(label) => `YEAR ${label}`}
                    />
                    <Line
                      type="linear"
                      dataKey="median_value"
                      stroke="#FF3D00"
                      strokeWidth={3}
                      dot={{ r: 5, fill: "#FF3D00", strokeWidth: 2, stroke: "white" }}
                      activeDot={{ r: 7, fill: "#CC3100", strokeWidth: 2, stroke: "white" }}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>

              <div className="flex justify-between mt-3 pt-3 border-t-2 border-concrete-200">
                {selectedTrend.trends.map((pt) => (
                  <div key={pt.year} className="text-center">
                    <div className="text-[10px] text-concrete-400 font-bold tracking-wider">{pt.year}</div>
                    <div className="text-lg text-concrete-900 mt-0.5" style={{ fontFamily: "var(--font-display)" }}>
                      {formatCurrency(pt.median_value)}
                    </div>
                    <div className="text-[9px] text-concrete-400 tracking-wider">{formatNumber(pt.count)} PROPS</div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* YoY Change Rankings */}
          {sortedTrends.length > 0 && (
            <div className="grid lg:grid-cols-2 gap-0 border-2 border-concrete-900">
              {/* Gainers */}
              <div className="p-4 sm:p-6 bg-white">
                <h3 className="text-[10px] font-bold text-concrete-900 mb-4 flex items-center gap-2 tracking-[0.2em]">
                  <span className="w-3 h-3 bg-pos" />
                  TOP GAINERS // YOY
                </h3>
                <div className="space-y-0">
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
                        className={`flex items-center justify-between py-2.5 px-3 cursor-pointer transition border-b border-concrete-200 last:border-b-0 ${
                          selected === t.neighbourhood_code
                            ? "bg-signal-bg border-l-[3px] border-l-signal"
                            : "hover:bg-concrete-50"
                        }`}
                      >
                        <div className="flex items-center gap-3">
                          <span className="text-[10px] text-concrete-400 w-4 tabular-nums font-bold">{String(i + 1).padStart(2, "0")}</span>
                          <span className="text-[11px] text-concrete-800 font-bold tracking-wider uppercase">{t.neighbourhood_name}</span>
                        </div>
                        <div className="text-right flex items-center gap-3">
                          <span className="text-[10px] text-concrete-500 tabular-nums">{formatCurrency(t.val2026)}</span>
                          <span className="text-[10px] font-bold text-pos bg-pos/10 px-2 py-0.5 tabular-nums tracking-wider">
                            +{t.change?.toFixed(1)}%
                          </span>
                        </div>
                      </div>
                    ))}
                </div>
              </div>
              {/* Decliners */}
              <div className="p-4 sm:p-6 bg-white border-t-2 lg:border-t-0 lg:border-l-2 border-concrete-900">
                <h3 className="text-[10px] font-bold text-concrete-900 mb-4 flex items-center gap-2 tracking-[0.2em]">
                  <span className="w-3 h-3 bg-neg" />
                  LARGEST DECLINES // YOY
                </h3>
                <div className="space-y-0">
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
                        className={`flex items-center justify-between py-2.5 px-3 cursor-pointer transition border-b border-concrete-200 last:border-b-0 ${
                          selected === t.neighbourhood_code
                            ? "bg-signal-bg border-l-[3px] border-l-signal"
                            : "hover:bg-concrete-50"
                        }`}
                      >
                        <div className="flex items-center gap-3">
                          <span className="text-[10px] text-concrete-400 w-4 tabular-nums font-bold">{String(i + 1).padStart(2, "0")}</span>
                          <span className="text-[11px] text-concrete-800 font-bold tracking-wider uppercase">{t.neighbourhood_name}</span>
                        </div>
                        <div className="text-right flex items-center gap-3">
                          <span className="text-[10px] text-concrete-500 tabular-nums">{formatCurrency(t.val2026)}</span>
                          <span className="text-[10px] font-bold text-neg bg-neg/10 px-2 py-0.5 tabular-nums tracking-wider">
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
            <div className="px-4 py-3 border-b-2 border-concrete-900 bg-concrete-900">
              <h3 className="text-[10px] font-bold text-white tracking-[0.2em]">
                ALL NEIGHBOURHOODS
              </h3>
              <p className="text-[9px] text-concrete-400 mt-0.5 tracking-wider">
                CLICK A ROW TO VIEW ASSESSED VALUE HISTORY
              </p>
            </div>
            <div className="overflow-x-auto">
            <table className="w-full text-[11px] min-w-[500px]">
              <thead>
                <tr className="border-b-2 border-concrete-900 bg-concrete-50">
                  {["NEIGHBOURHOOD", "MEDIAN VALUE", "YOY CHANGE", "PROPERTIES"].map((h) => (
                    <th
                      key={h}
                      className="text-left py-3 px-4 text-[9px] font-bold text-concrete-500 tracking-[0.2em]"
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
                      className={`border-b border-concrete-200 cursor-pointer transition ${
                        isActive
                          ? "bg-signal-bg border-l-[3px] border-l-signal"
                          : "hover:bg-concrete-50"
                      }`}
                    >
                      <td className="py-3 px-4 text-concrete-900 font-bold tracking-wider uppercase">
                        {s.neighbourhood_name}
                      </td>
                      <td className="py-3 px-4 text-concrete-700 tabular-nums tracking-wider" style={{ fontFamily: "var(--font-display)" }}>
                        {formatCurrency(getMedian(s))}
                      </td>
                      <td className="py-3 px-4">
                        <span
                          className={`text-[10px] font-bold px-2 py-0.5 tabular-nums tracking-wider ${
                            yoy != null && yoy >= 0
                              ? "bg-pos/10 text-pos"
                              : yoy != null
                              ? "bg-neg/10 text-neg"
                              : "text-concrete-400"
                          }`}
                        >
                          {formatPercent(yoy)}
                        </span>
                      </td>
                      <td className="py-3 px-4 text-concrete-500 tabular-nums tracking-wider">
                        {formatNumber(getCount(s))}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
