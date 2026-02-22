"use client";

import { Suspense, useEffect, useState, useMemo } from "react";
import { useSearchParams } from "next/navigation";
import { api, type MarketSummary } from "@/lib/api";
import { formatCurrency, formatPercent, formatNumber } from "@/lib/format";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
  Cell,
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
  { value: "name", label: "Name" },
  { value: "median_high", label: "Median (High to Low)" },
  { value: "median_low", label: "Median (Low to High)" },
  { value: "yoy_high", label: "YoY Growth (High)" },
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

  const [summaries, setSummaries] = useState<MarketSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [usingDemo, setUsingDemo] = useState(false);
  const [selectedType, setSelectedType] = useState("all");
  const [sortBy, setSortBy] = useState("median_high");
  const [selected, setSelected] = useState<string | null>(highlightCode);

  useEffect(() => {
    async function load() {
      try {
        const s = await api.getMarketAll();
        setSummaries(s);
      } catch {
        setSummaries(DEMO_SUMMARIES);
        setUsingDemo(true);
      } finally {
        setLoading(false);
      }
    }
    load();
  }, []);

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

  const chartData = sorted.map((s) => ({
    name: s.neighbourhood_name.length > 14
      ? s.neighbourhood_name.slice(0, 12) + "..."
      : s.neighbourhood_name,
    median: getMedian(s),
    yoy: getYoy(s) ?? 0,
    code: s.neighbourhood_code,
  }));

  const selectedSummary = selected
    ? summaries.find((s) => s.neighbourhood_code === selected)
    : null;

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="flex items-end justify-between">
        <div>
          <h1
            className="text-3xl text-sand-900 tracking-tight"
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
      <div className="flex items-center gap-4">
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
        <select
          value={sortBy}
          onChange={(e) => setSortBy(e.target.value)}
          className="px-3 py-1.5 text-xs font-medium rounded-lg border border-sand-200 bg-white text-sand-700 focus:outline-none focus:border-teal-400"
        >
          {SORT_OPTIONS.map((o) => (
            <option key={o.value} value={o.value}>{o.label}</option>
          ))}
        </select>
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
          {/* Chart */}
          <div className="card-hairline p-6">
            <h3 className="text-sm font-semibold text-sand-800 mb-4">
              Median Values by Neighbourhood
            </h3>
            <div className="h-[340px]">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={chartData} margin={{ top: 0, right: 0, left: 0, bottom: 40 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e8e4dd" vertical={false} />
                  <XAxis
                    dataKey="name"
                    tick={{ fontSize: 10, fill: "#7d7365" }}
                    angle={-45}
                    textAnchor="end"
                    height={60}
                  />
                  <YAxis
                    tick={{ fontSize: 10, fill: "#9a9080" }}
                    tickFormatter={(v) => formatCurrency(v)}
                  />
                  <Tooltip
                    contentStyle={{
                      background: "white",
                      border: "1px solid #e8e4dd",
                      borderRadius: 8,
                      fontSize: 12,
                    }}
                    formatter={(value) => [formatCurrency(value as number), "Median"]}
                  />
                  <Bar
                    dataKey="median"
                    radius={[4, 4, 0, 0]}
                    cursor="pointer"
                    onClick={(_data, index) => {
                      if (index != null && chartData[index]) setSelected(chartData[index].code);
                    }}
                  >
                    {chartData.map((entry) => (
                      <Cell
                        key={entry.code}
                        fill={selected === entry.code ? "#06c2ae" : "#c7fef4"}
                        stroke={selected === entry.code ? "#077d73" : "none"}
                        strokeWidth={selected === entry.code ? 1 : 0}
                      />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Detail + Table */}
          <div className="grid lg:grid-cols-[1fr,380px] gap-6">
            {/* Table */}
            <div className="card-hairline overflow-hidden">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-sand-200 bg-sand-50/50">
                    {["Neighbourhood", "Median Value", "YoY", "Properties"].map((h) => (
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
                        onClick={() => setSelected(isActive ? null : s.neighbourhood_code)}
                        className={`border-b border-sand-100 cursor-pointer transition ${
                          isActive
                            ? "bg-teal-50/50 border-l-2 border-l-teal-500"
                            : "hover:bg-sand-50"
                        }`}
                      >
                        <td className="py-3 px-4 text-sand-800 font-medium">
                          {s.neighbourhood_name}
                        </td>
                        <td className="py-3 px-4 text-sand-700" style={{ fontFamily: "var(--font-display)" }}>
                          {formatCurrency(getMedian(s))}
                        </td>
                        <td className="py-3 px-4">
                          <span
                            className={`text-xs font-medium px-2 py-0.5 rounded-full ${
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
                        <td className="py-3 px-4 text-sand-500">
                          {formatNumber(getCount(s))}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>

            {/* Selected detail panel */}
            <div className="card-hairline p-6">
              {selectedSummary ? (
                <div className="space-y-5">
                  <div>
                    <h3
                      className="text-xl text-sand-900"
                      style={{ fontFamily: "var(--font-display)" }}
                    >
                      {selectedSummary.neighbourhood_name}
                    </h3>
                    <p className="text-xs text-sand-400 mt-0.5">
                      {selectedSummary.neighbourhood_code}
                    </p>
                  </div>

                  <div className="space-y-4">
                    <h4 className="text-xs font-medium text-sand-400 uppercase tracking-wider">
                      By Property Type
                    </h4>
                    {Object.entries(selectedSummary.median_values).map(([type, val]) => (
                      <div key={type} className="flex items-center justify-between py-2 border-b border-sand-100">
                        <div>
                          <span className="text-sm text-sand-700 capitalize">{type}</span>
                          <span className="text-xs text-sand-400 ml-2">
                            ({formatNumber(selectedSummary.property_counts[type] || 0)})
                          </span>
                        </div>
                        <div className="text-right">
                          <div className="text-sm font-medium text-sand-800">
                            {formatCurrency(val)}
                          </div>
                          <div className={`text-[11px] ${
                            (selectedSummary.yoy_changes[type] ?? 0) >= 0
                              ? "text-emerald-600"
                              : "text-rose-500"
                          }`}>
                            {formatPercent(selectedSummary.yoy_changes[type])}
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>

                  {selectedSummary.interest_rate && (
                    <div className="pt-2">
                      <div className="text-[11px] text-sand-400 uppercase tracking-wider">
                        5-Year Fixed Rate
                      </div>
                      <div className="text-lg text-sand-800 mt-0.5" style={{ fontFamily: "var(--font-display)" }}>
                        {selectedSummary.interest_rate}%
                      </div>
                    </div>
                  )}
                </div>
              ) : (
                <div className="h-full flex flex-col items-center justify-center text-center py-12">
                  <svg className="w-10 h-10 text-sand-300 mb-3" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="1">
                    <path strokeLinecap="round" strokeLinejoin="round" d="M15 10.5a3 3 0 11-6 0 3 3 0 016 0z" />
                    <path strokeLinecap="round" strokeLinejoin="round" d="M19.5 10.5c0 7.142-7.5 11.25-7.5 11.25S4.5 17.642 4.5 10.5a7.5 7.5 0 1115 0z" />
                  </svg>
                  <p className="text-sm text-sand-400">
                    Select a neighbourhood to view details
                  </p>
                </div>
              )}
            </div>
          </div>
        </>
      )}
    </div>
  );
}
