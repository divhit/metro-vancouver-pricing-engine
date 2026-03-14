"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { api, type MarketSummary, type HealthResponse, type NeighbourhoodTrend } from "@/lib/api";
import { formatCurrency, formatPercent, formatNumber } from "@/lib/format";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
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
];

export default function DashboardPage() {
  const [summaries, setSummaries] = useState<MarketSummary[]>([]);
  const [trends, setTrends] = useState<NeighbourhoodTrend[]>([]);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [usingDemo, setUsingDemo] = useState(false);

  useEffect(() => {
    async function load() {
      try {
        const [s, h, t] = await Promise.all([
          api.getMarketAll(),
          api.getHealth(),
          api.getMarketTrends(),
        ]);
        setSummaries(s);
        setHealth(h);
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

  const totalProperties = summaries.reduce(
    (acc, s) => acc + Object.values(s.property_counts).reduce((a, b) => a + b, 0),
    0,
  );
  const avgMedian =
    summaries.length > 0
      ? summaries.reduce((acc, s) => {
          const vals = Object.values(s.median_values);
          return acc + (vals.length > 0 ? vals.reduce((a, b) => a + b, 0) / vals.length : 0);
        }, 0) / summaries.length
      : 0;
  const avgYoy =
    summaries.length > 0
      ? summaries.reduce((acc, s) => {
          const vals = Object.values(s.yoy_changes).filter((v): v is number => v != null);
          return acc + (vals.length > 0 ? vals.reduce((a, b) => a + b, 0) / vals.length : 0);
        }, 0) / summaries.length
      : 0;

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-end justify-between gap-2 border-b-2 border-concrete-900 pb-4">
        <div>
          <h1
            className="text-3xl sm:text-4xl text-concrete-900 tracking-[0.08em]"
            style={{ fontFamily: "var(--font-display)" }}
          >
            DASHBOARD // {getGreeting().toUpperCase()}
          </h1>
          <p className="text-concrete-500 text-[11px] mt-1 tracking-wider">
            METRO VANCOUVER MARKET OVERVIEW
          </p>
        </div>
        {usingDemo && (
          <span className="text-[10px] px-3 py-1 bg-signal text-white font-bold tracking-wider self-start sm:self-auto">
            DEMO DATA // NO API
          </span>
        )}
      </div>

      {/* KPI Cards */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-0 border-2 border-concrete-900">
        {[
          {
            label: "TOTAL PROPERTIES",
            value: loading ? "\u2014" : formatNumber(totalProperties),
            sub: `${summaries.length} neighbourhoods`,
          },
          {
            label: "AVG. MEDIAN VALUE",
            value: loading ? "\u2014" : formatCurrency(avgMedian),
            sub: "Across all types",
          },
          {
            label: "AVG. YOY CHANGE",
            value: loading ? "\u2014" : formatPercent(avgYoy),
            sub: "Year-over-year",
            highlight: true,
          },
          {
            label: "ENGINE STATUS",
            value: health?.status === "ok" ? "ONLINE" : health?.status?.toUpperCase() || "\u2014",
            sub: health ? `v${health.version} // ${health.model_count} models` : "Connecting...",
            status: health?.status,
          },
        ].map((kpi, i) => (
          <div
            key={kpi.label}
            className={`p-4 sm:p-5 bg-white animate-fade-in-up opacity-0 ${
              i > 0 ? "border-t-2 sm:border-t-0 sm:border-l-2 border-concrete-900" : ""
            } ${i === 2 ? "border-t-2 lg:border-t-0" : ""}`}
            style={{ animationDelay: `${i * 0.04}s`, animationFillMode: "forwards" }}
          >
            <div className="text-[9px] font-bold text-concrete-400 tracking-[0.2em]">
              {kpi.label}
            </div>
            <div
              className={`mt-2 text-2xl sm:text-3xl tracking-wider ${
                kpi.status === "ok"
                  ? "text-pos"
                  : kpi.status === "degraded"
                  ? "text-signal"
                  : kpi.highlight && avgYoy > 0
                  ? "text-pos"
                  : kpi.highlight && avgYoy < 0
                  ? "text-neg"
                  : "text-concrete-900"
              }`}
              style={{ fontFamily: "var(--font-display)" }}
            >
              {kpi.value}
            </div>
            <div className="text-[10px] text-concrete-400 mt-1 tracking-wider">{kpi.sub}</div>
          </div>
        ))}
      </div>

      {/* 2025 vs 2026 Trend Chart */}
      {!loading && trends.length > 0 && (
        <div className="card-hairline p-4 sm:p-6 animate-fade-in-up opacity-0" style={{ animationDelay: "0.2s", animationFillMode: "forwards" }}>
          <div className="flex flex-col sm:flex-row sm:items-center justify-between mb-4 gap-2">
            <div>
              <h2
                className="text-xl sm:text-2xl text-concrete-900 tracking-wide"
                style={{ fontFamily: "var(--font-display)" }}
              >
                2025 &rarr; 2026 ASSESSMENT TRENDS
              </h2>
              <p className="text-[10px] text-concrete-400 mt-0.5 tracking-wider">
                MEDIAN ASSESSED VALUES BY NEIGHBOURHOOD
              </p>
            </div>
            <Link
              href="/market"
              className="text-[10px] font-bold text-signal hover:text-signal-dark transition tracking-[0.15em]"
            >
              FULL ANALYSIS &rarr;
            </Link>
          </div>
          <div className="h-[250px] sm:h-[320px]">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={trends
                  .filter((t) => t.trends.length >= 2)
                  .map((t) => {
                    const y2025 = t.trends.find((p) => p.year === 2025);
                    const y2026 = t.trends.find((p) => p.year === 2026);
                    return {
                      name: t.neighbourhood_name.length > 11
                        ? t.neighbourhood_name.slice(0, 9) + "\u2026"
                        : t.neighbourhood_name,
                      "2025": y2025?.median_value ?? 0,
                      "2026": y2026?.median_value ?? 0,
                    };
                  })
                  .sort((a, b) => b["2026"] - a["2026"])}
                margin={{ top: 5, right: 5, left: 5, bottom: 50 }}
              >
                <CartesianGrid strokeDasharray="0" stroke="#dddbd5" vertical={false} />
                <XAxis
                  dataKey="name"
                  tick={{ fontSize: 9, fill: "#5c574c", fontFamily: "JetBrains Mono" }}
                  angle={-45}
                  textAnchor="end"
                  height={55}
                />
                <YAxis
                  tick={{ fontSize: 10, fill: "#7a7568", fontFamily: "JetBrains Mono" }}
                  tickFormatter={(v) => `$${(v / 1000000).toFixed(1)}M`}
                />
                <Tooltip
                  contentStyle={{
                    background: "white",
                    border: "2px solid #0a0a0a",
                    borderRadius: 0,
                    fontSize: 11,
                    fontFamily: "JetBrains Mono",
                  }}
                  formatter={(value) => [formatCurrency(value as number), ""]}
                />
                <Legend wrapperStyle={{ fontSize: 10, paddingTop: 4, fontFamily: "JetBrains Mono" }} />
                <Bar dataKey="2025" fill="#c4c1b8" radius={0} barSize={11} name="2025" />
                <Bar dataKey="2026" fill="#FF3D00" radius={0} barSize={11} name="2026" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {/* Neighbourhood Grid */}
      <div>
        <div className="flex items-center justify-between mb-4 border-b-2 border-concrete-900 pb-2">
          <h2
            className="text-xl sm:text-2xl text-concrete-900 tracking-wide"
            style={{ fontFamily: "var(--font-display)" }}
          >
            NEIGHBOURHOODS
          </h2>
          <Link
            href="/market"
            className="text-[10px] font-bold text-signal hover:text-signal-dark transition tracking-[0.15em]"
          >
            MARKET EXPLORER &rarr;
          </Link>
        </div>

        {loading ? (
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-0 border-2 border-concrete-900">
            {Array.from({ length: 6 }).map((_, i) => (
              <div key={i} className={`p-5 animate-pulse bg-white ${i > 0 ? "border-t-2 md:border-t-0 md:border-l-2 border-concrete-900" : ""}`}>
                <div className="h-4 bg-concrete-200 w-1/2 mb-3" />
                <div className="h-8 bg-concrete-100 w-2/3 mb-2" />
                <div className="h-3 bg-concrete-100 w-1/3" />
              </div>
            ))}
          </div>
        ) : (
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-0 border-2 border-concrete-900">
            {summaries.map((s, i) => (
              <NeighbourhoodCard key={s.neighbourhood_code} summary={s} index={i} />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

function NeighbourhoodCard({ summary, index }: { summary: MarketSummary; index: number }) {
  const totalCount = Object.values(summary.property_counts).reduce((a, b) => a + b, 0);
  const primaryType = Object.entries(summary.property_counts).sort(([, a], [, b]) => b - a)[0];
  const primaryMedian = primaryType ? summary.median_values[primaryType[0]] : 0;
  const primaryYoy = primaryType ? summary.yoy_changes[primaryType[0]] : null;

  return (
    <Link
      href={`/market?neighbourhood=${summary.neighbourhood_code}`}
      className={`p-5 bg-white hover:bg-signal-bg transition-colors duration-100 group animate-fade-in-up opacity-0 ${
        index > 0 ? "border-t-2 border-concrete-900" : ""
      } md:border-t-2 ${index % 2 !== 0 ? "md:border-l-2" : ""} ${index % 3 !== 0 ? "lg:border-l-2" : ""} border-concrete-900 first:border-t-0`}
      style={{ animationDelay: `${0.1 + index * 0.03}s`, animationFillMode: "forwards" }}
    >
      <div className="flex items-start justify-between">
        <div>
          <h3 className="text-[11px] font-bold text-concrete-900 group-hover:text-signal transition-colors tracking-wider uppercase">
            {summary.neighbourhood_name}
          </h3>
          <p className="text-[10px] text-concrete-400 mt-0.5 tracking-wider">
            {formatNumber(totalCount)} PROPERTIES
          </p>
        </div>
        {primaryYoy != null && (
          <span
            className={`text-[10px] font-bold px-2 py-0.5 tracking-wider ${
              primaryYoy >= 0
                ? "bg-pos/10 text-pos"
                : "bg-neg/10 text-neg"
            }`}
          >
            {formatPercent(primaryYoy)}
          </span>
        )}
      </div>

      <div className="mt-3">
        <div
          className="text-2xl tracking-wider text-concrete-900"
          style={{ fontFamily: "var(--font-display)" }}
        >
          {primaryMedian > 0 ? formatCurrency(primaryMedian) : "\u2014"}
        </div>
        <div className="text-[9px] text-concrete-400 mt-0.5 tracking-[0.2em] uppercase font-bold">
          MEDIAN {primaryType?.[0] || "VALUE"}
        </div>
      </div>

      <div className="mt-3 flex gap-2 flex-wrap">
        {Object.entries(summary.property_counts).map(([type, count]) => (
          <span
            key={type}
            className="text-[9px] px-1.5 py-0.5 bg-concrete-100 text-concrete-500 font-bold tracking-wider uppercase"
          >
            {type}: {formatNumber(count)}
          </span>
        ))}
      </div>
    </Link>
  );
}

function getGreeting(): string {
  const hour = new Date().getHours();
  if (hour < 12) return "morning";
  if (hour < 17) return "afternoon";
  return "evening";
}
