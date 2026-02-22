"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { api, type MarketSummary, type HealthResponse } from "@/lib/api";
import { formatCurrency, formatPercent, formatNumber } from "@/lib/format";

// Demo data for when API is not connected
const DEMO_SUMMARIES: MarketSummary[] = [
  {
    neighbourhood_code: "DOWNTOWN",
    neighbourhood_name: "Downtown",
    property_counts: { condo: 14200, townhome: 320, detached: 45 },
    median_values: { condo: 685000, townhome: 1250000, detached: 2800000 },
    yoy_changes: { condo: 3.2, townhome: 1.8, detached: -0.5 },
    interest_rate: 4.79,
  },
  {
    neighbourhood_code: "KITSILANO",
    neighbourhood_name: "Kitsilano",
    property_counts: { condo: 4800, townhome: 1100, detached: 2900 },
    median_values: { condo: 725000, townhome: 1450000, detached: 3200000 },
    yoy_changes: { condo: 4.1, townhome: 2.3, detached: 1.7 },
    interest_rate: 4.79,
  },
  {
    neighbourhood_code: "MOUNT-PLEASANT",
    neighbourhood_name: "Mount Pleasant",
    property_counts: { condo: 5200, townhome: 800, detached: 1600 },
    median_values: { condo: 620000, townhome: 1180000, detached: 1950000 },
    yoy_changes: { condo: 5.8, townhome: 3.1, detached: 2.4 },
    interest_rate: 4.79,
  },
  {
    neighbourhood_code: "KERRISDALE",
    neighbourhood_name: "Kerrisdale",
    property_counts: { condo: 2100, townhome: 450, detached: 3200 },
    median_values: { condo: 780000, townhome: 1600000, detached: 4100000 },
    yoy_changes: { condo: 2.1, townhome: 0.8, detached: -1.2 },
    interest_rate: 4.79,
  },
  {
    neighbourhood_code: "DUNBAR-SOUTHLANDS",
    neighbourhood_name: "Dunbar-Southlands",
    property_counts: { condo: 900, townhome: 280, detached: 4100 },
    median_values: { condo: 690000, townhome: 1350000, detached: 3800000 },
    yoy_changes: { condo: 3.5, townhome: 1.9, detached: 0.3 },
    interest_rate: 4.79,
  },
  {
    neighbourhood_code: "FAIRVIEW",
    neighbourhood_name: "Fairview",
    property_counts: { condo: 7800, townhome: 600, detached: 800 },
    median_values: { condo: 710000, townhome: 1380000, detached: 2600000 },
    yoy_changes: { condo: 4.4, townhome: 2.7, detached: 1.1 },
    interest_rate: 4.79,
  },
  {
    neighbourhood_code: "WEST-END",
    neighbourhood_name: "West End",
    property_counts: { condo: 18500, townhome: 120, detached: 20 },
    median_values: { condo: 595000, townhome: 1100000, detached: 2200000 },
    yoy_changes: { condo: 3.9, townhome: 2.0, detached: 0.0 },
    interest_rate: 4.79,
  },
  {
    neighbourhood_code: "HASTINGS-SUNRISE",
    neighbourhood_name: "Hastings-Sunrise",
    property_counts: { condo: 2400, townhome: 950, detached: 5200 },
    median_values: { condo: 520000, townhome: 980000, detached: 1650000 },
    yoy_changes: { condo: 6.2, townhome: 4.5, detached: 3.8 },
    interest_rate: 4.79,
  },
];

export default function DashboardPage() {
  const [summaries, setSummaries] = useState<MarketSummary[]>([]);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [usingDemo, setUsingDemo] = useState(false);

  useEffect(() => {
    async function load() {
      try {
        const [s, h] = await Promise.all([
          api.getMarketAll(),
          api.getHealth(),
        ]);
        setSummaries(s);
        setHealth(h);
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

  const firstName = "Aparna";

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="flex items-end justify-between">
        <div>
          <h1
            className="text-3xl text-sand-900 tracking-tight"
            style={{ fontFamily: "var(--font-display)" }}
          >
            Good {getGreeting()}, {firstName}
          </h1>
          <p className="text-sand-500 text-sm mt-1">
            Metro Vancouver market overview
          </p>
        </div>
        {usingDemo && (
          <span className="text-xs px-3 py-1 rounded-full bg-amber-50 text-amber-600 border border-amber-200">
            Demo Data — API not connected
          </span>
        )}
      </div>

      {/* KPI Cards */}
      <div className="grid grid-cols-4 gap-4">
        {[
          {
            label: "Total Properties",
            value: loading ? "—" : formatNumber(totalProperties),
            sub: `${summaries.length} neighbourhoods`,
          },
          {
            label: "Avg. Median Value",
            value: loading ? "—" : formatCurrency(avgMedian),
            sub: "Across all types",
          },
          {
            label: "Avg. YoY Change",
            value: loading ? "—" : formatPercent(avgYoy),
            sub: "Year-over-year",
            highlight: true,
          },
          {
            label: "Engine Status",
            value: health?.status === "ok" ? "Online" : health?.status || "—",
            sub: health ? `v${health.version} · ${health.model_count} models` : "Connecting...",
            status: health?.status,
          },
        ].map((kpi, i) => (
          <div
            key={kpi.label}
            className={`card-hairline p-5 animate-fade-in-up opacity-0`}
            style={{ animationDelay: `${i * 0.05}s`, animationFillMode: "forwards" }}
          >
            <div className="text-xs font-medium text-sand-400 uppercase tracking-wider">
              {kpi.label}
            </div>
            <div
              className={`mt-2 text-2xl font-light tracking-tight ${
                kpi.status === "ok"
                  ? "text-emerald-600"
                  : kpi.status === "degraded"
                  ? "text-amber-500"
                  : kpi.highlight && avgYoy > 0
                  ? "text-emerald-600"
                  : kpi.highlight && avgYoy < 0
                  ? "text-rose-500"
                  : "text-sand-900"
              }`}
              style={{ fontFamily: "var(--font-display)" }}
            >
              {kpi.value}
            </div>
            <div className="text-xs text-sand-400 mt-1">{kpi.sub}</div>
          </div>
        ))}
      </div>

      {/* Neighbourhood Grid */}
      <div>
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-semibold text-sand-800">
            Neighbourhoods
          </h2>
          <Link
            href="/market"
            className="text-xs font-medium text-teal-600 hover:text-teal-700 transition"
          >
            View Market Explorer &rarr;
          </Link>
        </div>

        {loading ? (
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4">
            {Array.from({ length: 6 }).map((_, i) => (
              <div key={i} className="card-hairline p-5 animate-pulse">
                <div className="h-4 bg-sand-200 rounded w-1/2 mb-3" />
                <div className="h-8 bg-sand-100 rounded w-2/3 mb-2" />
                <div className="h-3 bg-sand-100 rounded w-1/3" />
              </div>
            ))}
          </div>
        ) : (
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4">
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
      className="card-hairline p-5 hover:border-teal-300 transition-all duration-300 group animate-fade-in-up opacity-0"
      style={{ animationDelay: `${0.2 + index * 0.04}s`, animationFillMode: "forwards" }}
    >
      <div className="flex items-start justify-between">
        <div>
          <h3 className="text-sm font-semibold text-sand-800 group-hover:text-teal-700 transition-colors">
            {summary.neighbourhood_name}
          </h3>
          <p className="text-xs text-sand-400 mt-0.5">
            {formatNumber(totalCount)} properties
          </p>
        </div>
        {primaryYoy != null && (
          <span
            className={`text-xs font-medium px-2 py-0.5 rounded-full ${
              primaryYoy >= 0
                ? "bg-emerald-50 text-emerald-600"
                : "bg-rose-50 text-rose-500"
            }`}
          >
            {formatPercent(primaryYoy)}
          </span>
        )}
      </div>

      <div className="mt-4">
        <div
          className="text-2xl tracking-tight text-sand-900"
          style={{ fontFamily: "var(--font-display)" }}
        >
          {primaryMedian > 0 ? formatCurrency(primaryMedian) : "—"}
        </div>
        <div className="text-[11px] text-sand-400 mt-0.5">
          Median {primaryType?.[0] || "value"}
        </div>
      </div>

      {/* Type breakdown */}
      <div className="mt-3 flex gap-2">
        {Object.entries(summary.property_counts).map(([type, count]) => (
          <span
            key={type}
            className="text-[10px] px-1.5 py-0.5 rounded bg-sand-100 text-sand-500"
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
