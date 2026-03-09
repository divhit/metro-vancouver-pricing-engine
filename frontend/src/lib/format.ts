export function formatCurrency(value: number): string {
  if (value >= 1_000_000) {
    return `$${(value / 1_000_000).toFixed(2)}M`;
  }
  if (value >= 1_000) {
    return `$${(value / 1_000).toFixed(0)}K`;
  }
  return `$${value.toFixed(0)}`;
}

export function formatCurrencyFull(value: number): string {
  return new Intl.NumberFormat("en-CA", {
    style: "currency",
    currency: "CAD",
    maximumFractionDigits: 0,
  }).format(value);
}

export function formatPercent(value: number | null | undefined): string {
  if (value == null) return "N/A";
  const sign = value >= 0 ? "+" : "";
  return `${sign}${value.toFixed(1)}%`;
}

export function formatDistance(meters: number): string {
  if (meters <= 0) return "Same area";
  if (meters >= 1000) return `${(meters / 1000).toFixed(1)} km`;
  return `${Math.round(meters)} m`;
}

export function formatNumber(n: number): string {
  return new Intl.NumberFormat("en-CA").format(n);
}

export function gradeColor(grade: string): string {
  switch (grade) {
    case "A":
      return "text-emerald-600";
    case "B":
      return "text-amber-500";
    case "C":
      return "text-rose-500";
    default:
      return "text-stone-500";
  }
}

export function gradeBg(grade: string): string {
  switch (grade) {
    case "A":
      return "bg-emerald-50 border-emerald-200";
    case "B":
      return "bg-amber-50 border-amber-200";
    case "C":
      return "bg-rose-50 border-rose-200";
    default:
      return "bg-stone-50 border-stone-200";
  }
}

export function severityColor(severity: string): string {
  switch (severity) {
    case "high":
      return "text-rose-600 bg-rose-50 border-rose-200";
    case "medium":
      return "text-amber-600 bg-amber-50 border-amber-200";
    case "low":
      return "text-sky-600 bg-sky-50 border-sky-200";
    default:
      return "text-stone-500 bg-stone-50 border-stone-200";
  }
}
