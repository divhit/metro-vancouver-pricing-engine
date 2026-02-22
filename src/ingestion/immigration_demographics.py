"""
Immigration and demographic demand-side data ingestion.

Sources:
1. IRCC (Immigration, Refugees and Citizenship Canada) Open Data
   - Monthly permanent resident admissions by CMA
   - Temporary resident permits by province
   URL: https://open.canada.ca/data/en/dataset (IRCC datasets)

2. BC Stats Population Projections
   - Population projections by regional district and municipality
   - Household projections (formation rate drives housing demand)
   URL: https://www2.gov.bc.ca/gov/content/data/statistics/people-population-community/population/population-projections

These demand-side indicators help forecast housing demand pressure
in Metro Vancouver. Immigration accounts for nearly all of Metro
Vancouver's population growth, making IRCC admission data a leading
indicator for housing demand.

Key metrics:
- Monthly PR admissions to Vancouver CMA
- Immigration as % of population growth
- Household formation projections
- Population growth trajectory
"""

import logging
import tempfile
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)

# Open Canada API for IRCC data
OPEN_CANADA_API = "https://open.canada.ca/data/api/3/action/package_show"

# Known IRCC dataset IDs on Open Canada
IRCC_DATASETS = {
    "pr_admissions": "f7e5498e-0ad8-4f28-8e84-7b8c0ef629e3",
    "tr_permits": "360024f2-17e9-4558-bfc1-3616485d65b9",
}

# BC Stats data URLs (CSV downloads)
BC_STATS_BASE = "https://www2.gov.bc.ca/assets/gov/data/statistics/people-population-community/population"

# Vancouver CMA identifier in IRCC data
VANCOUVER_CMA_NAMES = [
    "vancouver",
    "vancouver, british columbia",
    "vancouver (british columbia)",
    "vancouver cma",
]

# Metro Vancouver regional district code
METRO_VANCOUVER_RD = "Greater Vancouver"

# BC regional districts in Metro Vancouver area
MV_REGIONAL_DISTRICTS = [
    "Greater Vancouver",
    "Metro Vancouver",
]


class ImmigrationDemographicsClient:
    """Client for immigration and demographic demand-side indicators.

    Ingests IRCC permanent resident data and BC Stats population/household
    projections relevant to Metro Vancouver housing demand.
    """

    def __init__(self, cache_dir: Optional[str] = None):
        """Initialize the immigration demographics client.

        Args:
            cache_dir: Directory to cache downloaded files.
        """
        self.cache_dir = Path(cache_dir) if cache_dir else Path(tempfile.gettempdir()) / "immig_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers["User-Agent"] = "MetroVancouverPricingEngine/1.0"

    # ----------------------------------------------------------------
    # IRCC Data (Open Canada)
    # ----------------------------------------------------------------

    def _get_ircc_resource_urls(self, dataset_id: str) -> list[dict]:
        """Get resource URLs from an IRCC dataset on Open Canada.

        Args:
            dataset_id: Open Canada dataset UUID.

        Returns:
            List of resource dicts with 'url', 'name', 'format'.
        """
        try:
            response = self.session.get(
                OPEN_CANADA_API, params={"id": dataset_id}, timeout=30
            )
            response.raise_for_status()
            result = response.json()

            resources = []
            for r in result.get("result", {}).get("resources", []):
                fmt = r.get("format", "").upper()
                if fmt in ("CSV", "XLSX", "XLS"):
                    resources.append({
                        "url": r["url"],
                        "name": r.get("name", ""),
                        "format": fmt,
                        "last_modified": r.get("last_modified"),
                    })

            return resources

        except requests.RequestException as e:
            logger.error(f"Failed to query Open Canada API: {e}")
            return []

    def fetch_pr_admissions(
        self, resource_url: Optional[str] = None
    ) -> pd.DataFrame:
        """Fetch IRCC permanent resident admission data.

        Downloads the monthly PR admissions dataset from Open Canada.
        Contains admissions by CMA, immigration category, and month.

        Args:
            resource_url: Direct URL to CSV. Auto-detects if None.

        Returns:
            DataFrame of PR admission records.
        """
        if resource_url is None:
            resources = self._get_ircc_resource_urls(IRCC_DATASETS["pr_admissions"])
            csv_resources = [r for r in resources if r["format"] == "CSV"]
            if not csv_resources:
                logger.error("No CSV resources found for PR admissions")
                return pd.DataFrame()
            resource_url = csv_resources[0]["url"]

        logger.info(f"Downloading PR admissions data from {resource_url}")
        try:
            response = self.session.get(resource_url, timeout=120)
            response.raise_for_status()

            cache_file = self.cache_dir / "pr_admissions.csv"
            cache_file.write_bytes(response.content)

            df = pd.read_csv(cache_file, low_memory=False, encoding="utf-8-sig")
            logger.info(f"Downloaded {len(df):,} PR admission records")
            return df

        except requests.RequestException as e:
            logger.error(f"Failed to download PR admissions: {e}")
            return pd.DataFrame()

    def filter_vancouver_cma(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter IRCC data to Vancouver CMA.

        Args:
            df: IRCC DataFrame with CMA/province columns.

        Returns:
            Filtered DataFrame for Vancouver CMA.
        """
        if df.empty:
            return df

        # Find the CMA column (varies by dataset version)
        cma_col = None
        for c in df.columns:
            c_lower = c.lower()
            if "cma" in c_lower or "census_metropolitan" in c_lower or "destination" in c_lower:
                cma_col = c
                break

        if cma_col is None:
            # Try province-level filter as fallback
            prov_col = None
            for c in df.columns:
                if "province" in c.lower() or "prov" in c.lower():
                    prov_col = c
                    break
            if prov_col:
                mask = df[prov_col].str.lower().str.contains("british columbia", na=False)
                filtered = df[mask].copy()
                logger.info(f"Filtered to {len(filtered):,} BC records (CMA column not found)")
                return filtered

            logger.warning("Could not find CMA or province column")
            return df

        # Match on known Vancouver CMA name variants
        mask = df[cma_col].str.lower().str.strip().isin(VANCOUVER_CMA_NAMES)
        filtered = df[mask].copy()
        logger.info(f"Filtered to {len(filtered):,} Vancouver CMA records")
        return filtered

    def compute_monthly_admissions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate PR admissions by month.

        Args:
            df: Filtered IRCC DataFrame (Vancouver CMA).

        Returns:
            Monthly admissions time series.
        """
        if df.empty:
            return pd.DataFrame()

        # Find date/period columns
        year_col = None
        month_col = None
        for c in df.columns:
            c_lower = c.lower()
            if "year" in c_lower:
                year_col = c
            if "month" in c_lower:
                month_col = c

        if year_col is None:
            logger.warning("No year column found in IRCC data")
            return pd.DataFrame()

        # Find count/value column
        count_col = None
        for c in df.columns:
            c_lower = c.lower()
            if any(k in c_lower for k in ["count", "value", "persons", "number", "admissions"]):
                count_col = c
                break

        if count_col is None:
            # Assume each row is one admission
            df["_count"] = 1
            count_col = "_count"

        group_cols = [year_col]
        if month_col:
            group_cols.append(month_col)

        df[count_col] = pd.to_numeric(df[count_col], errors="coerce").fillna(0)

        monthly = df.groupby(group_cols)[count_col].sum().reset_index()
        monthly = monthly.rename(columns={count_col: "admissions"})
        monthly = monthly.sort_values(group_cols)

        # Create date column if we have year + month
        if month_col and year_col:
            monthly["date"] = pd.to_datetime(
                monthly[year_col].astype(str) + "-" + monthly[month_col].astype(str).str.zfill(2) + "-01",
                errors="coerce",
            )

        logger.info(f"Computed monthly admissions: {len(monthly)} periods")
        return monthly

    def compute_admission_trends(self, monthly_df: pd.DataFrame) -> dict:
        """Compute admission trend indicators.

        Args:
            monthly_df: Monthly admissions DataFrame.

        Returns:
            Dict with trend metrics.
        """
        if monthly_df.empty or "admissions" not in monthly_df.columns:
            return {}

        admissions = monthly_df["admissions"]

        trends = {
            "total_admissions": int(admissions.sum()),
            "avg_monthly_admissions": round(admissions.mean(), 0),
            "latest_monthly": int(admissions.iloc[-1]) if len(admissions) > 0 else None,
            "max_monthly": int(admissions.max()),
        }

        # Year-over-year change (if enough data)
        if len(admissions) >= 24:
            recent_12 = admissions.iloc[-12:].sum()
            prior_12 = admissions.iloc[-24:-12].sum()
            if prior_12 > 0:
                trends["yoy_change_pct"] = round(
                    (recent_12 - prior_12) / prior_12 * 100, 1
                )

        # Rolling 3-month average
        if len(admissions) >= 3:
            trends["rolling_3m_avg"] = round(admissions.iloc[-3:].mean(), 0)

        return trends

    # ----------------------------------------------------------------
    # BC Stats Population Projections
    # ----------------------------------------------------------------

    def fetch_bc_population_projections(
        self, file_url: Optional[str] = None
    ) -> pd.DataFrame:
        """Fetch BC Stats population projections.

        BC Stats publishes PEOPLE (Population Extrapolation for
        Organizational Planning with Less Error) projections by
        regional district.

        Args:
            file_url: Direct URL to projections CSV/XLSX. Uses known
                     URL if None.

        Returns:
            DataFrame of population projections.
        """
        if file_url is None:
            # BC Stats publishes these as Excel files
            file_url = (
                f"{BC_STATS_BASE}/pop_municipal_estimates.csv"
            )

        logger.info(f"Downloading BC population projections from {file_url}")
        try:
            response = self.session.get(file_url, timeout=120)
            response.raise_for_status()

            cache_file = self.cache_dir / "bc_pop_projections.csv"
            cache_file.write_bytes(response.content)

            if file_url.endswith((".xlsx", ".xls")):
                df = pd.read_excel(cache_file, engine="openpyxl")
            else:
                df = pd.read_csv(cache_file, low_memory=False, encoding="latin-1")

            logger.info(f"Downloaded {len(df):,} population projection records")
            return df

        except requests.RequestException as e:
            logger.error(f"Failed to download population projections: {e}")
            return pd.DataFrame()

    def filter_metro_vancouver_projections(
        self, df: pd.DataFrame
    ) -> pd.DataFrame:
        """Filter population projections to Metro Vancouver.

        Args:
            df: BC Stats population DataFrame.

        Returns:
            Filtered to Metro Vancouver regional district.
        """
        if df.empty:
            return df

        df.columns = [c.strip() for c in df.columns]

        # Find the region/municipality column
        region_col = None
        for c in df.columns:
            c_lower = c.lower()
            if any(k in c_lower for k in ["region", "municipality", "area", "name"]):
                region_col = c
                break

        if region_col is None:
            logger.warning("Could not find region column in projections data")
            return df

        # Match Metro Vancouver regional districts and municipalities
        mv_names = [n.lower() for n in MV_REGIONAL_DISTRICTS]
        from .bc_property_transfer_tax import METRO_VANCOUVER_MUNICIPALITIES
        mv_munis = [m.lower() for m in METRO_VANCOUVER_MUNICIPALITIES]
        all_names = set(mv_names + mv_munis)

        mask = df[region_col].str.lower().str.strip().isin(all_names)
        filtered = df[mask].copy()
        logger.info(f"Filtered to {len(filtered):,} Metro Vancouver projection records")
        return filtered

    # ----------------------------------------------------------------
    # Demand indicators
    # ----------------------------------------------------------------

    def compute_demand_indicators(
        self,
        pr_monthly: Optional[pd.DataFrame] = None,
        pop_projections: Optional[pd.DataFrame] = None,
    ) -> dict:
        """Compute composite demand-side indicators.

        Combines immigration admissions and population projection data
        into a set of demand indicators for the pricing model.

        Args:
            pr_monthly: Monthly PR admissions (Vancouver CMA).
            pop_projections: BC Stats population projections.

        Returns:
            Dict of demand indicators.
        """
        indicators = {}

        # Immigration demand pressure
        if pr_monthly is not None and not pr_monthly.empty:
            trends = self.compute_admission_trends(pr_monthly)
            indicators.update({
                f"immigration_{k}": v for k, v in trends.items()
            })

            # Annualized admission rate (proxy for housing units needed)
            avg_monthly = trends.get("avg_monthly_admissions", 0)
            if avg_monthly:
                # ~2.5 persons per household average
                indicators["estimated_annual_housing_units_from_immigration"] = round(
                    avg_monthly * 12 / 2.5, 0
                )

        # Population growth pressure
        if pop_projections is not None and not pop_projections.empty:
            # Find year columns (typically numeric column headers)
            year_cols = [
                c for c in pop_projections.columns
                if str(c).isdigit() and 2020 <= int(c) <= 2050
            ]
            if year_cols:
                year_cols_sorted = sorted(year_cols, key=int)
                latest = int(year_cols_sorted[-1])
                earliest = int(year_cols_sorted[0])

                total_latest = pd.to_numeric(
                    pop_projections[str(latest)], errors="coerce"
                ).sum()
                total_earliest = pd.to_numeric(
                    pop_projections[str(earliest)], errors="coerce"
                ).sum()

                if total_earliest > 0:
                    indicators["projected_population_growth_pct"] = round(
                        (total_latest - total_earliest) / total_earliest * 100, 1
                    )
                    indicators["projected_population_latest"] = total_latest
                    indicators["projected_annual_growth_rate"] = round(
                        ((total_latest / total_earliest) ** (1 / max(latest - earliest, 1)) - 1) * 100,
                        2,
                    )

        return indicators
