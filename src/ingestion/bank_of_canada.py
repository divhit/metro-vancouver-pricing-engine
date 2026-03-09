"""
Bank of Canada Valet API ingestion.

Free, no authentication required.
Provides: policy rate, mortgage rates, bond yields, inflation.
"""

import logging
from datetime import date, timedelta
from typing import Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)

VALET_BASE = "https://www.bankofcanada.ca/valet"

# Key series for real estate pricing
SERIES = {
    "policy_rate": "V39079",
    "prime_rate": "V80691311",
    "mortgage_1yr": "V80691333",       # 1-year conventional mortgage
    "mortgage_3yr": "V80691334",       # 3-year conventional mortgage
    "mortgage_5yr_fixed": "V80691335", # 5-year conventional mortgage
    "cpi_all": "V41690973",
    "cpi_shelter": "V41691231",
    "bond_5yr": "V39055",
    "bond_10yr": "V39062",
}


class BankOfCanadaClient:
    """Client for the Bank of Canada Valet API."""

    def __init__(self):
        self.session = requests.Session()

    def get_series(
        self,
        series_name: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> pd.DataFrame:
        """Fetch a single data series.

        Args:
            series_name: Valet series ID (e.g., 'V80691335')
            start_date: Start of date range
            end_date: End of date range

        Returns:
            DataFrame with 'date' and 'value' columns
        """
        url = f"{VALET_BASE}/observations/{series_name}/json"
        params = {}
        if start_date:
            params["start_date"] = start_date.isoformat()
        if end_date:
            params["end_date"] = end_date.isoformat()

        response = self.session.get(url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()

        observations = data.get("observations", [])
        if not observations:
            return pd.DataFrame(columns=["date", "value"])

        records = []
        for obs in observations:
            records.append(
                {
                    "date": obs["d"],
                    "value": float(obs[series_name]["v"])
                    if obs[series_name]["v"] not in (None, "")
                    else None,
                }
            )

        df = pd.DataFrame(records)
        df["date"] = pd.to_datetime(df["date"])
        return df

    def get_mortgage_rates(
        self,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> pd.DataFrame:
        """Fetch all mortgage-related rates as a single DataFrame.

        Returns DataFrame with columns: date, policy_rate, prime_rate,
        mortgage_1yr, mortgage_3yr, mortgage_5yr_fixed.
        """
        mortgage_series = {
            k: v
            for k, v in SERIES.items()
            if k in ("policy_rate", "prime_rate", "mortgage_1yr", "mortgage_3yr", "mortgage_5yr_fixed")
        }

        dfs = {}
        for name, series_id in mortgage_series.items():
            df = self.get_series(series_id, start_date, end_date)
            df = df.rename(columns={"value": name})
            dfs[name] = df.set_index("date")

        # Merge all series on date
        result = pd.DataFrame()
        for name, df in dfs.items():
            if result.empty:
                result = df
            else:
                result = result.join(df, how="outer")

        return result.reset_index().sort_values("date")

    def get_current_rates(self) -> dict:
        """Get the most recent available rates."""
        end = date.today()
        start = end - timedelta(days=30)  # Look back 30 days for latest

        rates = {}
        for name, series_id in SERIES.items():
            df = self.get_series(series_id, start, end)
            if not df.empty:
                rates[name] = df.iloc[-1]["value"]
            else:
                rates[name] = None

        return rates

    def compute_stress_test_rate(self, contract_rate: float) -> float:
        """Compute the OSFI B-20 stress test qualifying rate.

        The qualifying rate is the higher of:
        - Contract rate + 2%
        - 5.25% floor
        """
        return max(contract_rate + 2.0, 5.25)

    def compute_max_mortgage(
        self,
        gross_income: float,
        rate: float,
        amortization_years: int = 25,
        gds_limit: float = 0.39,
        property_tax_annual: float = 5_000,
        heating_monthly: float = 100,
    ) -> float:
        """Compute maximum mortgage amount given income and rates.

        Uses GDS ratio calculation with OSFI stress test.

        Args:
            gross_income: Annual gross household income
            rate: Annual interest rate (use stress test rate)
            amortization_years: Amortization period
            gds_limit: Maximum GDS ratio (0.39 for insured)
            property_tax_annual: Estimated annual property tax
            heating_monthly: Monthly heating cost estimate

        Returns:
            Maximum mortgage principal
        """
        monthly_income = gross_income / 12
        max_housing_cost = monthly_income * gds_limit

        # Subtract non-mortgage housing costs
        property_tax_monthly = property_tax_annual / 12
        available_for_mortgage = (
            max_housing_cost - property_tax_monthly - heating_monthly
        )

        if available_for_mortgage <= 0:
            return 0.0

        # Monthly mortgage payment to principal conversion
        monthly_rate = rate / 100 / 12
        n_payments = amortization_years * 12

        if monthly_rate == 0:
            return available_for_mortgage * n_payments

        # PV of annuity formula
        pv_factor = (1 - (1 + monthly_rate) ** (-n_payments)) / monthly_rate
        max_principal = available_for_mortgage * pv_factor

        return round(max_principal, 2)
