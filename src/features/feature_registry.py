"""
Complete feature registry for the pricing engine.

Defines all ~150+ features organized by category, with metadata
for each: name, data type, source, property types applicable,
and importance tier.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class PropertyType(Enum):
    CONDO = "condo"
    TOWNHOME = "townhome"
    DETACHED = "detached"
    DEVELOPMENT_LAND = "development_land"
    ALL = "all"


class FeatureTier(Enum):
    """Feature importance tier for phased implementation."""
    PHASE_1 = 1  # Core structural + location (Months 1-3)
    PHASE_2 = 2  # Enriched features (Months 4-6)
    PHASE_3 = 3  # Supply pipeline + market context (Months 7-9)
    PHASE_4 = 4  # Advanced / experimental (Months 10+)


@dataclass
class FeatureDefinition:
    name: str
    description: str
    dtype: str  # float, int, bool, categorical
    source: str  # Data source key from data_sources.yaml
    property_types: list[PropertyType] = field(default_factory=lambda: [PropertyType.ALL])
    tier: FeatureTier = FeatureTier.PHASE_1
    nullable: bool = True
    unit: Optional[str] = None


# ============================================================
# STRUCTURAL FEATURES (25 variables)
# ============================================================

STRUCTURAL_FEATURES = [
    FeatureDefinition("living_area_sqft", "Total living area in square feet", "float", "mls", unit="sqft"),
    FeatureDefinition("lot_size_sqft", "Total lot size in square feet", "float", "bc_assessment", unit="sqft",
                       property_types=[PropertyType.DETACHED, PropertyType.TOWNHOME, PropertyType.DEVELOPMENT_LAND]),
    FeatureDefinition("bedrooms", "Number of bedrooms", "int", "mls"),
    FeatureDefinition("bathrooms_full", "Number of full bathrooms", "int", "mls"),
    FeatureDefinition("bathrooms_half", "Number of half bathrooms", "int", "mls"),
    FeatureDefinition("year_built", "Year of construction", "int", "mls"),
    FeatureDefinition("effective_age", "Effective age accounting for renovations", "float", "bc_assessment", unit="years"),
    FeatureDefinition("construction_type", "Wood-frame / concrete / steel", "categorical", "mls"),
    FeatureDefinition("num_stories", "Number of stories", "int", "mls"),
    FeatureDefinition("parking_stalls", "Number of parking stalls", "int", "mls"),
    FeatureDefinition("parking_type", "Underground / surface / garage / carport / none", "categorical", "mls"),
    FeatureDefinition("storage_lockers", "Number of storage lockers", "int", "mls",
                       property_types=[PropertyType.CONDO, PropertyType.TOWNHOME]),
    FeatureDefinition("floor_level", "Floor level of unit", "int", "mls",
                       property_types=[PropertyType.CONDO]),
    FeatureDefinition("unit_exposure", "N/S/E/W/NE/NW/SE/SW facing", "categorical", "mls",
                       property_types=[PropertyType.CONDO]),
    FeatureDefinition("layout_efficiency", "Sqft per bedroom ratio", "float", "derived", unit="sqft/bedroom"),
    FeatureDefinition("building_total_units", "Total units in building", "int", "mls",
                       property_types=[PropertyType.CONDO, PropertyType.TOWNHOME]),
    FeatureDefinition("suite_adu_present", "Secondary suite or ADU present", "bool", "mls",
                       property_types=[PropertyType.DETACHED]),
    FeatureDefinition("basement_type", "Full / partial / crawl / slab / none", "categorical", "mls",
                       property_types=[PropertyType.DETACHED, PropertyType.TOWNHOME]),
    FeatureDefinition("basement_finished", "Finished / unfinished / partial", "categorical", "mls",
                       property_types=[PropertyType.DETACHED, PropertyType.TOWNHOME]),
    FeatureDefinition("heating_system", "Forced air / radiant / baseboard / heat pump", "categorical", "mls"),
    FeatureDefinition("cooling_system", "Central AC / mini-split / none", "categorical", "mls"),
    FeatureDefinition("lot_frontage_ft", "Lot frontage in feet", "float", "bc_assessment", unit="ft",
                       property_types=[PropertyType.DETACHED, PropertyType.DEVELOPMENT_LAND]),
    FeatureDefinition("lot_depth_ft", "Lot depth in feet", "float", "bc_assessment", unit="ft",
                       property_types=[PropertyType.DETACHED, PropertyType.DEVELOPMENT_LAND]),
    FeatureDefinition("is_end_unit", "End unit flag (townhomes)", "bool", "mls",
                       property_types=[PropertyType.TOWNHOME]),
    FeatureDefinition("has_private_outdoor", "Has private yard/patio/rooftop", "bool", "mls"),
]

# ============================================================
# LOCATION FEATURES (20 variables)
# ============================================================

LOCATION_FEATURES = [
    FeatureDefinition("latitude", "Property latitude", "float", "mls"),
    FeatureDefinition("longitude", "Property longitude", "float", "mls"),
    FeatureDefinition("neighborhood_code", "GVR sub-area code", "categorical", "mls"),
    FeatureDefinition("municipality", "Municipality name", "categorical", "bc_assessment"),
    FeatureDefinition("dist_skytrain_m", "Distance to nearest SkyTrain station (meters)", "float", "translink_gtfs", unit="m"),
    FeatureDefinition("dist_future_skytrain_m", "Distance to nearest planned/under-construction station", "float", "translink_gtfs", unit="m", tier=FeatureTier.PHASE_3),
    FeatureDefinition("dist_waterfront_m", "Distance to nearest waterfront", "float", "openstreetmap", unit="m"),
    FeatureDefinition("dist_cbd_m", "Distance to downtown Vancouver CBD", "float", "derived", unit="m"),
    FeatureDefinition("walk_score", "Walk Score (0-100)", "int", "walk_score", tier=FeatureTier.PHASE_2),
    FeatureDefinition("transit_score", "Transit Score (0-100)", "int", "walk_score", tier=FeatureTier.PHASE_2),
    FeatureDefinition("bike_score", "Bike Score (0-100)", "int", "walk_score", tier=FeatureTier.PHASE_2),
    FeatureDefinition("school_catchment_id", "Secondary school catchment identifier", "categorical", "vancouver_open_data"),
    FeatureDefinition("school_ranking_score", "Fraser Institute school ranking (0-10)", "float", "fraser_institute", tier=FeatureTier.PHASE_2),
    FeatureDefinition("dist_nearest_park_m", "Distance to nearest park", "float", "openstreetmap", unit="m"),
    FeatureDefinition("park_area_500m_sqm", "Total park area within 500m radius", "float", "openstreetmap", unit="sqm", tier=FeatureTier.PHASE_2),
    FeatureDefinition("dist_commercial_node_m", "Distance to nearest commercial/retail cluster", "float", "openstreetmap", unit="m"),
    FeatureDefinition("census_tract_median_income", "Median household income of census tract", "float", "statistics_canada", tier=FeatureTier.PHASE_2),
    FeatureDefinition("census_tract_pop_density", "Population density of census tract", "float", "statistics_canada", tier=FeatureTier.PHASE_2, unit="persons/sqkm"),
    FeatureDefinition("crime_density_500m", "Crime incidents within 500m (trailing 12 months)", "float", "vpd_crime", tier=FeatureTier.PHASE_2),
    FeatureDefinition("amenity_density_500m", "Count of amenities within 500m (restaurants, shops, etc.)", "int", "openstreetmap", tier=FeatureTier.PHASE_2),
    FeatureDefinition("child_care_within_1km", "Licensed child care facilities within 1km", "int", "vancouver_open_data",
                       tier=FeatureTier.PHASE_2),
    FeatureDefinition("school_fsa_score_nearest", "FSA numeracy/literacy score for nearest school", "float", "bc_education",
                       tier=FeatureTier.PHASE_1),
    FeatureDefinition("post_secondary_dist_km", "Distance to nearest post-secondary institution", "float", "openstreetmap",
                       tier=FeatureTier.PHASE_3, unit="km"),
]

# ============================================================
# STRATA / BUILDING FEATURES (20 variables, condos/townhomes)
# ============================================================

STRATA_FEATURES = [
    FeatureDefinition("strata_fee_monthly", "Monthly strata fee", "float", "mls", unit="$",
                       property_types=[PropertyType.CONDO, PropertyType.TOWNHOME]),
    FeatureDefinition("strata_fee_per_sqft", "Monthly strata fee per sqft", "float", "derived", unit="$/sqft",
                       property_types=[PropertyType.CONDO, PropertyType.TOWNHOME]),
    FeatureDefinition("strata_fee_to_price_ratio", "Strata fee as % of property value (annualized)", "float", "derived",
                       property_types=[PropertyType.CONDO, PropertyType.TOWNHOME]),
    FeatureDefinition("contingency_reserve_pct", "Contingency reserve as % of annual operating", "float", "form_b",
                       property_types=[PropertyType.CONDO, PropertyType.TOWNHOME], tier=FeatureTier.PHASE_2),
    FeatureDefinition("depreciation_report_age_months", "Months since last depreciation report", "int", "form_b",
                       property_types=[PropertyType.CONDO, PropertyType.TOWNHOME], tier=FeatureTier.PHASE_2),
    FeatureDefinition("special_assessment_count_5yr", "Special assessments in past 5 years", "int", "form_b",
                       property_types=[PropertyType.CONDO, PropertyType.TOWNHOME], tier=FeatureTier.PHASE_2),
    FeatureDefinition("special_assessment_total_5yr", "Total special assessment $ in past 5 years", "float", "form_b",
                       property_types=[PropertyType.CONDO, PropertyType.TOWNHOME], tier=FeatureTier.PHASE_2, unit="$"),
    FeatureDefinition("insurance_deductible", "Building insurance deductible amount", "float", "form_b",
                       property_types=[PropertyType.CONDO, PropertyType.TOWNHOME], tier=FeatureTier.PHASE_2, unit="$"),
    FeatureDefinition("insurance_premium_per_unit", "Annual insurance premium per unit", "float", "form_b",
                       property_types=[PropertyType.CONDO, PropertyType.TOWNHOME], tier=FeatureTier.PHASE_3, unit="$/year"),
    FeatureDefinition("building_age_bucket", "New(0-5) / Recent(6-15) / Mature(16-30) / Older(31+)", "categorical", "derived",
                       property_types=[PropertyType.CONDO, PropertyType.TOWNHOME]),
    FeatureDefinition("rainscreen_status", "Yes / No / Partial / N/A", "categorical", "mls",
                       property_types=[PropertyType.CONDO, PropertyType.TOWNHOME], tier=FeatureTier.PHASE_2),
    FeatureDefinition("developer_brand_tier", "Premium / Standard / Budget / Unknown", "categorical", "derived",
                       property_types=[PropertyType.CONDO, PropertyType.TOWNHOME], tier=FeatureTier.PHASE_2),
    FeatureDefinition("pet_restriction", "Pets allowed / restricted / no pets", "categorical", "mls",
                       property_types=[PropertyType.CONDO, PropertyType.TOWNHOME]),
    FeatureDefinition("age_restriction", "None / 19+ / 55+", "categorical", "mls",
                       property_types=[PropertyType.CONDO, PropertyType.TOWNHOME]),
    FeatureDefinition("has_concierge", "Building has concierge", "bool", "mls",
                       property_types=[PropertyType.CONDO]),
    FeatureDefinition("has_pool", "Building has pool", "bool", "mls",
                       property_types=[PropertyType.CONDO]),
    FeatureDefinition("has_gym", "Building has gym/fitness", "bool", "mls",
                       property_types=[PropertyType.CONDO]),
    FeatureDefinition("ownership_structure", "Conventional strata / Bare land strata / Freehold", "categorical", "mls",
                       property_types=[PropertyType.TOWNHOME]),
    FeatureDefinition("crt_dispute_count_3yr", "CRT strata disputes in past 3 years", "int", "crt_strata",
                       property_types=[PropertyType.CONDO, PropertyType.TOWNHOME], tier=FeatureTier.PHASE_3),
    FeatureDefinition("owner_occupied_ratio", "% of units owner-occupied vs rented", "float", "form_b",
                       property_types=[PropertyType.CONDO, PropertyType.TOWNHOME], tier=FeatureTier.PHASE_3),
]

# ============================================================
# LAND AND DEVELOPMENT FEATURES (15 variables)
# ============================================================

LAND_FEATURES = [
    FeatureDefinition("zoning_code", "Current zoning designation", "categorical", "vancouver_open_data",
                       property_types=[PropertyType.DETACHED, PropertyType.DEVELOPMENT_LAND]),
    FeatureDefinition("max_fsr", "Maximum permitted FSR", "float", "vancouver_open_data",
                       property_types=[PropertyType.DETACHED, PropertyType.DEVELOPMENT_LAND]),
    FeatureDefinition("max_height_m", "Maximum permitted building height", "float", "vancouver_open_data",
                       property_types=[PropertyType.DETACHED, PropertyType.DEVELOPMENT_LAND], unit="m"),
    FeatureDefinition("lot_shape_regularity", "Regularity score (1=perfect rectangle)", "float", "derived",
                       property_types=[PropertyType.DETACHED, PropertyType.DEVELOPMENT_LAND]),
    FeatureDefinition("land_to_total_ratio", "Land value / total assessed value", "float", "bc_assessment",
                       property_types=[PropertyType.DETACHED, PropertyType.DEVELOPMENT_LAND]),
    FeatureDefinition("ocp_future_land_use", "OCP future land use designation", "categorical", "vancouver_open_data",
                       property_types=[PropertyType.ALL], tier=FeatureTier.PHASE_2),
    FeatureDefinition("is_tod_area", "Within Transit-Oriented Development area (Bill 47)", "bool", "derived",
                       property_types=[PropertyType.ALL], tier=FeatureTier.PHASE_2),
    FeatureDefinition("tod_tier", "TOD tier: 200m / 400m / 800m / none", "categorical", "derived",
                       property_types=[PropertyType.ALL], tier=FeatureTier.PHASE_2),
    FeatureDefinition("bill44_max_units", "Max units permitted under Bill 44", "int", "derived",
                       property_types=[PropertyType.DETACHED], tier=FeatureTier.PHASE_2),
    FeatureDefinition("assembly_potential_flag", "Property is in active assembly corridor", "bool", "derived",
                       property_types=[PropertyType.DETACHED, PropertyType.DEVELOPMENT_LAND], tier=FeatureTier.PHASE_3),
    FeatureDefinition("heritage_designation", "None / Registered / Designated / HCA", "categorical", "vancouver_open_data",
                       property_types=[PropertyType.DETACHED], tier=FeatureTier.PHASE_2),
    FeatureDefinition("alr_status", "In Agricultural Land Reserve", "bool", "bc_data_catalogue",
                       property_types=[PropertyType.DETACHED, PropertyType.DEVELOPMENT_LAND]),
    FeatureDefinition("floodplain_designation", "In designated floodplain", "bool", "vancouver_open_data",
                       property_types=[PropertyType.ALL]),
    FeatureDefinition("is_corner_lot", "Corner lot flag", "bool", "derived",
                       property_types=[PropertyType.DETACHED, PropertyType.DEVELOPMENT_LAND]),
    FeatureDefinition("dcl_rate_per_sqft", "Applicable DCL rate", "float", "vancouver_open_data",
                       property_types=[PropertyType.DEVELOPMENT_LAND], tier=FeatureTier.PHASE_3, unit="$/sqft"),
    FeatureDefinition("alr_flag", "Agricultural Land Reserve status", "bool", "bc_data_catalogue",
                       property_types=[PropertyType.DETACHED, PropertyType.DEVELOPMENT_LAND], tier=FeatureTier.PHASE_1),
    FeatureDefinition("contaminated_site_within_500m", "Proximity to BC Site Registry contaminated site", "bool", "bc_data_catalogue",
                       tier=FeatureTier.PHASE_2),
]

# ============================================================
# VIEW AND ENVIRONMENTAL FEATURES (12 variables)
# ============================================================

VIEW_ENVIRONMENTAL_FEATURES = [
    FeatureDefinition("view_type", "Water / Mountain / City / Park / None", "categorical", "mls"),
    FeatureDefinition("view_quality", "Unobstructed / Partial / Peek-a-boo / None", "categorical", "mls"),
    FeatureDefinition("dist_industrial_m", "Distance to nearest industrial zone", "float", "vancouver_open_data", unit="m"),
    FeatureDefinition("dist_major_road_m", "Distance to nearest major road (noise proxy)", "float", "openstreetmap", unit="m"),
    FeatureDefinition("contamination_risk_flag", "Near environmental remediation site", "bool", "bc_data_catalogue", tier=FeatureTier.PHASE_2),
    FeatureDefinition("liquefaction_susceptibility", "Seismic liquefaction risk score", "float", "seismic_microzonation", tier=FeatureTier.PHASE_3),
    FeatureDefinition("sea_level_rise_exposure", "In projected sea level rise zone (2050/2100)", "categorical", "vancouver_open_data", tier=FeatureTier.PHASE_3),
    FeatureDefinition("air_quality_index", "Average air quality index (nearest station)", "float", "environment_canada", tier=FeatureTier.PHASE_3),
    FeatureDefinition("yvr_flight_path", "Under YVR flight path noise zone", "bool", "derived", tier=FeatureTier.PHASE_3),
    FeatureDefinition("ndvi_500m", "Vegetation index within 500m radius", "float", "google_earth_engine", tier=FeatureTier.PHASE_4),
    FeatureDefinition("photo_condition_score", "AI condition score from listing photos", "float", "helldata_qualityscore", tier=FeatureTier.PHASE_2),
    FeatureDefinition("photo_kitchen_score", "AI kitchen quality sub-score", "float", "helldata_qualityscore", tier=FeatureTier.PHASE_2),
    FeatureDefinition("wildfire_wui_risk_class", "Wildland-Urban Interface risk class (1-5)", "int", "bc_data_catalogue",
                       property_types=[PropertyType.DETACHED], tier=FeatureTier.PHASE_2),
    FeatureDefinition("tree_canopy_pct_500m", "Tree canopy coverage within 500m radius", "float", "vancouver_open_data",
                       tier=FeatureTier.PHASE_3, unit="%"),
    FeatureDefinition("yvr_noise_nef", "YVR noise exposure forecast contour value", "float", "yvr_noise_contours",
                       tier=FeatureTier.PHASE_2),
    FeatureDefinition("air_quality_aqhi_avg", "Average AQHI for nearest sub-region", "float", "environment_canada",
                       tier=FeatureTier.PHASE_3),
    FeatureDefinition("broadband_speed_mbps", "Download speed at 250m road segment", "float", "ised_broadband",
                       tier=FeatureTier.PHASE_3, unit="Mbps"),
    FeatureDefinition("climate_risk_composite", "Composite climate risk (flood + fire + seismic + sea level)", "float", "derived",
                       tier=FeatureTier.PHASE_3),
]

# ============================================================
# MARKET CONTEXT FEATURES (18 variables)
# ============================================================

MARKET_CONTEXT_FEATURES = [
    FeatureDefinition("months_since_sale", "Months between sale and valuation date", "float", "derived", unit="months"),
    FeatureDefinition("sal_ratio_subarea_3m", "Sales-to-active ratio (sub-area, trailing 3 months)", "float", "mls"),
    FeatureDefinition("benchmark_index", "MLS HPI benchmark index (sub-area, property type)", "float", "crea_hpi", tier=FeatureTier.PHASE_2),
    FeatureDefinition("interest_rate_at_sale", "5-yr fixed mortgage rate at time of sale", "float", "bank_of_canada"),
    FeatureDefinition("stress_test_rate_at_sale", "OSFI qualifying rate at time of sale", "float", "derived"),
    FeatureDefinition("seasonal_month", "Month of sale (1-12)", "int", "derived"),
    FeatureDefinition("seasonal_quarter", "Quarter of sale (Q1-Q4)", "categorical", "derived"),
    FeatureDefinition("supply_pipeline_1km", "Units under construction + approved within 1km", "int", "cmhc", tier=FeatureTier.PHASE_3),
    FeatureDefinition("supply_pipeline_type", "Dominant pipeline type: rental / strata / mixed", "categorical", "derived", tier=FeatureTier.PHASE_3),
    FeatureDefinition("avg_dom_subarea", "Average days on market for sub-area (trailing 3 months)", "float", "mls"),
    FeatureDefinition("price_to_rent_ratio", "Price-to-annual-rent ratio for sub-area", "float", "derived", tier=FeatureTier.PHASE_3),
    FeatureDefinition("list_to_sale_ratio", "Average list-to-sale ratio for sub-area (trailing)", "float", "mls"),
    FeatureDefinition("vacancy_rate_zone", "CMHC rental vacancy rate for nearest zone", "float", "cmhc", tier=FeatureTier.PHASE_3),
    FeatureDefinition("presale_inventory_1km", "Unsold presale units within 1km", "int", "buildify", tier=FeatureTier.PHASE_4),
    FeatureDefinition("immigration_trend_12m", "12-month trailing immigration flow to CMA", "float", "ircc", tier=FeatureTier.PHASE_3),
    FeatureDefinition("employment_growth_ct", "Employment growth in census tract (trailing 12m)", "float", "statistics_canada", tier=FeatureTier.PHASE_3),
    FeatureDefinition("foreign_buyer_share", "Foreign buyer % of transactions (municipality)", "float", "bc_ptt", tier=FeatureTier.PHASE_3),
    FeatureDefinition("mortgage_renewal_exposure", "% of area mortgages due for renewal in 12 months", "float", "derived", tier=FeatureTier.PHASE_4),
    FeatureDefinition("foreign_buyer_pct_municipality", "Foreign buyer share from BC PTT data", "float", "bc_ptt",
                       tier=FeatureTier.PHASE_1),
    FeatureDefinition("ptt_transaction_volume_30d", "Property transfer volume trailing 30 days", "int", "bc_ptt",
                       tier=FeatureTier.PHASE_1),
    FeatureDefinition("immigration_pr_monthly_cma", "Monthly permanent residents to Vancouver CMA", "int", "ircc",
                       tier=FeatureTier.PHASE_2),
    FeatureDefinition("google_trends_real_estate_bc", "Google Trends index for BC real estate", "float", "google_trends",
                       tier=FeatureTier.PHASE_3),
    FeatureDefinition("empty_homes_tax_vacancy_rate", "EHT neighbourhood vacancy rate (Vancouver only)", "float", "vancouver_open_data",
                       tier=FeatureTier.PHASE_2),
    FeatureDefinition("svt_vacancy_rate_municipality", "SVT vacancy rate by municipality", "float", "bc_data_catalogue",
                       tier=FeatureTier.PHASE_2),
    FeatureDefinition("affordability_mppi", "National Bank MPPI (mortgage payment % of income)", "float", "national_bank",
                       tier=FeatureTier.PHASE_2),
    FeatureDefinition("str_density_500m", "Short-term rental listings within 500m", "int", "vancouver_open_data",
                       tier=FeatureTier.PHASE_2),
]

# ============================================================
# COMPARABLE SALES FEATURES (8 variables)
# ============================================================

COMPARABLE_FEATURES = [
    FeatureDefinition("comp_avg_ppsf_5nn", "Avg $/sqft of 5 nearest comparable sales (6 months)", "float", "derived", unit="$/sqft"),
    FeatureDefinition("comp_std_ppsf_5nn", "Std dev of comparable prices per sqft", "float", "derived", unit="$/sqft"),
    FeatureDefinition("comp_dist_nearest_m", "Distance to nearest comparable sale", "float", "derived", unit="m"),
    FeatureDefinition("comp_days_since_nearest", "Days since nearest comparable sale", "int", "derived", unit="days"),
    FeatureDefinition("comp_price_trend_3m", "3-month price trend slope of comparables", "float", "derived"),
    FeatureDefinition("comp_count_6m_1km", "Count of comparable sales within 1km and 6 months", "int", "derived"),
    FeatureDefinition("comp_median_dom", "Median DOM of comparable sales", "float", "derived", unit="days"),
    FeatureDefinition("comp_list_to_sale_avg", "Average list-to-sale ratio of comparables", "float", "derived"),
]

# ============================================================
# LEASEHOLD FEATURES (for leasehold properties)
# ============================================================

LEASEHOLD_FEATURES = [
    FeatureDefinition("is_leasehold", "Property is leasehold (not freehold)", "bool", "ltsa"),
    FeatureDefinition("lease_remaining_years", "Years remaining on lease", "float", "ltsa", unit="years"),
    FeatureDefinition("lease_type", "Prepaid / Non-prepaid / N/A", "categorical", "ltsa"),
    FeatureDefinition("leasehold_discount_curve", "Estimated discount vs freehold equivalent", "float", "derived"),
    FeatureDefinition("first_nations_leasehold", "Musqueam / Squamish / Tsawwassen / None", "categorical", "derived"),
]


# ============================================================
# AGGREGATE ALL FEATURES
# ============================================================

ALL_FEATURES = (
    STRUCTURAL_FEATURES
    + LOCATION_FEATURES
    + STRATA_FEATURES
    + LAND_FEATURES
    + VIEW_ENVIRONMENTAL_FEATURES
    + MARKET_CONTEXT_FEATURES
    + COMPARABLE_FEATURES
    + LEASEHOLD_FEATURES
)


def get_features_by_phase(phase: int) -> list[FeatureDefinition]:
    """Get all features for a given implementation phase."""
    return [f for f in ALL_FEATURES if f.tier.value <= phase]


def get_features_by_property_type(ptype: PropertyType) -> list[FeatureDefinition]:
    """Get all features applicable to a given property type."""
    return [
        f
        for f in ALL_FEATURES
        if PropertyType.ALL in f.property_types or ptype in f.property_types
    ]


def feature_summary() -> dict:
    """Return summary statistics about the feature set."""
    return {
        "total_features": len(ALL_FEATURES),
        "structural": len(STRUCTURAL_FEATURES),
        "location": len(LOCATION_FEATURES),
        "strata": len(STRATA_FEATURES),
        "land": len(LAND_FEATURES),
        "view_environmental": len(VIEW_ENVIRONMENTAL_FEATURES),
        "market_context": len(MARKET_CONTEXT_FEATURES),
        "comparable": len(COMPARABLE_FEATURES),
        "leasehold": len(LEASEHOLD_FEATURES),
        "by_phase": {
            f"phase_{i}": len(get_features_by_phase(i)) for i in range(1, 5)
        },
        "by_property_type": {
            pt.value: len(get_features_by_property_type(pt))
            for pt in PropertyType
            if pt != PropertyType.ALL
        },
    }
