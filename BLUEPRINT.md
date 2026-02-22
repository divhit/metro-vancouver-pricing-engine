# Building a data-driven property pricing engine for Metro Vancouver

A realtor entering the Metro Vancouver market in 2026 can build a genuine competitive advantage by combining MLS sales data, BC Assessment records, municipal open data, and machine learning into a hybrid automated valuation model targeting **sub-7% median error** on off-market properties and **sub-3% on listed ones**. The most important insight from this research is that Vancouver's pricing dynamics are unusually complex — driven by transit proximity, school catchments, development potential, strata health, and a rapidly shifting regulatory landscape — which means a one-size-fits-all model will fail. **Property-type-specific models layered with neighborhood microsegments and supply-pipeline adjustments** form the winning architecture. What follows is the most thorough blueprint available for building this system.

---

## 1. How condos, townhomes, and houses are priced differently

Vancouver's residential market splits into four fundamentally different pricing regimes, each demanding its own feature weights and comparable selection logic.

### Condos and apartments

The Metro Vancouver apartment benchmark sits at **$704,600** as of January 2026, down 5.9% year-over-year, with downtown units averaging roughly **$1,206 per square foot**. The single most important pricing variable after location is living area, but floor level, view, and building quality create enormous variance within the same building. Research shows a floor premium of roughly **$7/sqft per floor** for floors 1–25, accelerating to **$47/sqft per floor above the 35th storey**. In Coal Harbour specifically, values increase **3.3% per floor** from the 30th floor upward. South-facing exposure commands a 3–7% premium. Unobstructed water views add **20–40%**, mountain views 10–20%, and city views 5–15%.

Construction type creates a structural price gap: concrete high-rises trade at a **5–15% premium per square foot** over comparable wood-frame buildings due to superior soundproofing, durability perception, and lower per-unit strata fees ($0.55–$0.70/sqft monthly for concrete versus $0.45–$0.55 for wood-frame). Strata fees directly affect purchasing power — every $100/month increase reduces the mortgage a buyer can qualify for by roughly **$15,000–$20,000**.

The post-2019 strata insurance crisis remains a critical pricing input. Premiums spiked 50–300% for affected buildings, with deductibles leaping from $5,000–$25,000 to $100,000–$600,000. By 2025, premiums began declining (average 19–24% decreases), but buildings with poor claims histories or deferred maintenance still trade at significant discounts. The leaky condo crisis legacy from 1981–1999 construction also persists: non-remediated buildings carry 10–30% discounts, while fully rainscreened buildings can recoup $100,000+ per unit in value. Parking stalls are worth **$25,000–$50,000** in high-density areas, and storage lockers $3,000–$5,000.

Developer brand matters measurably. **Westbank** commands 10–20% presale premiums through architectural distinction. **Bosa Properties** achieves 5–15% premiums through craftsmanship reputation. Concord Pacific, Polygon, Cressey, and Onni round out the recognized tier. Since late 2022, BC law prohibits most strata rental restrictions (except 55+ buildings), largely eliminating the 5–10% discount that restricted buildings previously carried. Pet restrictions still suppress values by an estimated 3–5%, and 55+ age-restricted buildings trade at 10–20% discounts due to buyer pool limitations.

Depreciation reports — mandatory every three years for stratas with five or more lots — directly affect buyer confidence and lender willingness. Buildings without current reports face material discounts, while those showing underfunded contingency reserves (minimum 25% of annual operating budget required) signal special assessment risk.

### Townhomes and row houses

Townhomes have been the most resilient segment in the 2025–2026 correction, with the Metro benchmark at **$1,043,400** (down 5.4% YoY but outperforming condos). The pricing engine must distinguish three ownership structures: conventional strata (most common), bare land strata (owner holds title to the land underneath, commanding a 5–10% premium), and freehold (no strata structure at all, commanding 10–15% premiums). End units carry a **$20,000–$50,000 premium** (5–8%) over interior units for extra windows and sometimes larger footprints. Private outdoor space adds $30,000–$75,000 depending on size and neighborhood. Side-by-side double garages command $10,000–$20,000 more than tandem configurations. Monthly strata fees run lower than condos at $0.25–$0.40/sqft.

### Single-family detached homes

The Vancouver detached benchmark is **$1,850,800** (January 2026), but the critical insight for pricing these properties is that **land typically represents 70–90% of assessed value** on the West Side and 60–80% on the East Side. In ultra-premium areas like Point Grey and Shaughnessy, land can constitute 85–95% of value, meaning the house itself is nearly irrelevant to pricing — it's a land play.

Standard lot sizes differ dramatically by side of the city: **33 × 122 feet (~4,000 sqft)** on the East Side versus **50 × 122 feet (~6,100 sqft)** on the West Side. This width differential alone drives much of the East-West price gap. Corner lots carry 5–10% premiums for lane access and flexibility. South-facing backyards add 3–7%.

Zoning is now one of the most powerful pricing variables. Under **Bill 44** (enacted December 2023), most single-family lots now permit up to 4 units, and lots of 280 sqm or more within 400 meters of frequent transit allow 6 units with no parking requirements. Vancouver's R1-1 zone permits multiplexes at up to **1.0 FSR**. This legislation creates a density floor under every residential lot's value, though the City's economic testing suggests the impact on land prices is muted in the near term due to a density-bonus framework.

School catchment premiums run **10–20%** for top-tier secondary schools. Lord Byng Secondary (ranked #1 public secondary in BC for over a decade) anchors the Dunbar/Point Grey catchment where detached homes sell for $3–5M+. University Hill Secondary serves the UBC area, Prince of Wales covers Shaughnessy, and Magee serves Oakridge/Kerrisdale. Heritage designation cuts both ways: prestige value versus renovation restrictions and limited densification, netting out to roughly neutral in premium areas.

Environmental factors requiring model inputs include flood plain designation (5–15% discount), Agricultural Land Reserve status (order-of-magnitude value reduction versus residential land), soil contamination near former industrial sites ($100K–$1M+ remediation costs), and Vancouver's tree preservation bylaws ($5,000–$50,000+ in removal/mitigation fees that constrain buildable area).

### Development land and properties with rezoning potential

Development land pricing uses the **residual land value method**: total project revenue minus construction costs, soft costs, financing, developer profit, and city fees. The City of Vancouver targets **75% of the "land lift"** (value increase from rezoning) as a Community Amenity Contribution. Development Cost Levies currently run roughly **$38.42/sqft** for high-density residential (frozen at 2023 levels through 2026). Properties in successful land assemblies sell for **20–40% more** than individual residential sales. Broadway Plan station-area assembly parcels now price at **$500–$700 per buildable square foot**, while Cambie Corridor assemblies run $350–$500. Rezoning from single-family to high-density can generate **100–400%+ land value increases**, though the 2–4+ year timeline and community opposition risk justify significant discounting.

---

## 2. The supply pipeline reshaping Metro Vancouver pricing

An unprecedented **80,000+ housing units** are identified across major projects spanning the next 20–30 years, and this pipeline must be factored into any credible pricing model.

**Oakridge Park** is the nearest catalyst. This $5-billion, 28-acre redevelopment by QuadReal and Westbank delivers **4,300+ homes** across 13 towers, 650,000 sqft of luxury retail (including Louis Vuitton, Prada, and Brunello Cucinelli), a 9-acre rooftop park, and a major civic centre. Phase 1 retail opens spring 2026 with first residential towers completing 2025–2026. The project has already catalyzed at least 9 additional tower proposals within a 4–5 block radius, transforming Oakridge into Vancouver's first Municipal Town Centre outside downtown.

**Senákw**, the largest Indigenous-led housing development in Canadian history, delivers **6,000 purpose-built rental units** in 11 towers on 10.5 acres of Squamish Nation reserve land near Burrard Bridge. Because it sits on reserve land, it is not subject to City of Vancouver zoning. Phase 1 (three towers, ~1,400 units) topped out in mid-2025 with first occupancy expected late 2025/early 2026. OPTrust acquired Westbank's stake in August 2025. The project secured **$1.4 billion in CMHC financing** — the largest CMHC loan ever issued. Its massive rental supply is expected to moderate rents in Kitsilano, Fairview, and the downtown fringe. Metro Vancouver's rental vacancy already hit **3.7% in October 2025**, the highest since 1988.

**Jericho Lands** represents the largest development site in Vancouver outside downtown: 90 acres yielding **~13,000 homes** for 24,000 residents over 25–30 years. The MST Nations (Musqueam, Squamish, Tsleil-Waututh) and Canada Lands Company received Official Development Plan approval in April 2025. Notably, housing will be **leasehold strata** (not freehold), maintaining First Nations perpetual ownership, which historically prices at 10–30% below freehold equivalents. Phase 1 construction could begin around 2028.

The **Broadway Plan corridor** (approved June 2022, amended December 2024) covers 485 blocks and targets **30,000 new units** over 30 years. At least 35 rental towers totaling ~7,000 units have been proposed, plus another 14,000+ units at enquiry stage. December 2024 amendments removed tower-per-block limits near stations and allowed heights up to 40 storeys. The Broadway SkyTrain extension (6 new stations from VCC-Clark to Arbutus) is now expected to open in late 2027.

**River District** adds 7,000 homes on 130 acres of former industrial land in southeast Vancouver, though developer Wesgroup cancelled a condo project in September 2025 reflecting market weakness. **City of Lougheed** in Burnaby plans 10,000+ homes across 23+ towers on the former mall site, with Phase 2 approved in December 2025. The **Surrey-Langley SkyTrain extension** (16 km, 8 stations, opening ~2029) is driving pre-construction interest along the entire Fraser Highway corridor.

The supply impact pattern is clear from historical data: new supply creates short-term price softness (Metro Vancouver currently has ~2,500–3,500 unsold new condos, the highest in 24 years) but long-term value appreciation from amenities and transit investment. The critical finding for the pricing model is that **property types diverge sharply**: detached home prices remained roughly stable in 2025, townhomes outperformed (limited new supply), while condos declined 11% year-over-year due to an oversupply concentrated in that segment.

---

## 3. Macro forces and market conditions shaping the 2026 landscape

The Bank of Canada policy rate sits at **2.25%** after 200+ basis points of cuts from the 5.0% peak. Best 5-year fixed mortgage rates hover around 3.89–4.04%, with variable rates at 3.54–3.55%. The stress test qualifying rate (higher of 5.25% floor or contract rate plus 2%) constrains purchasing power by roughly 4–5%. OSFI is evaluating a potential shift to a **Loan-to-Income cap at 4.5×** gross income, which would meaningfully ease qualification for Vancouver buyers earning $150,000+ annually.

The market is definitively in buyer's territory: **11 months of supply**, a sales-to-active-listings ratio of 8.8%, and January 2026 transactions 29% below year-ago levels. Over 80% of Metro Vancouver homes sold below asking price in 2025. The composite benchmark of $1,101,900 sits roughly **10% below the April 2022 all-time high**. Affordability, while still extreme (mortgage payments consume 85% of median household income), has improved for eight consecutive quarters.

The tax environment has become increasingly punitive for non-resident and vacant property owners. BC's Speculation and Vacancy Tax rises to **3% for foreign owners in 2026** and 4% in 2027. The City of Vancouver Empty Homes Tax sits at 3% with a motion to increase to 5%. Combined with the 20% foreign buyer additional property transfer tax and the federal foreign buyer ban (extended through January 2027), the cost of holding vacant or foreign-owned property in Vancouver is now among the highest globally. The new **BC Home Flipping Tax** (January 2025) imposes 20% on net income from properties sold within 365 days, declining to 0% at 730 days.

Population dynamics are shifting: Metro Vancouver's population may **decline slightly in 2026** for the first time in decades, driven by federal immigration cuts (395,000 PR targets in 2025, declining to 365,000 by 2027) and temporary resident outflows. Immigration historically accounts for 90% of Metro Vancouver's population growth, so these cuts directly reduce housing demand. Rental vacancy at 3.7% and rents declining 5–7% YoY reflect this demand contraction.

Seasonal patterns remain important for model calibration: spring (March–May) is historically the busiest period with peak sales in May/June and prices firming; winter (November–February) sees the lowest volumes with December/January troughs. The seasonal swing in sales volume runs 10–15%, while price seasonality is more muted at 1–3%.

---

## 4. Every data source available to build this engine

A BC realtor has access to an unusually rich data ecosystem for building a pricing engine.

**MLS via Greater Vancouver REALTORS (formerly REBGV)** provides the foundational transaction dataset. Accessed through the **Paragon system** (bcres.paragonrels.com), it contains sale price, list price, original list price, price change history, days on market, cumulative days on market, property size, bedrooms/bathrooms, year built, strata fees, parking details, floor level, exposure, building amenities, zoning, PID, legal description, agent remarks, and 20+ years of historical sales. IDX/RETS feeds enable programmatic data access with proper licensing. The MLS Home Price Index tracks benchmark property prices by area and type. CREA's Data Distribution Facility provides national active listing data (not sold data) via a Web API.

**BC Assessment** delivers assessed values (total, land, improvement split), property characteristics (lot size, building size, year built, quality class, condition), ownership records, and three most recent sales per property with conveyance type classification. Bulk electronic data is available as annual XML files plus weekly CSV updates through their Data Advice service (six joinable CSV tables keyed on FOLIO_ID). Academic researchers can access historical assessment data through UBC's Abacus repository, with Jens von Bergmann's R package "abacusBCA" providing programmatic access.

**LTSA (Land Title and Survey Authority)** provides title searches (~$10.89 each via myLTSA portal) showing registered owners, legal descriptions, PIDs, mortgages, easements, covenants, and historical transfers. **ParcelMap BC** offers province-wide electronic mapping of all 2+ million titled parcels as open data through the BC Data Catalogue.

**City of Vancouver Open Data** (opendata.vancouver.ca) provides property tax data since 2006, complete zoning district maps, issued building permits since 2017, property parcel polygons, and development permit data — all accessible via REST API with JSON responses and updated daily to weekly.

**CMHC's Housing Market Information Portal** delivers housing starts, completions, under-construction counts, market absorption data, vacancy rates, and average rents at census-tract granularity. The open-source R package "cmhc" by mountainMath provides API access. **Walk Score** (api.walkscore.com) returns walkability, transit, and bike scores per address; research shows each Walk Score point increases home price by roughly **$3,250**. **Fraser Institute** publishes annual school rankings that demonstrably influence buyer behavior. The VPD's GeoDASH tool provides crime mapping data. City flood maps, ALR boundary maps from the Agricultural Land Commission, and Environment Canada climate data round out the environmental inputs.

---

## 5. How automated valuation models actually work — and what accuracy to target

The pricing engine should employ a **hybrid architecture** combining the interpretability of hedonic pricing with the accuracy of gradient-boosted trees, calibrated against professional appraisal methodologies.

**Hedonic pricing models** decompose property value into implicit prices of individual characteristics using regression: ln(Price) = β₀ + β₁(sqft) + β₂(bedrooms) + ... + ε. The semi-log form is standard, with coefficients representing percentage price changes per unit of each feature. Key variables in order of importance: living area, location (lat/long coordinates), year built, lot size, parking, bathrooms, floor level, and view. The primary limitations are assumed functional form and inability to capture complex non-linear interactions without manual feature engineering.

**Gradient boosting (XGBoost/LightGBM)** dramatically outperforms hedonic regression — one study showed **84.1% accuracy versus 42%** for hedonic models on the same dataset. These algorithms build decision trees sequentially, each correcting prior errors, and handle non-linear relationships, missing values, and feature interactions natively. For Vancouver's market, where the interaction between floor level and view quality, or between lot width and zoning, creates non-linear price effects, tree-based methods capture dynamics that regression cannot.

**Zillow's Zestimate** offers the reference architecture. The current "Neural Zestimate" (version 6.7, launched June 2021) processes 104+ million homes using end-to-end deep learning incorporating MLS data, tax records, property photos (via convolutional neural networks), and market trend signals. On-market accuracy is **1.83% median error**; off-market accuracy is **7.01%**. The key lesson: on-market versus off-market distinction is critical because listing price is the single most powerful signal.

**BC Assessment** uses mass appraisal — grouping similar properties for statistical analysis rather than individual inspection. Accuracy is typically within ±5% but can deviate 15–30% from market value due to the 6-month lag, no interior inspections, and mass grouping that misses unique features.

Professional appraisers use three approaches. The **sales comparison approach** (dominant for residential) selects 3–6 comparable recent sales and applies dollar adjustments: typically **$200–$800+/sqft** for living area differences, **$20,000–$60,000 per parking stall**, **$2,000–$10,000 per floor level**, **$5,000–$30,000 per bedroom**, and **$50,000–$300,000+** for significant views. The **cost approach** (land value plus replacement cost minus depreciation) applies mainly to new construction. The **income approach** (NOI ÷ cap rate) serves investment properties.

For accuracy benchmarks, the industry standard MAPE thresholds are: below 5 is excellent, 5–10 is good, 10–13 is minimum acceptable, and above 13 is "of no real value." A Vancouver-focused AVM should target **sub-5% MAPE on-market** and **7–10% off-market**, with a hit rate exceeding 90% of residential properties.

---

## 6. Neighborhood-level pricing nuances the model must capture

Vancouver's market exhibits extreme micro-geographic variation that generic AVMs miss entirely.

The **East-West divide** remains the defining structural feature: West Side detached benchmark of ~$2,956,400 versus East Side ~$1,760,000, a **1.84:1 ratio**. However, this gap is narrowing. West Side detached values dropped 12.2% YoY in January 2026 versus roughly 5% on the East Side. Individual East Side properties now occasionally exceed comparable West Side sales — a Killarney 34-foot lot sold for $2.8M versus a Point Grey 56-foot lot at $2.75M. The "East Side renaissance" driven by brewery culture, arts scenes, walkability improvements, and younger demographics is a real and measurable trend the model must track.

**Transit proximity** commands a **5–20% premium** within 400–800 meters of SkyTrain stations, but with an important nuance: a UBC study found that in wealthy West Side neighborhoods, being **farther** from SkyTrain actually correlated with higher prices (unwanted density effect), while East Side proximity consistently boosted values. This interaction between neighborhood wealth and transit proximity must be encoded in the model. Properties near upcoming extensions see 10–20% appreciation ahead of construction completion, with the strongest gains 3–5 years pre-opening. Broadway Plan station-area land already prices at $500–$700 per buildable square foot.

**Leasehold versus freehold** creates dramatic value differentials. UBC 99-year prepaid leasehold properties (expiring ~2102) trade nearly at freehold parity because of the long remaining term. But False Creek South leaseholds expiring 2036–2046 trade at massive discounts (a 1,000 sqft unit at $300,000 versus $1.2–1.5M freehold equivalent). The critical threshold: value drops sharply when the remaining lease falls below typical mortgage amortization (25 years), and most banks refuse financing below 30–40 years remaining. Musqueam non-prepaid leasehold properties require 50%+ down payment and carry the cautionary legacy of the 1995 rent review crisis.

**Land assembly dynamics** create pricing distortions the model must flag. Properties within Broadway Plan or Cambie Corridor assembly zones can be worth 2–3× the value of identical properties just outside plan boundaries. Holdout lots — those refusing to join assemblies — face the risk of being "designed around," leaving them stranded next to construction sites. The model should identify assembly potential based on OCP designation, proximity to other recent assembly sales, and lot dimensions suitable for aggregation.

The **new construction versus resale gap** has become structural. Presale condos carry GST (5%) that resale units do not, adding $25,000–$50,000+ to the effective price. Developer costs (land, construction, DCLs, CACs, financing) set a floor that doesn't adjust downward with the resale market, so the gap widens during corrections. Currently, developers are offering flash sales with up to 25% discounts plus incentive packages to move inventory — a dynamic the model should track as a market-cycle indicator.

---

## 7. Architectural blueprint for the pricing engine

### Data ingestion layer

The engine requires five primary data streams, each with different update frequencies:

- **MLS transaction data** (via Paragon RETS/IDX feed): All sales, listings, price changes, status changes, and property characteristics. Update daily. This is the single most valuable dataset — sale prices, days on market, and listing history form the training backbone. Approximately 20+ years of historical data available.
- **BC Assessment roll data** (via Data Advice CSV/XML): Assessed values (total/land/improvement split), lot dimensions, building characteristics, ownership, three most recent sales. Update annually (January completed roll) with weekly CSV refreshes.
- **City of Vancouver Open Data** (via REST API): Zoning districts, building permits, development permits, property tax records. Update weekly. Critical for identifying rezoning activity, permit status, and development pipeline proximity.
- **LTSA title data** (via myLTSA queries): Encumbrances, ownership history, legal descriptions. Query on-demand per property (cost: ~$10.89/search). Essential for identifying leasehold terms, easements, and covenants.
- **Supplementary APIs**: Walk Score/Transit Score (update every 6 months), CMHC housing starts/vacancy (monthly/semi-annual), Fraser Institute school rankings (annual), VPD crime data (quarterly), flood/ALR maps (static with periodic updates).

### Feature engineering — the complete variable set

The model requires approximately 80–120 engineered features organized into seven categories:

**Structural features** (20–25 variables): living area sqft, lot size sqft, bedrooms, bathrooms, year built, effective age, construction type (wood-frame/concrete/steel), number of stories, parking stalls (count and type), storage lockers, floor level (condos), unit exposure/facing, layout efficiency (sqft per bedroom), building total units, suite/ADU presence, basement type and finish level, heating/cooling system, firewall/party wall type.

**Location features** (15–20 variables): latitude/longitude, neighborhood code, sub-area code, distance to nearest SkyTrain station (meters), distance to waterfront, distance to CBD, Walk Score, Transit Score, Bike Score, school catchment ID and ranking score, distance to nearest park, distance to nearest commercial node, census tract median income, census tract population density.

**Strata/building features** (15–20 variables, condos/townhomes only): monthly strata fee per sqft, strata fee-to-price ratio, contingency reserve fund percentage, depreciation report age (months since last report), special assessment history (count and total in past 5 years), insurance deductible amount, building age bucket, rainscreen status (yes/no/partial), developer brand tier, rental restriction status, pet restriction status, age restriction status, concierge (yes/no), pool (yes/no), gym (yes/no).

**Land and development features** (10–15 variables, primarily SFH/land): zoning code, maximum permitted FSR, maximum permitted height, lot frontage, lot depth, lot shape regularity score, land-to-total assessment ratio, OCP future land use designation, transit-oriented area designation (yes/no), assembly potential flag (based on adjacent lot ownership patterns and plan designations), heritage designation status, ALR status, flood plain designation.

**View and environmental features** (5–8 variables): view type classification (water/mountain/city/park/none), view quality score (unobstructed/partial/peek-a-boo/none), floor-to-obstruction height ratio, proximity to industrial zones, proximity to major roads (noise), soil contamination risk flag, geotechnical risk flag.

**Market context features** (10–15 variables): months since sale (time adjustment), sales-to-active-listings ratio (trailing 3-month for sub-area), benchmark price index (sub-area, property type), interest rate at time of sale, seasonal indicator (month), new supply pipeline within 1km radius (units under construction + approved), average days on market for sub-area, price-to-rent ratio for sub-area, list-to-sale price ratio (sub-area trailing average).

**Comparable sales features** (5–8 variables): average price per sqft of 5 nearest comparable sales (same type, within 6 months), standard deviation of comparable prices, distance to nearest comparable, days since nearest comparable sale, price trend of comparables (3-month slope).

### Model architecture — a three-tier hybrid system

**Tier 1: Gradient-boosted ensemble (primary prediction)**. Train separate **LightGBM models** for each property type (condo, townhome, detached, development land) using the full feature set above. LightGBM is recommended over XGBoost for faster training with leaf-wise growth and superior handling of categorical features. Use log-transformed sale price as the target variable. Training data: all arm's-length MLS sales from the past 5 years, time-adjusted to the valuation date using sub-area benchmark index movements. Implement geographic cross-validation (spatial k-fold) to prevent data leakage from spatial autocorrelation. Target: 80,000–120,000 transactions for training across all types.

**Tier 2: Rules-based adjustment layer**. Apply deterministic adjustments for factors that ML models handle inconsistently or that require domain expertise:

- Leasehold adjustment: Apply discount curve based on remaining lease term (0% discount above 70 years, scaling to 30%+ below 30 years, with a financing cliff at 25 years)
- Assembly premium/discount: Flag properties in active assembly corridors; apply 15–30% premium if assembly indicators are positive
- Supply pipeline proximity adjustment: Discount 2–5% for properties within 500m of major projects delivering 500+ units within 24 months (short-term supply effect), with a reversal to 0–3% premium once projects are established (5+ years post-completion)
- Strata health adjustment: Penalize buildings with expired depreciation reports (-3%), high insurance deductibles (-5%), or special assessment history (-2% per $10K assessed)
- Heritage adjustment: Neutral to slight premium in Shaughnessy/First Shaughnessy; slight discount in areas where heritage limits density under Bill 44
- New construction GST offset: Add 5% effective premium when comparing presale to resale

**Tier 3: Comparable sales reconciliation**. After Tiers 1 and 2 produce an estimate, identify the 5 most similar recent sales using a weighted similarity score (property type match, proximity, size similarity, age similarity, strata fee similarity). Calculate the implied price from each comparable using standard appraisal adjustments. If the ML estimate diverges from the comparable-implied range by more than 10%, flag for manual review and weight the final estimate 60% ML / 40% comparable reconciliation. If within 10%, weight 80% ML / 20% comparable.

### Property-type-specific versus unified model

**Use separate models per property type.** The features, comparable pools, and pricing dynamics differ too fundamentally between condos, townhomes, detached homes, and development land. A unified model would waste capacity trying to learn that floor level matters for condos but not houses, or that FSR matters for development land but not rental apartments. However, use a **shared geographic embedding layer** — train a neural network on lat/long coordinates to learn neighborhood price surfaces, then use the embedding as input features for all four property-type models. This captures neighborhood effects consistently across types.

### Handling the supply pipeline

Create a **supply impact index** per geographic cell (500m × 500m grid across Metro Vancouver):

1. Count all approved/under-construction units within the cell and adjacent cells
2. Weight by expected completion date (inverse of months to completion)
3. Classify by type (rental versus strata, market versus affordable)
4. Calculate the ratio of pipeline units to existing housing stock in the cell
5. Apply historical supply-impact coefficients calibrated from past project completions (e.g., Olympic Village, Marine Gateway, Brentwood) to estimate the short-term price impact

Update this index monthly as new permits and project milestones are reported.

### Output presentation

For each property valuation, deliver:

- **Point estimate** (most likely sale price) — the Tier 1 + Tier 2 + Tier 3 blended result
- **Confidence range** (80% prediction interval) — derived from the model's residual distribution for similar properties in the same sub-area, typically ±5–12% of the point estimate
- **Confidence grade** (A/B/C) — based on comparable availability, data completeness, and model certainty. Grade A: 5+ close comparables within 6 months, all features populated, prediction interval under ±7%. Grade B: 3–4 comparables, most features populated, interval ±7–12%. Grade C: fewer than 3 comparables or significant data gaps, interval exceeding ±12%
- **Top 5 comparable sales** with adjustment details showing how each was reconciled to the subject
- **Key value drivers** — SHAP (SHapley Additive exPlanations) values showing which features push the price up or down versus the sub-area average
- **Market context dashboard** — current sales-to-active ratio, benchmark trend, days on market for similar properties, seasonal indicator
- **Risk flags** — leasehold expiry, depreciation report age, insurance concerns, assembly corridor, flood plain, supply pipeline exposure

### Validation methodology

Implement three validation approaches:

1. **Backtest accuracy**: Hold out the most recent 6 months of sales and measure MAPE, mean error, and percentage of estimates within ±5%, ±10%, and ±15% of actual sale price. Target: MAPE under 7% overall, under 5% on-market.
2. **Rolling monthly accuracy**: Each month, compare the previous month's estimates against actual outcomes. Track accuracy trends by property type and sub-area to identify model drift.
3. **Appraiser benchmarking**: For 20–30 properties per quarter, obtain professional appraisals and compare to model output. The model should match or exceed appraiser accuracy (typically ±5–10%).

### Technology stack

- **Data storage**: PostgreSQL with PostGIS extension for spatial queries; property-level feature store in a columnar format (Parquet files on cloud storage for ML training)
- **ETL pipeline**: Python (pandas, geopandas) for data ingestion and transformation; Apache Airflow or Prefect for orchestration of daily/weekly/monthly update jobs
- **ML framework**: LightGBM for primary models; SHAP for explainability; scikit-learn for feature preprocessing and cross-validation; optionally PyTorch for the geographic embedding neural network
- **API layer**: FastAPI serving real-time predictions; Redis cache for frequently queried properties
- **Frontend**: React dashboard with Mapbox GL for geographic visualization; property comparison views; market context widgets
- **Cloud infrastructure**: AWS (S3 for data lake, EC2/SageMaker for ML training, RDS for PostgreSQL) or GCP equivalent. Budget estimate: $200–$500/month for infrastructure at moderate scale.

### Phased implementation roadmap

**Phase 1 (Months 1–3): Data foundation and baseline model.** Secure MLS data access through REBGV IDX agreement. Ingest 5 years of historical sales, BC Assessment bulk data, and City of Vancouver open data. Build the PostgreSQL/PostGIS database. Engineer the structural and location feature sets. Train initial LightGBM models per property type using available features. Target accuracy: MAPE under 12%.

**Phase 2 (Months 4–6): Feature enrichment and accuracy improvement.** Integrate Walk Score API, Fraser Institute school data, CMHC supply data, and transit distance calculations. Add strata-specific features (fees, building age, construction type). Implement the Tier 2 rules-based adjustment layer for leasehold, assembly, and strata health. Build the comparable sales reconciliation (Tier 3). Target accuracy: MAPE under 9%.

**Phase 3 (Months 7–9): Supply pipeline and market context.** Build the supply impact index with development pipeline tracking. Add market context features (sales-to-active ratio, benchmark trends, seasonal adjustments). Integrate LTSA data for leasehold term and encumbrance flagging. Implement the geographic embedding neural network for neighborhood price surfaces. Target accuracy: MAPE under 7%.

**Phase 4 (Months 10–12): Production deployment and client-facing tools.** Build the FastAPI prediction service and React dashboard. Implement the output presentation layer (point estimate, confidence range, comparables, SHAP explanations, risk flags). Set up automated validation and monitoring. Create client-facing CMA report generation that combines model output with professional formatting. Launch beta testing with live listings.

**Phase 5 (Ongoing): Continuous improvement.** Monthly model retraining with new sales data. Quarterly accuracy review and feature refinement. Annual recalibration of rules-based adjustments. Integration of new data sources as they become available (property photos for condition scoring, satellite imagery for neighborhood change detection). Expansion to cover all Metro Vancouver municipalities with municipality-specific model variants.

---

## Conclusion: a defensible edge in a data-rich market

The Metro Vancouver real estate market in 2026 offers a rare combination of conditions for a data-driven pricing engine: deep historical transaction data through MLS, granular property characteristics from BC Assessment, rich municipal open data, and a market correction that makes accurate pricing more valuable than ever. The most important architectural decision is building **property-type-specific models** rather than one unified system, because the pricing physics differ fundamentally between a Coal Harbour penthouse and a South Vancouver teardown lot.

Three non-obvious insights should guide the implementation. First, the interaction between transit proximity and neighborhood wealth reverses sign — transit boosts East Side values but can depress West Side values — meaning a naive "distance to SkyTrain" feature will mislead the model without a wealth interaction term. Second, the supply pipeline's impact is asymmetric across property types: massive rental supply (Senákw's 6,000 units) primarily pressures condo values and rents while leaving detached home land values largely unaffected. Third, the leasehold discount is non-linear — nearly zero for 70+ year terms, then accelerating sharply below the 25-year financing threshold — creating a pricing cliff that most AVMs model poorly.

A realtor deploying this system achieves something no generic AVM can: the ability to quantify exactly why a Dunbar home in Lord Byng catchment commands a premium, why a non-rainscreened 1990s wood-frame in the West End carries hidden risk, or why a 33-foot East Vancouver lot near a planned SkyTrain station may be worth more as a sixplex site than as a single-family home. That specificity, backed by transparent data, is the competitive advantage.

---
---

# APPENDIX A: Comprehensive Data Source Catalog

*The following sections document every data source identified through exhaustive research, organized by category. This appendix significantly expands upon Section 4 of the original blueprint.*

---

## A1. MLS and Real Estate Transaction Data — Expanded

### A1.1 GVR BridgeAPI (RESO Web API) — Primary MLS Access

**Critical update**: Legacy RETS was shut down **February 28, 2025**. GVR now exclusively uses the **BridgeAPI** platform by Bridge Interactive, a Platinum Certified RESO Web API.

**Three API endpoints:**
- **IDX endpoint** — IDX-approved data (active listings for public display)
- **VOW endpoint** — Enhanced data for registered users, including **sold data**
- **Standard endpoint** — All listings with `FeedTypes` field indicating IDX, VOW, or both

**Data normalization:** All data normalized to the RESO Data Dictionary (1,700+ fields, 3,100+ lookups). Key fields include: ListPrice, ClosePrice, OriginalListPrice, BedroomsTotal, BathroomsTotalInteger, LivingArea, LotSizeArea, YearBuilt, PropertyType, PropertySubType, StandardStatus, DaysOnMarket, CumulativeDaysOnMarket, Latitude, Longitude, TaxAnnualAmount, TaxAssessedValue, ParkingTotal, GarageSpaces, PublicRemarks, PrivateRemarks, ListAgentKey, BuyerAgentKey, and hundreds more.

**Reciprocal data:** A single GVR API connection provides access to listings from **GVR, FVREB (Fraser Valley), and CADREB (Chilliwack)** through a reciprocal agreement.

**Licensing requirements:**
- Must hold a valid BC real estate license and be a member of a GVR-member brokerage
- IDX Data Agreement submitted to GVR (`idx@gvrealtors.ca`)
- VOW Data Agreement (separate) for sold data access
- Compliance review of website/application (can take several weeks to months)

**Cost:** IDX feed free of board charges; VOW feed ~$1,500/year; DDF $9/month; board fee ~$3.50/month

**Update frequency:** Real-time or near-real-time (15 minutes to 1 hour refresh cycles)

### A1.2 CREA Data Distribution Facility (DDF)

**What it is:** National database aggregating MLS listing data from most regional boards (~65% of Canadian listings).

**Access:** REST API at `ddfapi-docs.realtor.ca`. CREA members only.

**Limitation for AVM:** Provides **active listings only** — no sold/closed transaction data, no historical pricing. Useful for market monitoring but not model training.

**Cost:** $9/month per DDF feed.

### A1.3 CREA MLS Home Price Index (HPI)

**What it is:** Sophisticated benchmark price tool tracking "typical" home price changes using 15+ years of MLS data and hedonic regression.

**Data:** Benchmark prices, composite indices, single-family/townhouse/apartment indices by major market area. National, provincial, and board level.

**Access:** Public interactive tool at `crea.ca/housing-market-stats/mls-home-price-index/hpi-tool/`. Downloadable CSV/Excel. Board-level stats at `creastats.crea.ca`.

**Cost:** Free for public/research use. No dedicated API.

### A1.4 Third-Party MLS Data Aggregators

**Repliers** — Unified API aggregating MLS data from hundreds of systems across North America into one normalized RESO API. BC coverage via GVR reciprocal agreement. Month-to-month, no setup fees. 1M API requests/month included. Still requires valid MLS agreements. Contact: `repliers.com`

**Houski** — Canada's largest independent property data platform: **17M+ Canadian properties**, 200+ fields per property. Includes AVM valuations, tax assessments, transaction history, permits. **No MLS license required.** REST API with Python/JS/PHP/Ruby client libraries. Bulk CSV/JSON exports. Pay-per-use at ~$0.015/request. Enterprise annual contracts for companies with $2M+ revenue. Contact: `houski.ca/property-api`

**HouseSigma** — Consumer platform with AI valuations and sold data. Claims **2.72% on-market error, 6.79% off-market error**. No public API. Greater Vancouver coverage.

**Zolo.ca** — Canada's largest independent marketplace, 10M+ monthly users, 254K+ listings. Updated every 15 minutes. No public API.

**REW.ca** — BC-focused platform with "Property Insights" feature. No public API.

**WOWA.ca** — Home valuation tool using linear regression on tax/market data. Offers "Home Valuation API" but developer access details not publicly documented. Covers 90%+ of Canadian population.

**MappedBy** — REST API for US/Canada property data. Open signup. Contact: `hello@mappedby.com`

### A1.5 Presale / New Construction Data

**Buildify** — Canada's leading pre-construction data platform: **4,000+ development projects**, 150+ attributes per unit (floorplans, pricing, availability, unit mix). Updated daily. REST API and Broker Portal. Custom pricing after demo. Contact: `getbuildify.com`

**MLA Canada** — Western Canada's most comprehensive presale sales/marketing company. Project-level info. No public API.

**Rennie** — Vancouver-based presale specialist. No public API.

**BuzzBuzzHome (Livabl)** — North American new construction listing platform. Comprehensive project details. No public API.

### A1.6 Rental Data Sources

**CMHC Rental Market Survey** — Gold standard: annual survey of purpose-built rental buildings. Vacancy rates, average rents by bedroom type, turnover rates. Census-tract level. R package `cmhc` for programmatic access. Free.

**Rentals.ca** — Canada's largest rental marketplace. Monthly National Rent Reports. Data partnership with Statistics Canada. No public API.

**Liv.rent** — Vancouver-made platform. Monthly rent reports by neighborhood. No public API. Reports at `liv.rent/blog/rent-reports/vancouver/`

**PadMapper** — Map-focused rental aggregator. 1,500+ Metro Vancouver listings. No public API.

### A1.7 Teranet / PurView

**Teranet-National Bank House Price Index** — Monthly repeat-sales index for 11 Canadian markets including Vancouver. Based on land registry data (includes all transfers, not just MLS — ~20% more data). Most accurate Canadian HPI methodology. Free for index reports. Enterprise pricing for raw data.

**PurView** — Teranet's property intelligence product. BC coverage includes structural data, assessment data, LTSA title search integration. AVM product available. API delivery. Enterprise pricing.

---

## A2. Government and Assessment Data — Expanded

### A2.1 BC Assessment — Complete Product Catalog

**Data Advice (Primary Bulk Product)** — Six CSV tables joined on `FOLIO_ID`:

| Table | Contents |
|-------|----------|
| `bca_folio_descriptions` | Property descriptions, actual use codes, manual classification codes, property class |
| `bca_folio_addresses` | Civic addresses with `PRIMARY_FLAG` for primary address per folio |
| `bca_folio_legal_descriptions` | Legal descriptions (lot, plan, district lot, etc.) |
| `bca_folio_gnrl_prop_values` | Assessed land value and improvement value by `GEN_PROPERTY_CLASS_CODE` |
| `bca_folio_sales` | Three most recent sales per property (price, date, `CONVEYANCE_TYPE`) |
| Ownership/jurisdictions | Folio-level ownership and jurisdiction metadata |

**Residential Inventory Extract** — Detailed residential characteristics: bedrooms, bathrooms, finished area, basement, year built, quality grade. Updated quarterly.

**Commercial Inventory Extract** — Complete commercial building details: floor area, construction type, year built, floors. Updated quarterly.

**Permit Export** — Building permit data linked to folios. Updated monthly.

**Update frequency:** Annual full roll (January completed, April revised); weekly CSV full-roll theme files; weekly XML delta updates; monthly updates also available.

**Cost:** Commercial license required. Contact `propertyinfo@bcassessment.ca` or 1-866-825-8322. Individual BC OnLine queries: $7.00 (owner/location) or $9.50 (assessment roll).

**Academic access:** UBC Abacus repository — restricted to researchers at UBC, SFU, UNBC, UVic. Apply via `jeremy.buhler@ubc.ca`. R package `abacusBCA` provides programmatic access.

### A2.2 MountainMath R Package Ecosystem

Jens von Bergmann maintains a comprehensive suite of R packages for Canadian data analysis:

| Package | Purpose | Key Functions |
|---------|---------|---------------|
| `abacusBCA` | BC Assessment data from UBC Abacus | `get_bca_data()`, `get_bca_spatial_folios()` |
| `cancensus` | Statistics Canada Census (1996-2021) | Via CensusMapper API. All geo levels to dissemination area |
| `cansim` | Statistics Canada data tables (CANSIM/NDM) | Links to census via GeoUID |
| `cmhc` | CMHC Housing Market Information Portal | `get_cmhc()`, `list_cmhc_surveys()` |
| `tongfen` | Harmonize census data across years | Common geography reconciliation |
| `VancouvR` | City of Vancouver Open Data API | `get_cov_data()`, `list_cov_datasets()` |

All packages are free and open source. Some require API keys (free registration).

### A2.3 LTSA — Expanded

**Title Search returns:** Registered fee simple holders, legal description, PID, all charges/liens/interests (mortgages, easements, covenants, caveats), strata plan number. **Does NOT include:** phone numbers, chain of ownership, or property values.

**Search methods:** By PID, short legal description, charge number, owner name, or title number.

**Two tiers:**
- **myLTSA Explorer:** Up to 40 titles/documents annually. Pay-as-you-go via credit card. $3.40/transaction service charge.
- **myLTSA Enterprise:** Full suite. Deposit account. $2.10/transaction service charge.

**Fee schedule (April 2025):**
| Service | Fee |
|---------|-----|
| Title Search | $10.89 |
| Name Search | $10.89 |
| Document/Plan Order | $17.51 |
| State of Title Certificate | $16.50 |

**ParcelMap BC** — Four data products:
1. Fabric Extract (fully attributed, all parcels)
2. Real-World Changes (recent changes)
3. Fabric Spatial Improvements (accuracy updates)
4. Province-Wide Snapshot (complete point-in-time)

Free open data: Parcel Polygons on BC Data Catalogue (Open Government Licence). WMS/WFS services. JSON, GML, CSV, Shapefile, FGDB, KML formats. NAD83(CSRS)/BC Albers (EPSG:3153).

### A2.4 Municipal Open Data Portals — All Metro Vancouver

**City of Vancouver** (`opendata.vancouver.ca`) — Most comprehensive. Key datasets:

| Dataset | Description | Update |
|---------|-------------|--------|
| Property Tax Report | PID, zoning, land/improvement values, tax levy. Since 2006. | Weekly |
| Zoning Districts and Labels | Zoning polygons with codes | Weekly |
| Issued Building Permits | Type, value, address. Since 2017. | Ongoing |
| Property Parcel Polygons | Assessment-based land polygons | Periodic |
| Heritage Sites | Vancouver Heritage Register | Periodic |
| Designated Floodplain | Coastal and Still Creek floodplain | Static |
| Elementary/Secondary School Catchments | Catchment area boundaries | Periodic |
| Parks | Park boundaries as polygons | Periodic |
| Public Trees | Street/park trees with species, diameter | Periodic |
| Business Licences | Active licences by category | Ongoing |
| Local Area Boundary | 22 neighbourhood boundaries | Static |

API access with JSON responses. R package `VancouvR` for programmatic access.

**City of Burnaby** (`data.burnaby.ca`) — Zoning polygons, property boundaries, aerial photos, LiDAR, terrain models, 3D DEMs, parks. GIS formats via ArcGIS Hub. BurnabyMap interactive tool.

**City of Surrey** (`data.surrey.ca`) — **179,000+ property assessments** with ownership types, zoning, plan numbers, valuation brackets, geospatial indicators. Building permits, development permits, zoning, OCP designations. 200+ layers in COSMOS mapping system. 750GB+ bulk data available on request (`gis@surrey.ca`). CSV, KML, GeoJSON, GeoTIFF.

**City of Richmond** — GeoHub (ArcGIS-based) at `richmond-geo-hub-cor.hub.arcgis.com`. Parcel polygons, zoning. Richmond Interactive Map (RIM) for property lookup.

**City of Coquitlam** (`data.coquitlam.ca`) — 120+ datasets. Property boundaries, zoning, aerial imagery, LiDAR. Spatial Data Catalogue with GIS/CAD downloads. QtheMap interactive tool.

**District of North Vancouver** — GEOweb with **170+ free datasets**. Property parcels, zoning, parks, schools, heritage, census data. SHP/FGDB/DWG/KML. Automatically refreshed weekly.

**District of West Vancouver** — Open Data Portal with zoning maps, property data. WESTMap for property search.

**City of New Westminster** (`opendata.newwestcity.ca`) — Property parcels, zoning, building permits, heritage sites. CityViews Map interactive tool.

**Metro Vancouver Regional District** (`open-data-portal-metrovancouver.hub.arcgis.com`) — Regional boundaries, regional parks, land use designations, Urban Containment Boundary, growth projections, Housing Data Book, rental affordability data, aerial LiDAR 2022.

### A2.5 Statistics Canada Census Data

**Census Profile (1996-2021)** — Income, demographics, housing characteristics, education, labour, immigration. Available at all geographic levels down to dissemination areas.

**Key housing variables:** Dwelling type, tenure, shelter costs, housing condition, suitability, period of construction, rooms/bedrooms, condo status, subsidized housing.

**Access:** SDMX RESTful API (no key required), bulk CSV downloads, `cancensus` R package via CensusMapper API (free key). Next census: 2026.

**CANSIM/NDM Data Tables:** New Housing Price Index (monthly since 1981), building permits, CPI shelter component, GDP by industry. Via `cansim` R package or StatCan Web Data Service.

### A2.6 CMHC — Complete Dataset Catalog

| Survey | Data | Frequency | Geography |
|--------|------|-----------|-----------|
| Starts and Completions (SCSS) | Housing starts, completions, under construction, absorbed/unabsorbed units + prices | Monthly | Census tract |
| Rental Market Survey (RMS) | Vacancy rates, average rents, rent change, rental universe by bedroom/size/age | Annual (October) | CMA zones |
| Secondary Rental Market (SRMS) | Condo vacancy, rents, share rented | Annual | CMA |
| Seniors' Housing | Seniors' rental vacancy, rents | Annual | CMA |
| Housing Market Assessment | Overheating, price acceleration, overvaluation, overbuilding indicators | Quarterly | CMA |

R package `cmhc` for programmatic access. Open data on `open.canada.ca`. All free.

### A2.7 Natural Resources Canada Geospatial Data

**High Resolution DEM** — LiDAR-derived, 1-2m resolution, covering Metro Vancouver. GeoTIFF via FTP, STAC API. Free.

**Building Footprints** — Automatically extracted from LiDAR. **13.6M building footprints** nationwide. Includes elevation, min/max building heights. GeoJSON/Shapefile. Free.

**Flood Mapping** — National flood prediction maps (ML-based), near-real-time satellite flood extent, flood archives. Free.

### A2.8 BC Data Catalogue — Additional Datasets

| Dataset | Contents | Update |
|---------|----------|--------|
| ALR Boundaries | Agricultural Land Reserve polygons | Quarterly |
| Mapped Floodplains | Historical 200-year floodplain boundaries (1974-2000s) | Static |
| Environmental Remediation Sites | Known/potentially contaminated properties | Ongoing |
| Crown Land Tenures (TANTALIS) | Leases, licenses, reserves | Ongoing |
| School District Boundaries | All BC school districts | Periodic |
| Municipal Boundaries | Administrative boundaries | Periodic |

All free under Open Government Licence - BC. Available via BCGW WMS/WFS.

### A2.9 BC Housing

**New Homes Registry Data** — Monthly report on registered new homes by type, size, location across BC.

**Housing Needs Reports** — Municipality-level housing needs assessments (mandated by province).

**BC Stats housing data:** Building permit values/counts, housing starts, housing sales, Mortgage Payment Percent of Income (MPPI) quarterly by CMA.

All free.

---

## A3. Third-Party APIs and Alternative Data — Expanded

### A3.1 Walk Score API

**Endpoints:**
- `https://api.walkscore.com/score` — Walk Score, Transit Score, Bike Score
- `https://transit.walkscore.com/transit/score/` — Transit Score
- `https://transit.walkscore.com/transit/stop/search` — Stops near location
- `https://api.walkscore.com/traveltime/` — Travel time between points

**Free tier:** 5,000 calls/day. Paid tiers scale by site traffic. Enterprise for high-volume/offline. JSON responses.

### A3.2 TransLink GTFS Data

Complete transit schedules for all Metro Vancouver (SkyTrain, buses, SeaBus, West Coast Express). Stop locations (lat/lng), routes, schedules, frequencies, trip shapes. GTFS-RT V3 for real-time positions and alerts. Register at `developer.translink.ca`. Free.

### A3.3 Google Maps Platform

**Geocoding API** — $5/1K requests. **Places API** — Tiered: Basic $5/1K, Advanced $7/1K, Preferred $10/1K. **Distance Matrix / Routes API** — 10K free/month (Essentials tier), then $5-10/1K. **Street View** — $7/1K. As of March 2025, old $200/month credit replaced with SKU-specific free tiers.

### A3.4 Mapbox Isochrone API

Returns polygons of areas reachable within specified travel times (5-60 min) by driving/walking/cycling. 100K free temporary geocoding/month. Isochrones at $2/1K requests. 300 req/min rate limit. GeoJSON polygons. Cheaper than Google for travel-time analysis at scale.

### A3.5 OpenStreetMap / Overpass API

Comprehensive amenity data: parks, schools, transit stops, grocery stores, restaurants, building footprints, land use. Overpass QL query language. Public endpoint: `https://overpass-api.de/api/interpreter`. Bulk downloads via Geofabrik. Completely **free** (ODbL license). Excellent Metro Vancouver data quality.

### A3.6 Local Logic

**18 Location Scores** on 10-point scale across 250M+ addresses (US + Canada): walkability, transit, cycling, car dependency, parks, groceries, restaurants, shopping, schools, daycares, health, entertainment, quiet, vibrant, nightlife, diversity. 100B+ data points. REST API. Enterprise pricing.

### A3.7 Property Photo Analysis APIs

**Restb.ai** — AI property condition scoring (Disrepair/Poor/Average/Good/Excellent/Luxury), sub-scores for kitchen/bathroom/interior/exterior, room type detection, amenity detection. Enterprise pricing.

**HelloData QualityScore** — Photo-based condition/quality scores, room-type detection, amenity extraction. **$0.01 per image.** REST API.

**Repliers Property Photo Insights** — AI photo analysis for Canadian MLS listings. Included with Repliers subscription.

**Impact:** AVMs see **up to 18% decrease in MAE** when incorporating automated condition/quality scores.

### A3.8 School Data Sources

**Fraser Institute Rankings** — Report Card rankings for 1,015+ BC schools based on 8 academic indicators from FSA results. Overall score 0-10, percentile, 5-year trend. At `compareschoolrankings.org`. Free. No API (scraping required).

**BC Ministry of Education** — School locations, district boundaries, enrollment via BC Data Catalogue. Free.

**City of Vancouver** — Elementary and secondary school catchment area boundaries as GeoJSON/SHP on Open Data portal. Free.

### A3.9 Crime Data Sources

**VPD GeoDASH** — Geocoded crime incidents for Vancouver, 2003-2026. Updated weekly (Sundays). CSV bulk download at `geodash.vpd.ca/opendata/`. Property crime to hundred-block level. Free.

**Statistics Canada** — Table 35-10-0184-01: Incident-based crime stats for all BC police services, 1998-2024. Covers suburban municipalities. Free via WDS API.

**ICBC Crash Data** — Vehicle crash locations, severity, pedestrian/cyclist involvement. 5 years of data. Tableau dashboards and CSV. Free.

### A3.10 Environmental and Climate Data

**Environment Canada MSC GeoMet API** — Weather, climate normals, Air Quality Health Index. OGC WMS/WFS. Free.

**Metro Vancouver Air Quality** — Real-time PM2.5, PM10, O3, NO2, SO2 from monitoring network. Interactive map. Free.

**YVR Noise** — WebTrak flight tracking and noise levels. Directly affects property values in Richmond, South Vancouver, parts of Burnaby. Free.

**Sea Level Rise Projections:**
- Vancouver: ~50cm by midcentury, ~1m by 2100
- 1m rise = 13 km2 in floodplain, ~$30B in property value affected
- Annual flood damages projected $510-820M by 2070-2100
- Data from City of Vancouver Open Data (Designated Floodplain), Climate Central Risk Finder, NRCan Flood Mapping. All free.

### A3.11 Geotechnical and Seismic Risk

**Metro Vancouver Seismic Microzonation Project (MVSMMP)** — 29 detailed seismic hazard maps:
- Earthquake shaking amplification
- Liquefaction susceptibility (probabilistic)
- Landslide hazard potential
- Geodatabase: 15,000+ geotechnical data points, 797 shear wave velocity profiles, 1,389 CPT tests, 10,000+ stratigraphic logs

At `metrovanmicromap.ca`. Free. Critical for Richmond, False Creek, Delta properties.

### A3.12 Satellite and Aerial Imagery

**Google Earth Engine** — 90+ PB of imagery (Landsat 30m, Sentinel-2 10m). Compute NDVI, land use classification, green space coverage. Free for non-commercial. Paid for commercial.

**Planet Labs** — Daily 3m imagery (PlanetScope), 50cm (SkySat). Enterprise pricing. Construction monitoring, change detection.

**Metro Vancouver LiDAR 2022** — High-res elevation data. View analysis (ocean/mountain view premiums), flood risk, building heights. Free via Metro Vancouver Open Data.

### A3.13 Economic Indicators

**Bank of Canada Valet API** — Policy rate, mortgage rates (all terms), prime rate, bond yields, exchange rates, inflation. Historical series going back decades. **Completely free, no auth required.** JSON/CSV/XML. Key series: `V80691335` (5-yr mortgage), `V80691311` (prime rate).

**BC Stats** — Provincial/regional GDP, employment, wages, CPI, building permits. Free CSV/Excel downloads.

**Conference Board of Canada** — CMA-level 5-year forecasts with 100+ indicators. Enterprise pricing.

**Statistics Canada Web Data Service** — Census, Labour Force Survey, NHPI, CPI, building permits, GDP. Free SDMX/JSON API.

### A3.14 Immigration and Demographic Projections

**BC Stats** — Population projections by age cohort, regional district level. Metro Vancouver: ~42,500 residents/year, reaching 4.1M by 2050. 81%+ of BC immigrants settle in Metro Vancouver. Free.

**IRCC** — Immigration levels plans, PR admissions by destination, temporary resident data. Open data. Free.

**Metro Vancouver Growth Projections** — Municipality-level population, dwelling, employment projections. 25.2% growth projected 2024-2046. Free.

### A3.15 Construction Cost Indices

**Statistics Canada BCPI** — Building construction price indexes by type and division. Quarterly since Q1 1981, 15 CMAs including Vancouver. Table 18-10-0289-01. Free.

**RSMeans** — 92,000+ unit costs, updated quarterly. City Cost Indexes for 8 Canadian regions. Starting $2,268/year CAD.

**Altus Group Construction Cost Guide** — Industry-standard Metro Vancouver benchmarks: low-rise wood $250-350/sqft, mid-rise concrete $350-500/sqft, high-rise concrete $450-650/sqft, luxury high-rise $600-900+/sqft. Annual guide free with registration.

### A3.16 Building Energy Benchmarking

**City of Vancouver Energy Reporting** — Large buildings must report EUI and GHG emissions via ENERGY STAR Portfolio Manager since 2021.

**Building Benchmark BC** — 1,900+ participating buildings, 15M+ sqm. Voluntary public disclosure.

**BC Energy Step Code** — Steps 1-5 (5 = net-zero ready) for new construction. Data embedded in building permits.

---

## A4. Regulatory, Legal, and Strata Data — Expanded

### A4.1 Strata-Specific Data

**Form B Information Certificate** — The single most important strata due diligence document. Includes:
- Monthly strata fees and what they cover
- Contingency reserve fund balance
- Outstanding/planned special levies
- Current lawsuits
- Bylaw violations and complaints
- Insurance coverage (deductibles, premiums)
- Units rented vs. owner-occupied
- Parking/storage allocation
- Building envelope issues/remediation history
- Council-approved expenditures not yet billed

Cost: Max $35 statutory + $0.25/page for attachments. Total package often $100-300. Ordered through strata management company.

**Depreciation Reports** — No centralized public registry. Held by individual strata corporations. Since July 2024, waiver option eliminated for most stratas (5+ units). Reports cover 30-year component repair/replacement projections. Commissioning costs $5,000-$25,000 per building.

**eStrataHub** (Dye & Durham) — Leading platform for ordering strata documents electronically: Form F, Form B, minutes, bylaws, financials, depreciation reports. Per-document fees.

**Civil Resolution Tribunal (CRT)** — Searchable database of strata dispute decisions at `civilresolutionbc.ca`. Dispute types: strata fees, bylaw enforcement, common property, unfair actions. Searchable by strata plan number. Free. **Buildings with multiple CRT complaints may indicate governance problems — a unique pricing signal.**

**CHOA (Condominium Home Owners Association of BC)** — Case law digests, depreciation report benchmarks, insurance cost bulletins. Basic resources free, membership ~$25-50/year.

### A4.2 Tax Data Sources

**BC Property Transfer Tax** — Aggregate volumes/values published monthly via BC Data Catalogue. Municipality-level. Free CSV/XLSX. Individual transaction-level PTT data NOT publicly available.

**Foreign Buyer Tax Data** — BC Ministry of Finance quarterly reports: transactions by region, total value, percentage of market. Foreign buyer share dropped from ~10% (2016) to ~1-2% post-tax. Free.

**Speculation and Vacancy Tax (SVT)** — Annual reports: declarations by municipality, properties taxed, revenue collected, exemptions. Free.

**Empty Homes Tax (Vancouver)** — Annual reports: occupied/exempt/vacant declarations, properties deemed empty, revenue, neighborhood breakdown. Free via Vancouver Open Data.

**BC Home Flipping Tax** — 20% on net income within 365 days, declining to 0% at 730 days. Exemptions for life changes. Effective January 2025.

**Property Tax Rates** — All 21 Metro Vancouver municipalities publish annual mill rates. Historical rates (5-10 years) available. Key source: BC Ministry of Municipal Affairs "Local Government Statistics." Free.

**Tax Sale Data** — Annual tax sale property lists per municipality (typically September-October). Minimum bids, property details. Small volumes (5-20 per municipality) but indicate localized financial stress. Free.

### A4.3 Development and Planning Data

**Metro Vancouver Regional Growth Strategy (Metro 2050)** — Urban Containment Boundary maps, regional land use designations, growth projections, Frequent Transit Development Areas. The UCB fundamentally constrains land supply and is the single largest structural driver of Metro Vancouver land values. Free.

**Municipal OCPs — GIS Access:**
- Vancouver: VanMap (full GIS + Open Data)
- Burnaby: GIS portal
- Surrey: COSMOS (comprehensive, 200+ layers)
- Richmond: Richmond Interactive Map
- Coquitlam: QtheMap
- North Vancouver District: GeoWeb
- New Westminster: CityViews Map

**Rezoning Application Databases** — Active/completed rezoning applications, proposed density changes, staff reports. Real-time updates. Free per municipality website.

**Development Permit Tracking** — Vancouver DBLOL, Surrey via Open Data, most municipalities have online portals. Free.

**Community Amenity Contributions (CACs)** — Policy documents per municipality. Specific amounts in rezoning staff reports. Typically $30-100+/buildable sqft in Vancouver. Free.

**Development Cost Levies (DCLs)** — Rate schedules by area and use, updated annually. Vancouver: ~$30-180+/sqft depending on area/use. Free via municipal bylaws.

**Bill 44 (Small-Scale Multi-Unit Housing)** — Effective June 30, 2024. 3 units on lots <280m2, 4 units on lots >=280m2, 6 units near transit. Implementation bylaws per municipality. **Fundamentally changes development potential of every residential lot.** Free.

**Bill 47 (Transit-Oriented Development Areas)** — Within 200m of transit station: 20+ storeys. 200-400m: 12+ storeys. 400-800m: 8+ storeys. **Properties within 800m of designated stations may see land values increase 50-300%.** Free.

### A4.4 Court and Legal Data

**BC Supreme Court** — Foreclosure orders, partition/sale orders, builder's liens, real estate fraud. Via Court Services Online (`justice.gov.bc.ca/cso`). CanLII (`canlii.org`) for free case law search. No bulk API.

**BC Sheriff's Sales** — Properties seized for debt enforcement. Low volume, extreme distress indicator. Free.

**Bankruptcy/Insolvency** — OSB aggregate statistics by district at `ic.gc.ca`. Individual searches free. Quarterly aggregates.

### A4.5 Heritage Data

**Vancouver Heritage Register** — Heritage-designated and -registered properties. Categories A/B/C. Heritage Conservation Areas, Heritage Revitalization Agreements. Searchable on VanMap. Free.

**First Nations Reserve/Treaty Lands** — Reserve boundary GIS from Indigenous Services Canada. Musqueam leasehold properties trade at **30-60% discounts** versus freehold. Tsawwassen treaty lands with development authority. Free GIS data.

### A4.6 Infrastructure Data

**TransLink Future Transit Plans** — Transport 2050, SkyTrain extensions (Surrey-Langley ~2029, UBC under planning), BRT plans. Station locations, alignment maps, ridership projections. Free.

**Telecommunications** — CRTC broadband availability maps, Telus PureFibre coverage, Rogers cable coverage. Fiber availability increasingly a pricing factor. Free.

### A4.7 Real Estate Industry Reports and Indices

**Teranet-National Bank HPI** — Repeat-sales methodology. Most accurate Canadian HPI. Monthly for Vancouver CMA. Free (headline). Enterprise for granular data.

**Royal LePage House Price Survey** — Quarterly. Free.

**UDI (Urban Development Institute) Pacific Region** — Presale market data, absorption rates, construction cost trends, government charges analysis. Some free, full access via membership.

**Altus Group Construction Cost Guide** — Annual benchmarks. Free with registration.

**Urbanation** — Pre-construction condo tracking. Expanding to Vancouver. Institutional subscription $5,000-15,000/year.

### A4.8 Mortgage and Lending Data

**OSFI B-20 Guideline** — Stress test: greater of contract rate + 2% or 5.25%. Max GDS 39% (insured), TDS 44%. Max amortization: 25yr insured, 30yr uninsured (extended for first-time/new builds). Potential LTI cap at 4.5x under evaluation. Free.

**Canadian Bankers Association** — Mortgage arrears by province, average balance, renewal projections, fixed/variable split. **Renewal tsunami**: borrowers who locked at 1.5-2% in 2020-21 face 4-5%+ at renewal. Quarterly. Free.

**Bank of Canada Financial System Review** — Household debt-to-income, mortgage debt growth, housing vulnerability assessments, stress scenarios. Annual + quarterly Monetary Policy Report. Free.

---

## A5. Data Source Priority Matrix

### Tier 1 — Essential (Must Have for MVP)

| Source | Data | Cost | Access | Impact |
|--------|------|------|--------|--------|
| BC Assessment Data Advice | Property characteristics, assessed values, sales | Paid (negotiated) | CSV/XML bulk | CRITICAL |
| GVR BridgeAPI (MLS) | Listing + sold data | ~$1,500/yr VOW | RESO Web API | CRITICAL |
| City of Vancouver Open Data | Zoning, permits, parcels, tax | Free | REST API | HIGH |
| TransLink GTFS | Transit accessibility | Free | GTFS files | HIGH |
| Walk Score API | Walkability/Transit/Bike scores | Free (5K/day) | REST API | HIGH |
| Statistics Canada | Census, demographics, income | Free | SDMX API | HIGH |
| CMHC | Starts, completions, vacancy, rents | Free | R package + portal | HIGH |
| ParcelMap BC | Parcel boundaries, PIDs | Free | WMS/WFS | HIGH |
| Bank of Canada Valet API | Interest rates, mortgage rates | Free | REST API | HIGH |

### Tier 2 — High Value (Significant Accuracy Improvement)

| Source | Data | Cost | Access | Impact |
|--------|------|------|--------|--------|
| Fraser Institute Rankings | School quality scores | Free | Web scraping | HIGH |
| VPD GeoDASH + StatCan Crime | Crime data | Free | CSV download | HIGH |
| Bill 44/47 implementation | Development potential | Free | Municipal sites | HIGH |
| Municipal OCP/zoning GIS | Land use designations | Free | GIS portals | HIGH |
| OSFI stress test parameters | Purchasing power | Free | Published guidelines | HIGH |
| Teranet HPI | Price index benchmark | Free (index) | PDF/web | MEDIUM-HIGH |
| HelloData QualityScore | Photo condition scoring | $0.01/image | REST API | MEDIUM-HIGH |
| OpenStreetMap/Overpass | Amenity density | Free | Overpass QL | MEDIUM-HIGH |
| BC Data Catalogue (ALR, flood, contamination) | Environmental constraints | Free | WMS/WFS | MEDIUM-HIGH |

### Tier 3 — Moderate Value (Enhanced Features)

| Source | Data | Cost | Access | Impact |
|--------|------|------|--------|--------|
| Local Logic | 18 location scores | Paid (enterprise) | REST API | MEDIUM |
| Mapbox Isochrone | Travel time polygons | ~$2/1K | REST API | MEDIUM |
| Seismic Microzonation | Liquefaction/earthquake risk | Free | GIS layers | MEDIUM |
| Surrey/Burnaby/Richmond Open Data | Suburban property data | Free | Various | MEDIUM |
| NRCan Building Footprints/DEM | Building heights, elevation | Free | GeoTIFF/GeoJSON | MEDIUM |
| CRT Strata Disputes | Building governance quality | Free | Web search | MEDIUM |
| Metro Vancouver LiDAR | View analysis, flood risk | Free | Open Data Portal | MEDIUM |
| ICBC Crash Data | Road safety | Free | CSV/dashboard | MEDIUM |
| Sea Level Rise / Flood Data | Climate risk | Free | GeoJSON/SHP | MEDIUM |

### Tier 4 — Supplementary / Experimental

| Source | Data | Cost | Access | Impact |
|--------|------|------|--------|--------|
| Houski API | Alternative property data | $0.015/req | REST API | MEDIUM |
| Buildify | Presale/new construction | Paid | REST API | LOW-MEDIUM |
| Google Earth Engine | Vegetation, land use change | Free (research) | Python/JS API | LOW-MEDIUM |
| Altus Group Data Studio | Commercial RE data | Paid (enterprise) | API | LOW-MEDIUM |
| BC Stats / Conference Board | Economic forecasts | Free-Paid | CSV/API | LOW-MEDIUM |
| StatCan BCPI | Construction cost trends | Free | CSV/API | LOW |
| Building Energy Benchmarking | EUI, GHG emissions | Free | Web | LOW |
| Reddit/Social Sentiment | Neighborhood perception | Free-Paid | API | LOW |
| RSMeans | Granular construction costs | $2,268+/yr | Platform | LOW |
| Cell Tower Data | Connectivity proxy | Free | ISED/OpenCelliD | VERY LOW |

---

## A6. Additional Feature Engineering Variables Identified Through Research

The following additional features were identified beyond the original 80-120 variable set:

### Strata-specific (from Form B / depreciation reports)
- CRT dispute count (per building, trailing 5 years)
- Strata management company quality tier
- Owner-occupied vs. rented unit ratio
- Active litigation flag
- Insurance premium per unit (annual)
- Contingency reserve fund balance per unit
- Years since last special assessment
- Depreciation report funding adequacy score

### Regulatory and policy
- Bill 44 eligible unit count (based on lot size + transit proximity)
- Bill 47 TOD area tier (200m/400m/800m from station)
- OCP future land use designation code
- Active rezoning application within 500m flag
- DCL rate applicable to property
- CAC rate applicable (if in density bonus zone)
- Heritage Conservation Area flag
- First Nations leasehold flag + remaining term
- BC Home Flipping Tax exposure (months since purchase)

### Environmental and risk
- Seismic liquefaction susceptibility score
- Sea level rise exposure zone (current/2050/2100)
- Air quality index (nearest monitoring station average)
- YVR flight path noise zone flag
- Environmental remediation site proximity
- NDVI (vegetation index) within 500m radius
- Building energy performance score (EUI where available)

### Market and economic
- Mortgage renewal tsunami exposure (% of area mortgages due for renewal within 12 months)
- Immigration flow trend (12-month trailing by CMA)
- Employment growth by census tract (trailing 12 months)
- Rental vacancy rate (nearest CMHC zone)
- Presale inventory within 1km (unsold units)
- Tax sale frequency in sub-area (trailing 3 years)
- Foreign buyer share trend by municipality
- SVT/EHT vacancy declaration rate by neighborhood

### Photo and condition-based
- HelloData/Restb.ai overall condition score
- Kitchen quality sub-score
- Bathroom quality sub-score
- Interior finish quality sub-score
- Exterior condition sub-score
- Renovation probability score (based on permit history + age)

---

## A7. Key Non-Obvious Insights for Model Architecture

### Insight 1: Transit-Wealth Interaction Effect
A naive "distance to SkyTrain" feature will mislead the model. In wealthy West Side neighborhoods, proximity to SkyTrain **depresses** values (unwanted density effect), while East Side proximity **boosts** values. The model must include an interaction term: `transit_distance × neighborhood_median_income`. This single interaction term can improve MAPE by 0.5-1% in affected areas.

### Insight 2: Supply Pipeline Asymmetry
Massive rental supply (Senákw 6,000 units) primarily pressures **condo values and rents** while leaving **detached home land values** largely unaffected. The supply impact index must be property-type-aware, not a single number per geographic cell. Build separate supply impact indices for rental, strata condo, townhome, and detached segments.

### Insight 3: Leasehold Pricing Cliff
The leasehold discount is non-linear: ~0% above 70 years remaining, modest 5-10% at 40-70 years, then **accelerating sharply** below 30 years as it crosses the typical mortgage amortization threshold. Below 25 years, most banks refuse financing entirely, creating a near-binary cliff. This requires a custom discount curve, not a linear feature.

### Insight 4: Bill 44/47 Creates a Density Floor
Every residential lot in Metro Vancouver now has a minimum density potential (3-6 units depending on size/transit proximity). This creates a **density floor value** that may exceed the property's value as a single-family home, particularly for lots near transit. The model should compute residual land value under maximum Bill 44/47 density and use `max(comparable_value, density_floor_value)`.

### Insight 5: Strata Insurance as Hidden Value Destroyer
The strata insurance crisis (premiums up 50-780%) creates a pricing signal that most AVMs miss entirely. A building with $500K annual insurance vs. $100K translates to ~$200-400/month per unit in strata fee differences, reducing qualifying mortgage by $30,000-60,000 per unit. This data is only available from Form B certificates — building a database of per-building insurance costs from accumulated Form Bs creates a significant competitive moat.

### Insight 6: Photo Condition Scoring ROI
At $0.01/image (HelloData), processing all MLS listing photos costs roughly $50-100 per 5,000-10,000 images. Research shows this can reduce MAE by up to 18%. This is the single highest ROI data source that most AVMs do not incorporate.

### Insight 7: CRT Disputes as Building Quality Signal
A building with 5+ CRT disputes in the past 3 years likely has governance problems, deferred maintenance, or chronic neighbor conflicts — all of which depress prices. This free, publicly searchable database provides a unique signal no competitor AVM uses.

### Insight 8: Presale Inventory as Market Cycle Indicator
Metro Vancouver currently has ~2,500-3,500 unsold new condos, the highest in 24 years. Tracking presale inventory levels (via Buildify or UDI data) and developer incentive packages (flash sales, assignment restrictions loosened) provides a forward-looking market cycle indicator. When developers offer 25% discounts plus closing cost packages, it signals significant market weakness in the condo segment.
