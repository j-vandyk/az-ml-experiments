> **Archived.** Extended catalog of 13 aspirational predictor domains (satellite, financial markets, displacement, digital, etc.) beyond the current pipeline. A summary lives in `../data-and-predictors.md` §6. Promote a domain back into the active reference (and add a numbered `01_data_pull/` notebook) once it is actually pulled.

---

# Extended Predictor & Dataset Catalog
## Nation-State Instability Forecasting — Sources Beyond the Standard Political Science Toolkit

**Compiled:** April 2026
**Scope:** Datasets and predictor types not covered in `# Dataset & Variable Synthesis.md`. Organized by domain. Each entry includes: provider, coverage, access model, key variables, operationalization for ML, and relevant studies.

---

## HOW TO READ THIS DOCUMENT

The existing synthesis (`# Dataset & Variable Synthesis.md`) covers the core political science datasets: ACLED, GDELT, UCDP, Polity V, V-Dem, PITF, WDI, WGI, FAO, EPR, PRIO-GRID, Twitter signals, LDA topics, UNHCR, and ITU mobile penetration.

This document covers everything else — ten additional domains that expand predictor coverage into satellite imagery, financial markets, health, humanitarian operations, environmental stress, arms flows, international relations signals, civil society mobilization, Africa-specific governance, and subnational granularity.

---

## DOMAIN 1: Satellite & Remote Sensing

### VIIRS Nighttime Lights (Day/Night Band)
**Provider:** NOAA / NASA | **Coverage:** Global, 2012–present | **Frequency:** Monthly composites | **Access:** Free (NASA Earthdata)

Successor to DMSP-OLS with substantially higher resolution (750m vs. 2.7km) and better dynamic range. The existing synthesis mentions DMSP; VIIRS supersedes it for any work post-2012.

**Key products:**
- `VNP46A1`: Daily nighttime lights (cloud-masked)
- `VNP46A2`: Gap-filled daily
- `VNP46A3`: Monthly composite (most useful for country-month panel)

**ML operationalization:**
- Country-level mean radiance per month (proxy for economic activity)
- Month-over-month radiance change (economic shock detector — drops after conflict onset, sanctions, power grid destruction)
- Sub-national radiance Gini (spatial inequality proxy)
- City-level radiance as proxy for urban economic conditions
- Radiance anomaly score: deviation from 12-month rolling mean (sensitive to sudden disruption)

**Studies using nighttime lights as conflict predictor:** Bluhm & Krause (2018); Henderson, Storeygard & Weil (2012) — established lights as GDP proxy; multiple ViEWS grid-level models use `nlights_calib_mean` from PRIO-GRID (which is DMSP-derived — VIIRS version is the upgrade).

---

### Sentinel-1 SAR (Synthetic Aperture Radar)
**Provider:** ESA (European Space Agency) | **Coverage:** Global, 2014–present | **Frequency:** 6–12 day revisit | **Access:** Free (Copernicus Open Access Hub)

C-band SAR imagery penetrates clouds and works at night. Used in conflict research to detect:
- Building destruction (rubble changes backscatter signature)
- IDP camp emergence and expansion
- Agricultural abandonment (field texture changes)
- Military vehicle movement tracks

**ML operationalization:**
- Coherence change detection: SAR coherence between two dates drops when structures are destroyed
- Settlement area growth rate (IDP camp formation)
- Agricultural land use abandonment rate
- Port and border crossing activity (ship presence, convoy tracks)

**Studies:** UNOSAT (UN Satellite Centre) produces conflict damage assessments using SAR; Witmer & O'Loughlin (2011) early SAR conflict work; multiple humanitarian organizations use Sentinel-1 for IDP camp monitoring.

---

### MODIS / Landsat Vegetation (NDVI)
**Provider:** NASA | **Coverage:** Global, 1999–present (MODIS); 1972–present (Landsat) | **Frequency:** 16-day (MODIS) / 16-day (Landsat) | **Access:** Free (NASA Earthdata, Google Earth Engine)

**Key products:**
- `MOD13A3`: Monthly 1km NDVI composites
- `MOD09GA`: Daily surface reflectance

**ML operationalization:**
- Annual NDVI deviation from 10-year baseline (agricultural stress / crop failure)
- Growing season NDVI anomaly → food production shock predictor
- Conflict-induced agricultural abandonment (NDVI decline in conflict-affected grid cells)
- Forest loss rate (Hansen Global Forest Change dataset — linked to resource conflict)

**Studies:** Hsiang, Burke & Miguel (2013) meta-analysis linking climate/agricultural shocks to conflict; Maystadt & Ecker (2014) Somalia drought-conflict link; Harari & La Ferrara (2018) using NDVI anomaly as crop shock instrument for conflict.

---

### FIRMS (Fire Information for Resource Management System)
**Provider:** NASA | **Coverage:** Global, 2000–present | **Frequency:** Near-real-time (daily) | **Access:** Free

Active fire detections from MODIS and VIIRS. Relevant as conflict indicator because:
- Arson of villages, crops, and infrastructure is a common conflict tactic
- Fire hotspot clusters in non-agricultural contexts (no burning season) signal conflict activity
- Used by ACLED as corroborating evidence for some event codings

**ML operationalization:**
- Monthly fire count in non-agricultural grid cells
- Fire anomaly (count vs. seasonal baseline)
- Fire density in proximity to settlement areas

---

### WorldPop / HRSL (High Resolution Settlement Layer)
**Provider:** University of Southampton (WorldPop); Meta + CIESIN (HRSL) | **Coverage:** Global | **Frequency:** Annual | **Access:** Free

**WorldPop:** 100m gridded population estimates by age and sex. Disaggregates national census data to grid cells using satellite-derived settlement layers.

**HRSL:** 30m binary settlement mask (is there a human settlement in this pixel?) derived from Meta's analysis of Bing satellite imagery.

**ML operationalization:**
- Grid-cell population at risk (denominator for event rate normalisation)
- Distance to nearest settlement (geographic exposure)
- Population density change (rapid urbanisation stress)
- Youth population share at subnational level (youth bulge at grid level)

---

### GRACE / GRACE-FO (Groundwater and Water Storage)
**Provider:** NASA/DLR | **Coverage:** Global, 2002–present | **Frequency:** Monthly | **Access:** Free

Measures changes in Earth's gravity field caused by water mass redistribution — groundwater depletion, drought-driven aquifer decline, ice melt.

**ML operationalization:**
- Groundwater storage anomaly (months below historical mean → agricultural water stress)
- Long-term aquifer depletion trend (structural water insecurity, especially MENA, Sahel)

**Studies:** Kelley et al. (2015) Syrian conflict — groundwater depletion linked to drought and rural displacement; Maystadt et al. using water stress for African conflict.

---

## DOMAIN 2: Financial & Market Signals

### Sovereign Bond Spreads (EMBI / EMBIG)
**Provider:** JP Morgan (proprietary); proxies available via FRED, Bloomberg, World Bank | **Coverage:** ~70 emerging market countries | **Frequency:** Daily | **Access:** Bloomberg (paid); FRED provides some series free

**EMBI (Emerging Market Bond Index):** Spread between EM sovereign bond yield and comparable US Treasury yield. Higher spread = higher perceived default/instability risk.

**ML operationalization:**
- Monthly mean spread (market's real-time instability probability)
- Spread volatility (uncertainty)
- Spread spike: month-over-month change exceeding 2 standard deviations
- Spread inversion relative to regional peers

**Why valuable:** Bond markets are forward-looking; spread widening often precedes political crises by weeks to months. CoupCast and several IMF forecasting papers have noted this lag structure.

**Studies:** IMF Barrett et al. (2021) uses financial variables; Bekaert, Harvey & Lundblad (2005) on financial liberalisation and risk.

---

### Currency Exchange Rate & Volatility
**Provider:** BIS (Bank for International Settlements), central banks, FRED | **Coverage:** Global | **Frequency:** Daily | **Access:** Free (BIS, FRED)

**Key variables:**
- `fx_rate_usd`: Official exchange rate (domestic currency per USD)
- `fx_black_market_premium`: Black market rate / official rate − 1 (where available; Reinhart & Rogoff dataset)
- `fx_volatility_30d`: Rolling 30-day standard deviation of daily returns
- `fx_depreciation_ytd`: Year-to-date depreciation vs. USD

**ML operationalization:**
- Currency crisis flag: depreciation > 25% in 6 months (Laeven & Valencia IMF database)
- Black market premium as hidden capital flight indicator
- EMBERS used daily exchange rate as one of its LASSO sub-model inputs

**Studies:** EMBERS (Ramakrishnan 2014) — daily exchange rate was a top LASSO predictor for Latin American unrest.

---

### IMF Balance of Payments & Reserves
**Provider:** IMF | **Coverage:** Global | **Frequency:** Quarterly (some monthly) | **Access:** Free (IMF Data)

**Key variables:**
- `reserves_months_imports`: International reserves in months of import coverage (< 3 months = crisis threshold)
- `ca_balance_gdp`: Current account balance (% GDP)
- `imf_program_active`: Binary — country has active IMF structural adjustment program
- `imf_program_type`: Type of IMF program (SBA, EFF, ECF, etc.)
- `imf_disbursement_delayed`: Flag if IMF disbursement suspended or delayed

**ML operationalization:**
- Low reserves flag (< 3 months) as economic vulnerability predictor
- IMF program entry as crisis indicator (governments typically seek IMF programs when they cannot finance themselves)
- Program suspension as political stress signal (government unable/unwilling to meet conditionality)

**Studies:** IMF WP Barrett et al. (2021) incorporates several balance of payments variables; multiple ILC studies use IMF program status.

---

### World Bank CPIA (Country Policy and Institutional Assessment)
**Provider:** World Bank | **Coverage:** ~75 IDA-eligible countries | **Frequency:** Annual | **Access:** Free (World Bank Data)

Internal World Bank assessment of policy quality across 16 criteria grouped into 4 clusters:
1. Economic management (monetary, fiscal, debt)
2. Structural policies (trade, financial sector, business)
3. Policies for social inclusion (gender, equity, building human resources)
4. Public sector management and institutions (property rights, accountability, corruption)

**ML operationalization:**
- CPIA cluster scores (1–6) as governance quality predictors at annual frequency
- CPIA decline of >0.5 points as governance deterioration signal
- CPIA composite as fragile state identifier

---

### Laeven & Valencia Currency, Banking, and Sovereign Debt Crisis Database
**Provider:** IMF (Laeven & Valencia) | **Coverage:** Global, 1970–2017 | **Frequency:** Annual event list | **Access:** Free (IMF Working Paper)

Codes three types of financial crises at country-year level:
- Systemic banking crises
- Currency crises (>30% depreciation + acceleration)
- Sovereign debt crises (default or restructuring)

**ML operationalization:**
- Binary flags for each crisis type (lagged as predictors)
- Crisis co-occurrence (twin or triple crises) as extreme stress indicator
- Years since last financial crisis (time-since-last pattern like conflict history)

**Studies:** Used as predictor in IMF WP Barrett et al. (2021); financial crisis → unrest link documented in several studies.

---

### Reinhart & Rogoff "This Time Is Different" Dataset
**Provider:** Reinhart & Rogoff (Harvard / Peterson Institute) | **Coverage:** 66 countries, 1800–2010 | **Frequency:** Annual event list | **Access:** Free (authors' website)

Historical record of sovereign defaults, banking crises, inflation crises, currency crashes, and stock market crashes.

**ML operationalization:**
- Sovereign default history flag (prior default increases subsequent default and instability risk)
- Serial defaulter classification
- Debt overhang indicator

---

## DOMAIN 3: Health, Pandemic & Humanitarian Stress

### WHO Global Health Observatory / Disease Surveillance
**Provider:** WHO | **Coverage:** Global | **Frequency:** Annual / event-based | **Access:** Free (WHO GHO API)

**Key variables:**
- `who_emergency_alert`: Binary — WHO Grade 1/2/3 emergency declared for country
- `cholera_cases_per_100k`: Annual cholera incidence (proxy for water/sanitation collapse, strongly correlated with conflict)
- `measles_cfr`: Measles case fatality rate (proxy for health system collapse)
- `polio_cases`: Polio case counts (proxy for vaccination system breakdown — often first to collapse in conflict)
- `malaria_deaths_per_100k`: Malaria mortality rate

**ML operationalization:**
- WHO emergency grade as state capacity failure indicator
- Cholera incidence as water system / governance collapse signal
- Vaccination coverage decline as state service delivery failure

**Studies:** Ghobarah, Huth & Russett (2003) conflict-health nexus; Østby (2013) health inequality and conflict; several ViEWS structural model components use health indicators.

---

### FEWS NET (Famine Early Warning Systems Network)
**Provider:** USAID / FEWS NET consortium | **Coverage:** ~30 food-insecure countries (primarily Africa, Middle East, South/Central Asia) | **Frequency:** Monthly | **Access:** Free (fews.net)

**Key outputs:**
- `ipc_phase`: IPC (Integrated Food Security Phase Classification) country or region level (1=Minimal, 2=Stressed, 3=Crisis, 4=Emergency, 5=Famine)
- `ipc_population_phase3plus`: Number of people in IPC Phase 3+ (Crisis or worse)
- `food_security_outlook`: 4-month forward food security projection (Likely Improved / No Change / Likely Deteriorated)
- `remote_area_access`: Binary flag for humanitarian access constraints

**ML operationalization:**
- IPC Phase 3+ population as food insecurity severity predictor
- Deteriorating outlook flag as leading indicator (FEWS NET analysts update monthly based on market prices, rainfall, and conflict)
- IPC phase transitions (1→3 = shock event)
- Food security stress combined with ACLED protest data = unrest escalation predictor

**Studies:** Barnett (2009) food security-conflict nexus; Fjelde & von Uexkull (2012) food price shocks and communal conflict in Sub-Saharan Africa.

---

### OCHA Financial Tracking Service (FTS)
**Provider:** UN OCHA | **Coverage:** ~50 humanitarian response countries | **Frequency:** Annual / quarterly | **Access:** Free (fts.unocha.org API)

Tracks humanitarian funding requests and contributions globally.

**Key variables:**
- `humanitarian_appeal_total_usd`: Size of humanitarian appeal (reflects assessed severity)
- `funding_gap_pct`: Percentage of humanitarian appeal unfunded
- `ocha_flash_appeal`: Binary — flash appeal launched (indicates sudden-onset crisis)
- `cluster_funding`: Funding by sector (food, health, shelter, WASH, protection)

**ML operationalization:**
- Large unfunded gap + active conflict = stress amplifier
- Flash appeal launch as sudden-onset crisis flag
- Appeal size normalized by population as crisis severity indicator

---

### ACAPS Crisis Severity Index
**Provider:** ACAPS (Assessment Capacities Project) | **Coverage:** ~60 countries in crisis | **Frequency:** Quarterly | **Access:** Free (acaps.org)

Composite severity score across five dimensions: people affected, access constraints, humanitarian conditions, complexity, and phase of crisis.

**ML operationalization:**
- ACAPS severity score as crisis intensity baseline
- Severity score change as escalation/de-escalation indicator
- Access constraint score as proxy for state territorial control

---

### EM-DAT (Emergency Events Database)
**Provider:** CRED (Centre for Research on the Epidemiology of Disasters), UCLouvain | **Coverage:** Global, 1900–present | **Frequency:** Event-based | **Access:** Free (registration required; emdat.be)

Every recorded natural and technological disaster globally: earthquakes, floods, droughts, storms, epidemics, industrial accidents, transport accidents.

**Key variables per event:**
- `dis_type`: Disaster type (Drought, Flood, Earthquake, Storm, Epidemic, etc.)
- `start_date`, `end_date`
- `total_deaths`, `total_affected`, `total_damage_usd`
- `country`, `subregion`

**ML operationalization:**
- Disaster deaths per capita in prior 12 months (state capacity stress)
- Disaster damage (% GDP) as economic shock
- Drought event flag (combined with food price data for famine precursor)
- Disaster frequency trend (climate stress intensification)
- Post-disaster political instability window (coups often follow major disasters — "disaster shock" hypothesis)

**Studies:** Multiple ViEWS structural predictors include EM-DAT disaster variables; Eastin (2016) natural disasters and conflict onset.

---

## DOMAIN 4: Displacement & Migration (Beyond UNHCR Totals)

### IDMC Global Internal Displacement Database
**Provider:** IDMC (Internal Displacement Monitoring Centre) | **Coverage:** Global | **Frequency:** Annual (GRID report) + event-level | **Access:** Free (internal-displacement.org)

Distinguishes between conflict-induced and disaster-induced internal displacement. More granular than UNHCR IDP totals.

**Key variables:**
- `idps_conflict_stock`: Total IDPs from conflict at year-end (stock)
- `idps_conflict_new`: New displacements from conflict in year (flow)
- `idps_disaster_new`: New displacements from disasters in year
- `idps_conflict_duration_months`: Duration of displacement episode

**ML operationalization:**
- New displacement flow as leading conflict escalation indicator (population fleeing = escalation signal)
- Displacement as % of population (severity normalizer)
- Displacement velocity: new IDPs / prior month stock (acceleration signal)

---

### IOM Displacement Tracking Matrix (DTM)
**Provider:** IOM (International Organization for Migration) | **Coverage:** ~50 countries (primarily fragile/conflict-affected) | **Frequency:** Monthly (where active) | **Access:** Free (dtm.iom.int)

Field-based tracking of IDP and returnee movements in countries with active operations. More granular and timely than IDMC for countries with DTM presence.

**Key variables:**
- `idp_sites_count`: Number of displacement sites
- `idp_population`: Population in tracked sites
- `returnee_flow_monthly`: Monthly returnee movements (improving security signal)
- `assessed_location_count`: Coverage of assessment (data quality indicator)

**ML operationalization:**
- Site count growth rate as conflict expansion indicator
- Returnee flow as conflict de-escalation signal
- DTM coverage as proxy for humanitarian access (indirectly indicates state control)

---

### World Bank Migration and Remittances Data
**Provider:** World Bank | **Coverage:** Global | **Frequency:** Annual | **Access:** Free (World Bank Data)

**Key variables:**
- `remittances_inflows_usd`: Total remittance inflows (USD millions)
- `remittances_pct_gdp`: Remittances as % of GDP
- `remittances_per_capita`: Per capita remittances
- `migration_stock_abroad`: Total emigrants abroad

**ML operationalization:**
- Remittances % GDP as economic resilience indicator (household income buffer during shocks)
- Remittance decline as economic stress signal (often drops before formal GDP indicators)
- High remittance dependence as structural vulnerability (shock transmission channel from diaspora country)
- Brain drain proxy (high emigration rate → human capital depletion)

---

### UNHCR Refugee Data Finder (More Granular Than Totals)
**Provider:** UNHCR | **Coverage:** Global | **Frequency:** Annual (with some monthly operational data) | **Access:** Free (data.unhcr.org)

Beyond the total refugee counts in the existing synthesis, UNHCR provides:
- **Origin-destination matrices:** Which countries are sending refugees to which
- **Refugee recognition rates:** % of asylum claims accepted (proxy for legitimacy of fear)
- **Return movements:** Voluntary returns (de-escalation signal)
- **Stateless persons:** Population with no recognized nationality (governance failure proxy)
- **Persons of concern by situation:** Linked to specific named crises

**ML operationalization:**
- Origin country refugee outflow rate as conflict intensity indicator (lagged 0–6 months after onset)
- Destination country burden ratio (refugees / GDP) as spillover stress
- Return rate as peace consolidation indicator

---

## DOMAIN 5: Telecommunications, Digital & Information Environment

### NetBlocks Internet Shutdown Events
**Provider:** NetBlocks (netblocks.org) | **Coverage:** Global (events-based) | **Frequency:** Event-based | **Access:** Free reports; structured data via API/partnership

Documents internet shutdowns, platform blockages, and throttling events globally. Each event includes: country, date, duration, affected platforms, type (total blackout / platform block / throttle), stated reason (if given).

**ML operationalization:**
- Monthly binary: internet shutdown occurred (binary)
- Shutdown severity score (total blackout = 3, platform block = 2, throttle = 1)
- Shutdown frequency in prior 12 months (authoritarian information control trend)
- Platform-specific blocks (Twitter/X, Facebook, WhatsApp shutdowns as protest control signal)

**Studies:** Rød & Weidmann (2015) internet as protest facilitator; Deibert et al. (Citizen Lab) internet censorship and autocratization.

---

### Freedom House Freedom on the Net
**Provider:** Freedom House | **Coverage:** ~70 countries | **Frequency:** Annual | **Access:** Free (freedomhouse.org)

Annual country scores across three dimensions: obstacles to access, limits on content, violations of user rights.

**Key variables:**
- `fotn_total_score`: Total score (0–100, higher = less free)
- `fotn_obstacles_access`: Score for physical/economic/legal barriers to internet access
- `fotn_limits_content`: Score for censorship, content blocking, manipulation
- `fotn_violations_rights`: Score for surveillance, prosecution of users, attacks on journalists

**ML operationalization:**
- Score trajectory (worsening = autocratization signal)
- Sharp score decline (crackdown event)
- Combined with VPN usage data: state censorship intensification before political events

---

### Meta (Facebook) Data for Good — Movement & Conflict
**Provider:** Meta (partnership required) | **Coverage:** Countries with significant Facebook usage | **Frequency:** Daily | **Access:** Research partnership (data.humdata.org for humanitarian subset)

Products relevant to instability forecasting:
- **Facebook Disaster Maps:** Population movement before/after disasters and conflict
- **Social Connectedness Index (SCI):** Cross-country social network density (migration and diaspora network proxy)
- **Crisis Response data:** Engagement with crisis-related content

**ML operationalization:**
- Population movement velocity (displacement early warning)
- Intra-country movement patterns (internal displacement detection)
- Cross-border connection density as contagion pathway predictor

---

### Wikipedia Edit Activity
**Provider:** Wikimedia Foundation | **Coverage:** Global (language-specific) | **Frequency:** Real-time | **Access:** Free (Wikipedia API, Wikimedia dumps)

Edit spikes on country/conflict/political crisis articles in Wikipedia have been shown to precede or coincide with conflict events. EMBERS used Wikipedia as one of its data streams.

**Key derived features:**
- `wiki_edits_conflict_articles`: Monthly edit count on conflict-related Wikipedia articles for a country
- `wiki_new_conflict_articles`: Count of new conflict-related articles created
- `wiki_revert_rate`: Edit reversion rate (proxy for contested information / information war)
- `wiki_language_divergence`: Divergence between country's local-language and English Wikipedia coverage of political events

**Studies:** Keegan et al. (2013) Wikipedia and breaking news; EMBERS used Wikipedia traffic as LASSO sub-model predictor; Yasseri et al. (2014) Wikipedia edit wars as political polarisation measure.

---

### Tor Project Metrics
**Provider:** Tor Project (metrics.torproject.org) | **Coverage:** Global | **Frequency:** Daily | **Access:** Free

Daily country-level estimates of Tor users and relay bandwidth. Spikes indicate either increased surveillance fear or circumvention of censorship.

**Key variables:**
- `tor_users_direct`: Estimated direct Tor users per country per day
- `tor_users_bridge`: Estimated bridge (censorship-circumventing) Tor users
- `tor_bridge_pct`: % of Tor users using bridges (higher = more censorship)

**ML operationalization:**
- Tor usage spike (>2σ above rolling mean) as censorship/crackdown signal
- Bridge ratio increase as censorship intensification indicator
- Used by EMBERS as a predictor — one of the few signals that captures population anticipation of repression

---

### V-Dem Digital Society Project (DSP)
**Provider:** V-Dem Institute | **Coverage:** ~170 countries | **Frequency:** Annual | **Access:** Free (part of V-Dem dataset)

**Key variables:**
- `v2smgovdom`: Government dissemination of false information domestically
- `v2smgovfilprc`: Government Internet filtering in practice
- `v2smfordom`: Foreign dissemination of false information
- `v2smonper`: Online political persecution
- `v2smorgavgact`: Average people's use of social media for political activities
- `v2smpolsoc`: Polarization of society on social media
- `v2smprivex`: Regime surveillance of private digital communication

**ML operationalization:**
- Disinformation score as regime information control indicator
- Social media political use as civil society mobilization capacity
- Digital repression index (composite of surveillance + filtering + persecution)

---

## DOMAIN 6: Natural Resources & Environmental Stress

### SPEI (Standardized Precipitation-Evapotranspiration Index)
**Provider:** CSIC (Spanish National Research Council) | **Coverage:** Global, 1901–present | **Frequency:** Monthly | **Access:** Free (spei.csic.es)

Multi-scale drought index accounting for both precipitation deficit and evapotranspiration demand (unlike SPI which only measures precipitation). Available at 1, 3, 6, 12, 24, 48-month timescales.

**Key variables:**
- `spei_3`: 3-month SPEI (short-term agricultural drought)
- `spei_12`: 12-month SPEI (medium-term water balance)
- `spei_24`: 24-month SPEI (long-term/hydrological drought)

**ML operationalization:**
- SPEI < −1.5 (severe drought flag) in growing season months
- Consecutive drought months (duration measure)
- Drought following prior drought (compound stress)
- SPEI anomaly relative to 30-year baseline

**Studies:** Raleigh & Kniveton (2012) rainfall variability and conflict; Theisen, Holtermann & Buhaug (2011) drought and civil war; Burke et al. (2009) climate-conflict meta-analysis.

---

### ENSO (El Niño–Southern Oscillation) Indices
**Provider:** NOAA / CPC | **Coverage:** Global (teleconnection) | **Frequency:** Monthly | **Access:** Free (CPC, NOAA)

**Key indices:**
- `oni`: Oceanic Niño Index (primary ENSO indicator; El Niño = positive, La Niña = negative)
- `mei_v2`: Multivariate ENSO Index (more comprehensive)
- `soi`: Southern Oscillation Index (pressure-based)

**ML operationalization:**
- El Niño phase as agricultural shock predictor for affected regions (Sub-Saharan Africa, South Asia, Latin America)
- ENSO phase × regional crop dependence interaction
- Lagged 3–6 months ahead of growing season as food insecurity precursor

**Studies:** Hsiang, Meng & Cane (2011) — ENSO doubles the risk of civil conflict globally; Ciccone (2011) critical replication study; Harari & La Ferrara (2018) ENSO as IV for conflict.

---

### FAO Locust Watch / Desert Locust Information Service
**Provider:** FAO | **Coverage:** Locust-prone regions (Africa, Middle East, South Asia) | **Frequency:** Monthly | **Access:** Free (locust.fao.org)

Desert locust swarms can destroy 80–100% of crops in affected areas within hours. Locust outbreaks are episodic and highly visible early warning indicators of food insecurity.

**Key variables:**
- `locust_situation`: Monthly situation assessment (No Significant Activity / Recession / Upsurge / Plague)
- `locust_affected_countries`: List of affected countries
- `locust_area_km2`: Area affected by swarms

**ML operationalization:**
- Upsurge/Plague binary flag for country-month
- Locust shock in food-insecure country-months (interaction with FEWS NET IPC level)

---

### IRENA (International Renewable Energy Agency) — Energy Access
**Provider:** IRENA | **Coverage:** Global | **Frequency:** Annual | **Access:** Free (irena.org)

**Key variable:**
- `electrification_rate`: % of population with electricity access (national and urban/rural split)
- `energy_intensity`: Energy consumption per unit GDP (development proxy)

**ML operationalization:**
- Low electrification rate as infrastructure failure / state reach indicator
- Rural-urban electrification gap as spatial inequality predictor

---

### Oil Price and Petro-State Shock Variables
**Provider:** World Bank Commodity Markets (Pink Sheet); EIA; IMF Primary Commodity Prices | **Coverage:** Global | **Frequency:** Monthly | **Access:** Free

**Key variables:**
- `crude_oil_price_usd`: Monthly Brent or WTI crude price
- `oil_revenue_pct_gdp`: Country-level oil revenue as % GDP (WDI `NY.GDP.PETR.RT.ZS`)
- `oil_price_shock`: Binary — oil price drops >20% in 6 months
- `commodity_terms_of_trade`: IMF CTOT index (country-specific commodity price index based on export basket)

**ML operationalization:**
- Oil price shock × high oil dependence = fiscal crisis predictor for petro-states
- Commodity terms of trade decline as economic stress indicator for commodity exporters
- Rentier state fiscal buffer (reserves vs. deficit)

**Studies:** Ross (2012) oil curse and conflict; Besley & Persson (2011) oil and state capacity; multiple ILC studies include oil revenue variables.

---

## DOMAIN 7: Arms Trade & Terrorism

### SIPRI Arms Transfers Database
**Provider:** SIPRI (Stockholm International Peace Research Institute) | **Coverage:** Global, 1950–present | **Frequency:** Annual | **Access:** Free (sipri.org/databases/armstransfers)

Records transfers of major conventional weapons between states (and sometimes to non-state actors).

**Key variables:**
- `arms_imports_tiv`: Trend Indicator Value of arms imports (normalized unit for comparison)
- `arms_exports_tiv`: TIV of arms exports
- `arms_supplier_country`: Main supplier country
- `arms_import_growth_5yr`: 5-year rolling change in arms imports

**ML operationalization:**
- Rapid arms import growth as conflict preparation signal
- Small arms import surge as internal repression/civil war precursor
- Arms transfer from authoritarian state as coup-proofing indicator

---

### SIPRI Military Expenditure Database
**Provider:** SIPRI | **Coverage:** ~170 countries, 1949–present | **Frequency:** Annual | **Access:** Free (sipri.org/databases/milex)

**Key variables:**
- `milex_usd_constant`: Military expenditure (constant 2022 USD)
- `milex_pct_gdp`: Military expenditure as % GDP
- `milex_pct_govt_spending`: Military share of government spending
- `milex_per_capita`: Military expenditure per capita

**ML operationalization:**
- Military spending surge (>20% increase over 2 years) as coup-proofing or conflict preparation signal
- High military burden (>4% GDP) as militarized state indicator
- Military vs. social spending ratio as state priority signal

---

### Global Terrorism Database (GTD)
**Provider:** START (National Consortium for the Study of Terrorism and Responses to Terrorism), University of Maryland | **Coverage:** Global, 1970–2021 | **Frequency:** Annual (updated) | **Access:** Free (registration required; start.umd.edu/gtd)

~200,000 terrorist attack events globally. Each event coded for: location, date, attack type, target type, weapons used, casualties, perpetrator group, claimed responsibility.

**Key variables per event:**
- `iyear`, `imonth`, `iday`: Date
- `country_txt`: Country
- `attacktype1_txt`: Assassination / Armed Assault / Bombing / Facility Attack / Hijacking / etc.
- `targtype1_txt`: Government / Military / Police / Private Citizens / Business / etc.
- `weaptype1_txt`: Explosives / Firearms / Chemical / Biological / etc.
- `nkill`: Confirmed fatalities
- `nwound`: Confirmed wounded
- `gname`: Perpetrator group name
- `success`: Binary — attack succeeded

**ML operationalization:**
- Monthly attack count per country
- Monthly fatalities from terrorism
- Attack type diversity (entropy of attack types — higher entropy = more developed insurgency)
- Target government/military ratio (counterinsurgency vs. mass terror)
- Suicide attack rate (organizational commitment proxy)
- New group emergence flag

---

### Small Arms Survey
**Provider:** Graduate Institute Geneva | **Coverage:** Global | **Frequency:** Annual report + periodic surveys | **Access:** Free reports; some datasets downloadable

**Key data products:**
- **Weapons and Markets:** Small arms flows and stockpile estimates
- **Civilian firearms ownership** per 100 people by country
- **Arms embargo violations** records
- **Weapons seizure data** (from national reports)

**ML operationalization:**
- Civilian firearms per 100 as armed mobilization capacity
- Illicit arms flow estimates as conflict escalation predictor
- Arms embargo violation events as conflict escalation signal

---

## DOMAIN 8: International Relations & Institutional Signals

### UN Security Council Voting & Resolutions
**Provider:** UN DGACM | **Coverage:** Global, 1946–present | **Frequency:** Event-based | **Access:** Free (UN research databases; Erik Voeten's UNSC Voting Data)

**Key variables:**
- `unsc_resolution_adopted`: Binary — UNSC resolution adopted mentioning country
- `unsc_resolution_vetoed`: Binary — UNSC resolution vetoed
- `unsc_chapter7`: Binary — Chapter VII (enforcement) resolution adopted
- `p5_voting_alignment`: Voting agreement between P5 members on country-specific resolutions
- `unsc_sanctions_adopted`: Binary — UNSC sanctions adopted

**ML operationalization:**
- Chapter VII adoption as international escalation recognition
- P5 voting divergence (US vs. Russia/China) as geopolitical tension indicator
- Sanctions adoption as regime isolation signal
- Repeated veto of country-specific resolutions as impunity signal (linked to genocide early warning)

**Studies:** Goldsmith et al. (2013) genocide model uses international isolation; multiple atrocity prevention models use UNSC attention as predictor.

---

### Targeted Sanctions Databases
**Provider:** Multiple | **Coverage:** Global | **Frequency:** Event-based | **Access:** Free

**Sources:**
- **UN Consolidated Sanctions List** (UN Security Council): lists individuals and entities under UN sanctions
- **US Treasury OFAC SDN List**: US-designated individuals/entities; country-level program data
- **EU Consolidated Sanctions List**: EU sanctions programs
- **UK OFSI Consolidated List**: Post-Brexit UK sanctions

**ML operationalization:**
- Sanctions program activation (binary): country under US/EU/UN sanctions
- New sanctions designations per month (escalation signal)
- Sanctions program duration (entrenched isolation)
- Senior official designation rate (targeting of regime leadership)
- Financial sector sanctions (banks and central bank) as economic asphyxiation measure

---

### ICC (International Criminal Court) Docket
**Provider:** ICC (icc-cpi.int) | **Coverage:** ~30 situations | **Frequency:** Event-based | **Access:** Free

**Key variables:**
- `icc_situation_country`: Country under ICC investigation/situation
- `icc_stage`: Stage (preliminary examination / investigation / pre-trial / trial)
- `icc_arrest_warrant`: Binary — arrest warrant issued
- `icc_defendant_type`: Head of state / military commander / militia leader
- `icc_conviction`: Binary — conviction entered

**ML operationalization:**
- ICC investigation opening as international atrocity recognition
- Arrest warrant for head of state as regime survival threat (associated with coup risk increase)
- ICC situation combined with V-Dem repression indicators

---

### World Bank Project Disbursements & Suspensions
**Provider:** World Bank | **Coverage:** ~150 countries | **Frequency:** Project-level | **Access:** Free (World Bank Projects API)

**Key variables:**
- `wb_active_projects`: Count of active World Bank projects in country
- `wb_disbursement_ratio`: Actual vs. planned disbursement rate
- `wb_project_suspension`: Binary — project suspended in year
- `wb_total_portfolio_usd`: Total World Bank portfolio

**ML operationalization:**
- Project suspension rate as governance/stability signal (WB suspends projects when security prevents implementation)
- Disbursement ratio decline as capacity deterioration
- Large portfolio with low disbursement = implementation crisis

---

### OECD DAC Aid Flows (CREDITOR Reporting System)
**Provider:** OECD | **Coverage:** Global | **Frequency:** Annual | **Access:** Free (OECD.Stat)

**Key variables:**
- `oda_disbursements_usd`: Official Development Assistance received
- `oda_pct_gni`: ODA as % of GNI (aid dependence)
- `aid_volatility`: Year-over-year coefficient of variation of aid flows
- `donor_concentration`: Herfindahl index of aid donor concentration
- `humanitarian_aid_pct`: Share of ODA that is humanitarian (proxy for crisis severity)

**ML operationalization:**
- High aid dependence × aid volatility = fiscal shock risk
- Sudden aid withdrawal as political pressure indicator
- Rising humanitarian share as crisis deepening signal

---

## DOMAIN 9: Governance, Rule of Law & Anti-Corruption

### Transparency International Corruption Perceptions Index (CPI)
**Provider:** Transparency International | **Coverage:** ~180 countries | **Frequency:** Annual | **Access:** Free (transparency.org)

Composite index based on surveys of business executives and expert assessments of public sector corruption.

**Key variable:** `cpi_score` (0–100; higher = less corrupt)

**ML operationalization:**
- CPI score as governance quality predictor
- CPI score decline of >5 points as governance deterioration signal
- Combined with WGI control of corruption (similar concept, different methodology — both can be used)

---

### Bertelsmann Transformation Index (BTI)
**Provider:** Bertelsmann Stiftung | **Coverage:** ~137 developing and transition countries | **Frequency:** Biennial (odd years) | **Access:** Free (bti-project.org)

Expert-coded assessment covering political transformation (stateness, rule of law, political participation, democracy), economic transformation, and governance quality.

**Key variables:**
- `bti_democracy_score`: Democratic transformation score (1–10)
- `bti_stateness`: State monopoly on use of force (1–10)
- `bti_rule_of_law`: Separation of powers and judicial independence
- `bti_civil_society`: Civil society traditions and participation
- `bti_management_index`: Governance quality composite
- `bti_status`: Regime status (Democracy / Defective Democracy / Moderate Autocracy / Hard Autocracy)

**ML operationalization:**
- Stateness score < 4 as state fragility indicator
- Management index decline as leadership quality deterioration
- BTI provides finer-grained governance distinctions than Polity/V-Dem for autocracy subtypes

---

### World Justice Project Rule of Law Index
**Provider:** World Justice Project | **Coverage:** ~140 countries | **Frequency:** Annual | **Access:** Free (worldjusticeproject.org)

Survey-based index across 8 rule of law dimensions:

| Factor | Description |
|--------|-------------|
| Constraints on government powers | Accountability of government to law |
| Absence of corruption | In executive, legislative, judiciary, police |
| Open government | Transparency and civic participation |
| Fundamental rights | Basic freedoms |
| Order and security | Crime, civil conflict |
| Regulatory enforcement | Fair implementation of regulations |
| Civil justice | Accessible and effective civil courts |
| Criminal justice | Effective and impartial criminal system |

**ML operationalization:**
- Factor 5 (Order and Security) as conflict/violence baseline predictor
- Factor 1 (Constraints on government powers) as autocratization early warning
- Cross-year score change as governance trajectory indicator

---

### Basel AML Index
**Provider:** Basel Institute on Governance | **Coverage:** ~150 countries | **Frequency:** Annual | **Access:** Free (baselgovernance.org)

Composite risk score for money laundering and terrorist financing vulnerability.

**Key variable:** `aml_risk_score` (1–10; higher = higher risk)

**ML operationalization:**
- High AML risk as indicator of illicit financial flows (funding armed groups, corrupt elite)
- AML score change as financial governance trajectory
- Combined with FATF grey/black list status

---

## DOMAIN 10: Civil Society, Protest Mobilisation & Elections

### Mass Mobilization Project (MMP)
**Provider:** Clark & Regan (Penn State / Notre Dame) | **Coverage:** 162 countries, 1990–2020 | **Frequency:** Annual event list | **Access:** Free (massmobilization.org)

Every mass mobilization event involving 50+ participants demanding government action. Crucially codes **government responses** to protest, not just the protests themselves.

**Key variables per event:**
- `country`, `year`, `startday`, `endday`
- `protest`: Protest type (Demand for Change / Demand for Policy / Demand for End of War / Labor Wage / etc.)
- `participants`: Size category (50–99, 100–999, 1000–1999, 2000–4999, 5000–9999, 10000+)
- `region`: Sub-national region
- `protesteridentity`: Group identity of protesters (Government / Military / Labor / Students / Human Rights / etc.)
- `protesterviolence`: Binary — protesters used violence
- `stateresponse1-7`: Government response (Ignore / Accommodate / Crowd Dispersal / Shootings / Killings / Arrests / Ban / Beatings)
- `accomodation`: Binary — demands accommodated

**ML operationalization:**
- Violent state response as escalation predictor (shooting protesters → coup risk increase; Chenoweth & Stephan)
- Protest size trajectory (escalating events)
- Accommodation rate (government flexibility indicator)
- Protest frequency trend (mobilization capacity)
- Multi-type protest (labor + students + religious groups simultaneously = broad coalition)

---

### NAVCO (Nonviolent and Violent Campaigns and Outcomes) Data
**Provider:** Chenoweth & Lewis (Harvard Kennedy School) | **Coverage:** Global, 1900–2019 | **Frequency:** Campaign-level | **Access:** Free (Erica Chenoweth's website)

**NAVCO 1.0:** 323 maximalist campaigns (seeking regime change, independence, or expulsion of occupation). Codes whether campaign was violent or nonviolent, outcome, peak participation.

**NAVCO 2.0:** Annual data on same campaigns — participation, tactical diversity, regime response, international support.

**Key variables:**
- `camp_type`: Nonviolent / Violent / Mixed
- `success`: Binary campaign outcome
- `peak_participation`: Peak protest participants (absolute and % population)
- `govt_response`: Repression / Accommodation / Concession
- `international_isolation`: International sanctions or support
- `defection`: Security force defection from regime

**ML operationalization:**
- Active maximalist campaign as regime survival risk
- Participation rate > 3.5% of population as tipping point (Chenoweth's 3.5% rule)
- Security force defection as coup precursor
- Nonviolent campaign with 3.5%+ participation → near-certain regime change

---

### NELDA (National Elections Across Democracy and Autocracy)
**Provider:** Hyde & Marinov (Yale) | **Coverage:** ~180 countries, 1945–2020 | **Frequency:** Election-event | **Access:** Free

**Key variables per election:**
- `year`, `countryname`
- `nelda3`: Was the election competitive? (candidates/parties allowed to run vs. single-party)
- `nelda6`: Did opposition candidates have a realistic chance of winning?
- `nelda10`: Was there significant violence before the election?
- `nelda14`: Were there reports of government harassment of opposition?
- `nelda15`: Were there international election monitors?
- `nelda24`: Did the incumbent lose?
- `nelda30`: Were the results widely accepted?
- `nelda48`: Were there post-election protests?

**ML operationalization:**
- Election proximity (months to scheduled election) as instability window
- Competitive election in autocracy (nelda6=yes) as coup risk spike
- Post-election protest binary as unrest predictor
- Incumbent loss in hybrid regime as potential coup trigger
- International monitoring presence as legitimacy indicator

---

### Carnegie Endowment Global Protest Tracker
**Provider:** Carnegie Endowment for International Peace | **Coverage:** Global, 2017–present | **Frequency:** Real-time event tracking | **Access:** Free (carnegieendowment.org/publications/interactive/protest-tracker)

Tracks all significant protest movements globally with characterization of demands, scale, and outcome.

**Key variables:**
- `country`, `start_date`, `end_date` (if concluded)
- `status`: Ongoing / Concluded / Unclear
- `outcome`: Successful / Partial Success / Unsuccessful / Unclear
- `demands`: Economic / Political / Social / Racial / Electoral / Anti-war / etc.
- `scale`: Local / National / Transnational
- `state_response`: Accommodation / Repression / Mixed / None

**ML operationalization:**
- Active protest count per country per month
- Economic demand protests (linked to food/inflation indicators)
- Political demand protests (linked to regime type and election cycle)
- Multi-demand protests as regime stability threat

---

### V-Dem Electoral Episodes Dataset
**Provider:** V-Dem | **Coverage:** ~180 countries, 1900–present | **Frequency:** Episode-based | **Access:** Free (part of V-Dem)

**Key variables:**
- `eep_start`, `eep_end`: Episode dates
- `eep_electoral_malpractice`: Level of manipulation
- `eep_outcome`: Election canceled / boycotted / manipulated / violent
- `eep_country_in_episode`: Binary

**ML operationalization:**
- Active electoral malpractice episode as regime legitimacy crisis
- Election cancellation as autocratization signal
- Malpractice → protest → coup sequence detection

---

## DOMAIN 11: Africa-Specific & Regional Data Sources

### Ibrahim Index of African Governance (IIAG)
**Provider:** Mo Ibrahim Foundation | **Coverage:** 54 African countries | **Frequency:** Annual | **Access:** Free (mo.ibrahim.foundation)

The most comprehensive Africa-specific governance assessment. Covers 4 categories, 14 sub-categories, and 103 indicators.

**Four main categories:**
1. **Security and Rule of Law** (safety, rule of law, accountability, personal freedoms)
2. **Participation, Rights and Inclusion** (participation, rights, gender, social inclusion)
3. **Foundations for Economic Opportunity** (public management, business environment, infrastructure, rural sector)
4. **Human Development** (health, education, welfare)

**Key variables:**
- `iiag_overall_score`: Country-level composite (0–100)
- `iiag_security_score`: Security sub-category (most relevant for instability)
- `iiag_governance_trend`: 10-year trend direction
- Per-indicator scores for all 103 indicators

**ML operationalization:**
- Security sub-score as conflict risk predictor
- Score decline of >5 points as governance crisis
- Security-specific indicators (state authority over territory, absence of violence) as direct instability signals
- Outperforms WGI/Polity for Africa-specific models due to Africa-specific calibration

---

### Afrobarometer
**Provider:** Afrobarometer consortium | **Coverage:** 34–38 African countries | **Frequency:** Every 2–3 years (rounds) | **Access:** Free (afrobarometer.org)

Face-to-face survey of ~1,200 respondents per country. Rounds 1–9 available (2000–2023).

**Key questions relevant to instability forecasting:**
- *"How much do you trust the president / parliament / police / military / local government?"* (Q40 series)
- *"How satisfied are you with the way democracy works?"*
- *"Do you think the country is going in the right direction?"*
- *"Have you or your family gone without food / water / medical care / electricity in the past year?"*
- *"How often do you feel unsafe in your neighborhood?"*
- *"Do you support the use of violence to prevent an election outcome?"*
- *"Would you support military rule if the government is doing a bad job?"*
- *"Has the government performed well enough to deserve re-election?"*
- *"Were you afraid to vote your conscience?"*

**ML operationalization:**
- Country-round aggregate trust scores as institutional legitimacy measure
- "Going in right direction" as political mood indicator
- Support for military rule question as coup tolerance indicator
- Deprivation index (% without food/water/medicine) as material grievance measure
- Fear of election violence as conflict anticipation signal

**Note:** Biennial/triennial frequency requires interpolation for monthly models.

---

### African Development Bank (AfDB) Datasets
**Provider:** AfDB | **Coverage:** 54 African countries | **Frequency:** Annual | **Access:** Free (afdb.org/en/knowledge/statistics)

**Relevant data products:**
- **African Economic Outlook (AEO):** Country-level macro forecasts and growth narratives
- **Country Policy and Institutional Assessment (AfDB-CPIA):** Africa-specific governance quality scores
- **Africa Infrastructure Development Index (AIDI):** Transport, electricity, ICT, WASH access composite
- **African Gender Index:** Gender equality sub-national data
- **Statistical Yearbook:** Comprehensive annual statistics

**Key variables:**
- `afdb_cpia_score`: Governance quality (1–6 scale, same methodology as World Bank CPIA but Africa-specific calibration and coverage)
- `aidi_score`: Infrastructure quality composite
- `aeo_growth_forecast`: GDP growth forecast (forward-looking)
- `aeo_risks_flagged`: Qualitative risk flags in country notes

**ML operationalization:**
- AIDI score as state service delivery capacity
- Infrastructure gap (AIDI) as governance legitimacy indicator
- AfDB-CPIA for countries where World Bank CPIA has gaps

---

### FEWS NET Livelihoods and Market Data
**Provider:** FEWS NET / WFP / FAO | **Coverage:** ~30 food-insecure countries | **Frequency:** Monthly | **Access:** Free (fews.net, WFP VAM)

Beyond the IPC phases, FEWS NET produces:
- **Staple food price monitoring:** Country-specific market prices for maize, sorghum, millet, rice, wheat at key markets
- **Terms of trade (pastoral):** Ratio of livestock price to staple food price (critical for Sahel/Horn pastoralists)
- **Market functionality:** Whether key markets are functioning
- **Access analysis:** Physical and economic access to food

**ML operationalization:**
- Staple food price spike (>2σ above seasonal mean) at key market as immediate unrest trigger
- Pastoral terms of trade collapse as Sahel conflict precursor
- Market closure as conflict escalation signal

---

### Sahel and West Africa Club (SWAC) / OECD Sahel Data
**Provider:** SWAC/OECD | **Coverage:** Sahel + West Africa | **Frequency:** Annual | **Access:** Free

**Products:**
- **Sahel and West Africa Atlas:** Cross-border security, migration, and economic data
- **Cross-border security events:** Banditry, jihadist incidents by corridor

**ML operationalization:**
- Cross-border corridor violence as regional spillover indicator
- Specific to Sahel conflict dynamics not well-captured by global datasets

---

## DOMAIN 12: Subnational & Georeferenced Survey Data

### DHS Program (Demographic and Health Surveys)
**Provider:** ICF / USAID | **Coverage:** ~90 low/middle-income countries | **Frequency:** Every 5 years per country | **Access:** Free (registration required; dhsprogram.com)

Nationally representative household surveys with GPS coordinates (cluster-level).

**Key variables relevant to instability:**
- `wealth_index`: Household wealth quintile
- `access_health_facility`: Distance/time to health care
- `water_source_improved`: Improved water access
- `electricity_access`: Household electricity
- `child_mortality_rate`: Under-5 mortality (sub-national)
- `women_decision_making`: Women's autonomy index (linked to gender inequality/conflict)
- `ethnicity_household`: Ethnic group of household head (when public)
- `conflict_exposure_questions`: Some rounds include direct conflict exposure items

**ML operationalization:**
- Sub-national wealth distribution as spatial inequality predictor
- Rural-urban inequality in health/education access
- Cluster-level aggregates to district/province for subnational models
- Wealth index variance within country as horizontal inequality measure

---

### Afrobarometer Subnational (Geo-coded)
**Provider:** Afrobarometer / ICPSR | **Coverage:** Survey rounds with GPS coordinates | **Frequency:** Per round | **Access:** Free with registration

GPS coordinates of enumeration areas enable subnational aggregation to admin-1 or admin-2 levels.

**ML operationalization:**
- Regional trust deficit mapping (which regions distrust central government most)
- Regional deprivation variation within country
- Subnational political mood as conflict escalation predictor in specific regions

---

### PRIO-GRID Extensions (v3.0)
**Provider:** PRIO | **Coverage:** Global, 0.5° grid | **Frequency:** Annual / static layers | **Access:** Free

The existing synthesis covers core PRIO-GRID variables. Version 3.0 adds:
- `travel_time_capital`: Travel time to capital city (minutes) — updated from OpenStreetMap
- `water_scarcity`: Annual freshwater withdrawal / renewable freshwater (FAO AQUASTAT derived)
- `mining_sites`: Count of active mining concessions within cell (from USGS/S&P Capital IQ)
- `oil_field`: Binary — oil or gas field present within cell (PRIO resource data)
- `cell_excluded_group`: Excluded ethnic group presence (from EPR spatial)
- `road_density`: Road network density (OSM-derived)

**ML operationalization:**
- High travel time to capital = low state reach = conflict haven
- Mining site presence as resource conflict predictor (especially artisanal mining)
- Water scarcity combined with agricultural dependence as climate-conflict variable

---

## DOMAIN 13: Composite & Early Warning Systems (as Data Sources)

### Fund for Peace Fragile States Index (FSI)
**Provider:** Fund for Peace | **Coverage:** 179 countries | **Frequency:** Annual | **Access:** Free (fundforpeace.org)

The existing synthesis mentions FSI but the 12 sub-indicators deserve individual documentation:

| Indicator | Code | Description |
|-----------|------|-------------|
| Security Apparatus | C1 | Monopoly on use of force, armed groups, crime |
| Factionalized Elites | C2 | Political fragmentation, power struggles |
| Group Grievance | C3 | Communal grievances, ethnic/religious divisions |
| Economic Decline | E1 | GDP, poverty, debt levels |
| Uneven Development | E2 | Spatial and group economic inequality |
| Human Flight & Brain Drain | E3 | Professional emigration, refugee outflows |
| State Legitimacy | P1 | Government effectiveness and accountability |
| Public Services | P2 | Health, education, infrastructure delivery |
| Human Rights | P3 | Political repression, press freedom |
| Demographic Pressures | S1 | Population pressures, disease, environment |
| Refugees & IDPs | S2 | Displacement burden |
| External Intervention | X1 | Foreign interference, international missions |

Each scored 0–10 (0=most stable, 10=most fragile). Total FSI = sum of all 12 = 0–120.

**ML operationalization:**
- Individual sub-scores as thematic predictors (C1 = security apparatus most directly linked to coup risk)
- Score trajectory (worsening across multiple indicators = systemic deterioration)
- Sub-score divergence (e.g., high C2 + low C1 = factionalized elite without security capacity = coup risk)

---

### Political Terror Scale (PTS)
**Provider:** Gibney et al. (University of North Carolina Asheville) | **Coverage:** ~195 countries, 1976–present | **Frequency:** Annual | **Access:** Free (politicalterrorscale.org)

Expert-coded annual level of political violence and repression by governments.

**Scale (1–5):**
1. Countries under a secure rule of law; political murders are rare; torture exceptional; political imprisonment rare
2. Limited amounts of imprisonment for non-violent political activity; beating of prisoners common; political murder rare; limited political murder
3. Extensive political imprisonment; execution or other political murders and brutality may be common; unlimited detention without trial
4. Civil and political rights violations have expanded to large numbers of the population; murders, disappearances, and torture common
5. Terror has expanded to the whole population; leaders incite and tolerate murders, disappearances, and torture

**Two versions:** Based on Amnesty International reports (PTS_A) and US State Department reports (PTS_S). Both included in dataset.

**ML operationalization:**
- PTS score as government repression intensity predictor
- PTS level 4–5 as genocide precursor (Harff 2003)
- Transition from PTS 2→4 within 2 years as crisis escalation
- PTS_A vs. PTS_S divergence as political bias indicator

---

### Early Warning Project (EWP) Risk Assessments
**Provider:** United States Holocaust Memorial Museum | **Coverage:** ~162 countries | **Frequency:** Annual | **Access:** Free (earlywarningproject.org)

Published annual risk assessments for onset of mass atrocities. Uses a Statistical Risk Assessment (SRA) model producing country-level probability estimates.

**Key outputs:**
- `epa_risk_score`: Annual probability of mass atrocity onset (0–1)
- `epa_rank`: Country rank by risk
- `epa_opinion_pool`: Expert-opinion pool assessment (separate from statistical model)

**ML operationalization:**
- EWP SRA score as mass atrocity risk predictor (pre-computed ensemble probability — can be used as meta-feature)
- EWP + ACLED fatalities + V-Dem repression as three-component atrocity early warning composite

---

## VARIABLE INTERACTION EFFECTS WORTH ENGINEERING

Based on cross-study synthesis, the following interaction terms have been found predictive:

| Interaction | Mechanism | Relevant Outcome |
|-------------|-----------|-----------------|
| Anocracy × Economic contraction | Partial democracies under economic stress → coup or breakdown | Coup, regime change |
| Food price spike × High urban population | Urban food riots more likely in food-import dependent cities | Mass unrest |
| ENSO El Niño × Agricultural dependence × Poverty | Crop failure in poor, food-insecure agrarian countries | Civil war, unrest |
| Social media penetration × State repression | Information access + repression = protest coordination vs. crackdown | Mass unrest, coup |
| Coup history × Economic contraction | Repeated-coup countries more vulnerable to economic shocks | Coup |
| Refugee outflow × Neighboring armed conflict | Conflict spillover through population displacement | Civil war diffusion |
| Election proximity × Incumbency advantage low | Contested elections in weakly institutionalized states | Mass unrest, coup |
| Arms import surge × Executive constraint decline | Military buildup + weakening oversight = coup-proofing or coup preparation | Coup |
| FEWS NET IPC ≥ 3 × ACLED protest count | Food crisis + protest mobilization = escalation to violence | Mass unrest → riot → civil conflict |
| Oil price crash × High oil revenue dependence | Fiscal crisis → spending cuts → patronage network collapse | Coup, unrest |

---

## PRACTICAL NOTES ON DATA ACQUISITION

| Dataset | Access Pathway | Typical Lag |
|---------|---------------|-------------|
| FEWS NET IPC | fews.net direct download; API available | 1 month |
| IDMC | internal-displacement.org; Python library `pydisplace` | 6–12 months |
| GTD | registration at start.umd.edu; annual update | 12–18 months |
| SIPRI milex | sipri.org; April annual release | 12 months |
| NAVCO | Erica Chenoweth website; Harvard Dataverse | Static (through 2019) |
| MMP | massmobilization.org; Harvard Dataverse | ~24 months |
| NELDA | Harvard Dataverse | ~18 months |
| BTI | bti-project.org; biennial release (odd years) | 6 months after scoring year |
| Afrobarometer | afrobarometer.org; round release timing varies | 12–24 months after fieldwork |
| IIAG | mo.ibrahim.foundation; annual November release | 12 months |
| NetBlocks | netblocks.org; structured API requires partnership | Near real-time |
| EM-DAT | emdat.be; registration required | 3–6 months |
| Powell-Thyne coup | Clayton Thyne's website; annual update | 12 months |
| EWP risk scores | earlywarningproject.org; annual April release | 6 months |
| V-Dem | v-dem.net; annual March release | 12 months |
| VIIRS nightlights | NASA Earthdata; Google Earth Engine | 1–3 months |
| SPEI | spei.csic.es | 1–2 months |

---

*Compiled April 2026 from: SIPRI databases; START GTD codebook; Powell-Thyne coup dataset documentation; FEWS NET IPC reference manual; IDMC GRID reports; Mass Mobilization Project codebook (Clark & Regan); NAVCO 2.0 codebook (Chenoweth & Lewis); NELDA codebook v6 (Hyde & Marinov); V-Dem Codebook v15 (including Digital Society Project indicators); Fund for Peace FSI methodology; Ibrahim Foundation IIAG methodology; Afrobarometer questionnaire documentation (rounds 1–9); Mo Ibrahim Foundation data portal; AfDB statistics portal; EM-DAT CRED database; Laeven & Valencia (IMF WP 2018); Reinhart & Rogoff (2009); GRACE/GRACE-FO mission documentation; ESA Sentinel-1 SAR mission documentation; NASA FIRMS documentation; WorldPop project; Small Arms Survey annual reports; Basel AML Index methodology; World Justice Project Rule of Law Index methodology; Bertelsmann BTI codebook; Transparency International CPI methodology; Carnegie Endowment Global Protest Tracker.*