> **Archived.** Codebooks for core datasets, dependent-variable taxonomy, and the variable-usage matrix across 10 prior studies. Operational content has been merged into `../data-and-predictors.md` (sections 1–3). Kept here as the long-form reference and source citation.

---

# Dataset & Variable Synthesis
## Dependent Variables (Labels) and Independent Variables (Predictors) Across the ML Political Instability Literature

---

## HOW TO READ THIS DOCUMENT

This document synthesizes every dataset used across the reviewed literature into three layers:

1. **Source Dataset Profiles** — what each dataset contains at the raw record level
2. **Dependent Variable (Label) Taxonomy** — how each instability outcome is operationalized across studies
3. **Independent Variable (Predictor) Taxonomy** — how predictors are grouped by thematic domain, with specific variable names

Where a variable appears in multiple studies, its most common operationalization is noted. Where operationalizations differ significantly between studies, both are documented.

**Unit of analysis** in most studies is the **country-year** (annual, ~170 countries) or **country-month** (monthly). Some use subnational grids (PRIO-GRID, 0.5° × 0.5°). This distinction matters for feature engineering: monthly models require high-frequency predictors; annual models can use slower-moving structural data.

---

## PART 1: SOURCE DATASET PROFILES

### 1.1 Event Data — Ground Truth for Instability

---

#### ACLED (Armed Conflict Location and Event Data)
**Provider:** ACLED nonprofit | **Coverage:** Global, 1997–present | **Frequency:** Real-time (event-level) | **Open:** Yes (free registration)

**Unit of observation:** The individual event (a single episode of political violence or protest on a specific day in a specific location involving named actors).

**Raw fields (31 columns in current schema):**

| Field | Type | Description |
|---|---|---|
| `event_id_cnty` | string | Unique country-specific event ID |
| `event_date` | date | Date of event (YYYY-MM-DD) |
| `year` | integer | Year of event |
| `time_precision` | integer | Confidence in date (1=day, 2=week, 3=month) |
| `disorder_type` | string | Top-level: Political Violence / Demonstrations / Strategic Developments |
| `event_type` | string | Battles / Explosions/Remote Violence / Violence Against Civilians / Riots / Protests / Strategic Developments |
| `sub_event_type` | string | 25 sub-types (e.g., Armed Clash, Government Regaining Territory, Protest with Intervention) |
| `actor1` | string | Primary named actor (e.g., "Military Forces of Ethiopia") |
| `assoc_actor_1` | string | Actor associated with Actor 1 |
| `inter1` | integer | Actor 1 type code (1=state, 2=rebel, 3=militia, 4=communal, 5=rioter, 6=protester, 7=civilian, 8=external state) |
| `actor2` | string | Secondary named actor |
| `assoc_actor_2` | string | Actor associated with Actor 2 |
| `inter2` | integer | Actor 2 type code |
| `interaction` | integer | Interaction type code (dyadic; derived from inter1/inter2) |
| `civilian_targeting` | string | Flag if civilians were deliberately targeted |
| `iso` | integer | ISO country code |
| `region` | string | Broad geographic region |
| `country` | string | Country name |
| `admin1` | string | First-level administrative division |
| `admin2` | string | Second-level administrative division |
| `location` | string | Named town/city/place |
| `latitude` | float | Latitude (degrees) |
| `longitude` | float | Longitude (degrees) |
| `geo_precision` | integer | Spatial precision (1=named location, 2=nearby town, 3=admin level) |
| `source` | string | Primary source for the event |
| `source_scale` | string | Scope of source (International/National/Subnational/Other) |
| `notes` | text | Free-text event description |
| `fatalities` | integer | Best estimate of deaths (note: often imprecise/underreported) |
| `tags` | string | Additional classification tags |
| `timestamp` | integer | Data ingestion timestamp |

**Derived features commonly created from ACLED for ML:**
- Monthly/quarterly event counts by event_type per country or grid cell
- Rolling N-day event counts (7, 14, 30, 90, 180 days)
- Fatality counts and delta (change vs. prior period)
- Actor type diversity index (H-index of inter1/inter2 codes)
- Geographic dispersion of events (entropy or convex hull area)
- Protest-to-battle ratio (demonstrations / political violence)
- Civilian targeting rate (civilian_targeting events / total events)
- New actor entry flag (actor appearing for first time)
- Event type transition flags (e.g., protests → riots → battles)

---

#### GDELT (Global Database of Events, Language, and Tone)
**Provider:** GDELT Project | **Coverage:** Global, 1979–present | **Frequency:** Every 15 minutes (since 2013) | **Open:** Yes (BigQuery, direct download)

**Unit of observation:** A coded event as an (Actor1, EventCode, Actor2) triple extracted from a news article.

**Key fields (58 fields in GDELT 2.0):**

| Field | Type | Description |
|---|---|---|
| `GlobalEventID` | integer | Unique event identifier |
| `Day` | integer | Date in YYYYMMDD format |
| `Actor1Name` | string | Name of first actor (e.g., "GEORGE W BUSH", "KURD") |
| `Actor1CountryCode` | string | 3-character CAMEO country code for Actor 1 |
| `Actor1Type1Code` | string | CAMEO role code (e.g., "MIL" = military, "GOV" = government) |
| `Actor1EthnicCode` | string | CAMEO ethnic affiliation code |
| `Actor1ReligionCode` | string | CAMEO religious affiliation code |
| `Actor2Name` | string | Name of second actor |
| `Actor2CountryCode` | string | 3-character CAMEO country code for Actor 2 |
| `EventCode` | string | Full CAMEO event code (4-digit hierarchy, 300+ types) |
| `EventBaseCode` | string | Level-2 CAMEO code (aggregate) |
| `EventRootCode` | string | Level-1 CAMEO code (20 root types) |
| `QuadClass` | integer | 1=Verbal Cooperation, 2=Material Cooperation, 3=Verbal Conflict, 4=Material Conflict |
| `GoldsteinScale` | float | Event stability impact: −10 (destabilizing) to +10 (stabilizing), assigned by event type |
| `NumMentions` | integer | Total news mentions in the 15-minute update window |
| `NumSources` | integer | Number of distinct source outlets |
| `NumArticles` | integer | Number of source documents |
| `AvgTone` | float | Average article sentiment: −100 (negative) to +100 (positive); typical range −10 to +10 |
| `Actor1Geo_Lat` | float | Latitude of Actor 1 location |
| `Actor1Geo_Long` | float | Longitude of Actor 1 location |
| `ActionGeo_Lat` | float | Latitude where event occurred |
| `ActionGeo_Long` | float | Longitude where event occurred |
| `SOURCEURL` | string | URL of source article |

**CAMEO event root codes (EventRootCode) — the 20 top-level event categories:**

| Code | Category | Conflict-relevant? |
|---|---|---|
| 01 | Make public statement | Neutral |
| 02 | Appeal | Neutral |
| 03 | Express intent to cooperate | Cooperative |
| 04 | Consult | Cooperative |
| 05 | Engage in diplomatic cooperation | Cooperative |
| 06 | Engage in material cooperation | Cooperative |
| 07 | Provide aid | Cooperative |
| 08 | Yield | Cooperative |
| 09 | Investigate | Neutral |
| 10 | Demand | Conflict signal |
| 11 | Disapprove | Conflict signal |
| 12 | Reject | Conflict signal |
| 13 | Threaten | Conflict signal |
| 14 | Protest (non-violent) | Protest signal |
| 15 | Exhibit force posture | Conflict signal |
| 16 | Reduce relations | Conflict signal |
| 17 | Coerce | Conflict signal |
| 18 | Assault | Violence |
| 19 | Fight | Violence |
| 20 | Use unconventional mass violence | Mass violence / genocide signal |

**Derived features commonly created from GDELT for ML:**
- Monthly Goldstein Scale mean and volatility (std dev) per country
- Monthly AvgTone mean and trend per country
- QuadClass 3+4 event count ratio (conflict event proportion)
- Rolling CAMEO 14/15/18/19/20 event counts (protest, force, assault, fight, mass violence)
- NumMentions-weighted event intensity score
- Actor-type co-occurrence matrices (government vs. military, government vs. civilian)
- Geographic dispersion of conflict events (grid-level)

---

#### UCDP-GED (Uppsala Conflict Data Program — Georeferenced Event Dataset)
**Provider:** UCDP, Uppsala University | **Coverage:** Global, 1989–present | **Frequency:** Annual (candidate events: monthly) | **Open:** Yes

**Unit of observation:** A conflict incident (a reported use of armed force by an organized actor resulting in ≥1 battle death or ≥25 civilian deaths for one-sided violence).

**Key fields:**

| Field | Type | Description |
|---|---|---|
| `id` | integer | Unique event ID |
| `year` | integer | Year of event |
| `type_of_violence` | integer | 1=State-based, 2=Non-state, 3=One-sided violence |
| `conflict_new_id` | integer | Conflict ID linking events to the same conflict |
| `dyad_new_id` | integer | Dyad (pair of actors) identifier |
| `side_a` | string | Government/state side actor name |
| `side_b` | string | Opposing actor name |
| `country` | string | Country where event occurred |
| `region` | string | Geographic region |
| `date_start` | date | Start date of event |
| `date_end` | date | End date of event |
| `deaths_a` | integer | Best estimate of deaths on Side A |
| `deaths_b` | integer | Best estimate of deaths on Side B |
| `deaths_civilians` | integer | Best estimate of civilian deaths |
| `deaths_unknown` | integer | Deaths with unknown affiliation |
| `best` | integer | Best estimate of total deaths |
| `high` | integer | High-end estimate of total deaths |
| `low` | integer | Low-end estimate of total deaths |
| `latitude` | float | Latitude of event location |
| `longitude` | float | Longitude of event location |
| `geom_wkt` | geometry | Event location as WKT polygon/point |
| `priogrid_gid` | integer | PRIO-GRID cell ID for spatial joins |
| `source_article` | text | Source article citation |

**Three types of organized violence coded:**
- **State-based:** Armed conflict between two organized actors, at least one of which is a state government (minimum 25 battle deaths/year threshold)
- **Non-state:** Conflict between two organized actors, neither being a state government
- **One-sided:** An organized actor using armed force against civilians resulting in ≥25 deaths/year

---

#### ICEWS (Integrated Crisis Early Warning System)
**Provider:** Lockheed Martin / US DoD (public release via Harvard Dataverse) | **Coverage:** Global, 1995–present | **Frequency:** Daily | **Open:** Partially (via Harvard Dataverse)

Similar structure to GDELT (CAMEO coding), but events are coded by human-assisted machine process with higher quality control. Key additional variables: `story_id`, `publisher`, `sentence_number` (within article). Used extensively in EMBERS and the Ward/Beger ILC models.

---

#### RSUI (Reported Social Unrest Index)
**Provider:** Barrett, Appendino, Nguyen (IMF) | **Coverage:** Global, 1996–present | **Frequency:** Monthly | **Open:** Yes (via IMF working paper)

**Construction:** Monthly count of newspaper articles in Dow Jones Factiva that include a country name AND the words "protest" OR "riot" OR "revolution" within 10 words of "unrest," scaled by total article count in the period.

**Single output variable:** `rsui_score` — a continuous monthly index of newspaper-based unrest intensity per country. In ML studies, this is binarized to:
- `unrest_binary` = 1 if RSUI score exceeds a threshold in year t+1 (label for prediction)

---

### 1.2 Structural / Macropolitical Datasets — Sources of Independent Variables

---

#### Polity V (Political Regime Characteristics and Transitions)
**Provider:** Center for Systemic Peace | **Coverage:** Global, 1800–2018 | **Frequency:** Annual | **Open:** Yes

**Core variables:**

| Variable | Type | Description | Range |
|---|---|---|---|
| `polity2` | integer | Net democracy-autocracy score (composite) | −10 to +10 |
| `polity` | integer | Raw polity score (may include special codes) | −10 to +10 (special: −66=interruption, −77=interregnum, −88=transition) |
| `democ` | integer | Institutionalized Democracy component | 0–10 |
| `autoc` | integer | Institutionalized Autocracy component | 0–10 |
| `xrreg` | integer | Executive Recruitment Regulation | 1–3 |
| `xrcomp` | integer | Executive Recruitment Competitiveness | 0–3 |
| `xropen` | integer | Executive Recruitment Openness | 0–4 |
| `xconst` | integer | Executive Constraints (Decision Rules) | 1–7 |
| `parcomp` | integer | Competitiveness of Political Participation | 0–5 |
| `parreg` | integer | Regulation of Participation | 1–5 |
| `exrec` | integer | Executive Recruitment (combined) | 1–8 |
| `exconst` | integer | Executive Constraints (combined) | 1–7 |
| `polcomp` | integer | Political Competition (combined) | 0–10 |
| `durable` | integer | Years since last regime change (durability) | 0–N |
| `fragment` | integer | State fragmentation flag | 0/1 |
| `regtrans` | integer | Regime transition code | various |

**PITF-derived variable used in Goldstone et al. 2010:**
- `pitf_regime_type`: 5-category nonlinear regime classification (Full Democracy, Partial Democracy w/factions, Partial Democracy, Partial Autocracy, Full Autocracy) — derived from `xrreg`, `xrcomp`, `xconst`, `parcomp`; **the single most powerful predictor in the PITF model**

**How used in ML studies:**
- As predictor (independent variable): regime type, executive constraints, durability, participation competition
- Sometimes as label (dependent): `regtrans` (regime transition occurred), `polity2` change > threshold (democratic backsliding)

---

#### V-Dem (Varieties of Democracy, v15)
**Provider:** V-Dem Institute, University of Gothenburg | **Coverage:** Global, 1789–present | **Frequency:** Annual | **Open:** Yes

V-Dem provides 400+ indicators. The most commonly used in ML instability prediction:

**Regime-level indices:**
| Variable | Description |
|---|---|
| `v2x_libdem` | Liberal Democracy Index (0–1) |
| `v2x_polyarchy` | Electoral Democracy Index (0–1) |
| `v2x_partipdem` | Participatory Democracy Index (0–1) |
| `v2x_egaldem` | Egalitarian Democracy Index (0–1) |
| `v2x_delibdem` | Deliberative Democracy Index (0–1) |
| `v2x_regime` | Regime type (0=Closed Autocracy, 1=Electoral Autocracy, 2=Electoral Democracy, 3=Liberal Democracy) |
| `v2x_civlib` | Civil Liberties Index (0–1) |
| `v2x_freexp_altinf` | Freedom of Expression and Alternative Information (0–1) |

**Early warning / backsliding indicators:**
| Variable | Description |
|---|---|
| `v2x_jucon` | Judicial Constraints on the Executive (0–1) |
| `v2x_legcon` | Legislative Constraints on the Executive (0–1) |
| `v2juncind` | Judicial Independence (0–1) |
| `v2mecenefm` | Media censorship effort by government |
| `v2meharjrn` | Harassment of journalists |
| `v2cscnsult` | CSO (civil society) consultation |
| `v2csreprss` | CSO repression |
| `v2elvotbuy` | Vote buying |
| `v2elfrfair` | Freedom and fairness of elections |
| `v2clacfree` | Freedom of academic and cultural expression |
| `v2clrgunev` | Equal and secure rights |
| `v2xcs_ccsi` | Core Civil Society Index (0–1) |
| `v2regendtype` | How regime ended (coup, election, etc.) — used as label |
| `v2x_corr` | Political Corruption Index (0–1) |
| `v2peedueq` | Educational equality |
| `v2pepwrsoc` | Power distributed by socioeconomic position |

**ERT (Episodes of Regime Transformation) — derived from V-Dem:**
| Variable | Description |
|---|---|
| `e_v2x_polyarchy_diff` | Year-on-year change in polyarchy score |
| `ert_episode` | Flag: country is in an autocratization/democratization episode |
| `ert_direction` | Episode direction (autocratization / democratization) |
| `ert_outcome` | Episode outcome (transition, reversal, ongoing) |

---

#### PITF State Failure Problem Set
**Provider:** Political Instability Task Force / Center for Systemic Peace | **Coverage:** Global, 1948–present | **Frequency:** Annual | **Open:** Yes

**Case-level variables (onset = label):**

| Variable | Type | Description |
|---|---|---|
| `sftprev` | binary | 1 = country experienced state failure event this year |
| `ethwar` | binary | Ethnic war onset (≥1,000 total deaths, ≥100/year, ethnic-based) |
| `revwar` | binary | Revolutionary war onset (organized group challenging government) |
| `genocide` | binary | Genocide/politicide onset (state-directed mass killing of communal or political group) |
| `adverse` | binary | Adverse regime change (sudden loss of state authority, non-democratic replacement) |
| `sfyear` | integer | Year event occurred |
| `sfbdate` | date | Start date of event |
| `sfedate` | date | End date of event |
| `sfintens` | integer | Intensity score (1=low to 4=high) |
| `sfnrebels` | integer | Number of rebels active |
| `sfprebels` | integer | Peak number of rebels |
| `sfareaf` | float | Percentage of territory affected |
| `sfdeaths` | integer | Estimated deaths |

**Note:** The four binary onset variables (`ethwar`, `revwar`, `genocide`, `adverse`) are the standard **dependent variables** across the PITF-based literature (Goldstone 2010, Fox 2021, King & Zeng 2001, Harff 2003).

---

#### Powell-Thyne Coup Dataset
**Provider:** Clayton Thyne & Jonathan Powell (UCF) | **Coverage:** Global, 1950–present | **Frequency:** Annual event list | **Open:** Yes

**Variables:**

| Variable | Type | Description |
|---|---|---|
| `country` | string | Country name |
| `year` | integer | Year of coup |
| `month` | integer | Month of coup |
| `day` | integer | Day of coup |
| `coup` | integer | 1=Successful, 2=Attempted |
| `version` | string | Dataset version |

**In ML studies (CoupCast, Ward/Beger):** Monthly binary indicator `coup_binary` (was there a coup attempt in this country-month?) is the **dependent variable**. An additional distinction is sometimes made: `coup_success` (outcome = new leader) vs. `coup_attempt` (failed).

---

#### Archigos (Leadership Data)
**Provider:** Bueno de Mesquita et al. / Ward group | **Coverage:** Global, 1875–2015 | **Frequency:** Leader-level | **Open:** Yes

**Key variables used as predictors in coup/ILC models:**

| Variable | Type | Description |
|---|---|---|
| `leader` | string | Leader name |
| `startdate` | date | Start of tenure |
| `enddate` | date | End of tenure |
| `exitcode` | integer | How leader left (regular = election/term; irregular = coup, killed, fled, etc.) |
| `military` | binary | Leader had military background |
| `age_at_entry` | integer | Leader's age when taking power |
| `leader_gender` | binary | Leader gender |
| `irregular_exit` | binary | Exited irregularly (the **label** for ILC prediction) |

---

#### World Bank World Development Indicators (WDI)
**Provider:** World Bank | **Coverage:** Global, 1960–present | **Frequency:** Annual | **Open:** Yes (API)

Most commonly used WDI variables in instability prediction:

**Economic:**
| Indicator Code | Description |
|---|---|
| `NY.GDP.PCAP.KD` | GDP per capita (constant 2015 USD) |
| `NY.GDP.MKTP.KD.ZG` | GDP growth rate (% annual) |
| `FP.CPI.TOTL.ZG` | Inflation (consumer price index, % annual) |
| `SL.UEM.TOTL.ZS` | Unemployment (% of total labor force) |
| `NY.GNP.PCAP.KD` | GNI per capita (Atlas method) |
| `BX.KLT.DINV.WD.GD.ZS` | FDI (net inflows % of GDP) |
| `BN.CAB.XOKA.GD.ZS` | Current account balance (% of GDP) |
| `GC.DOD.TOTL.GD.ZS` | Central government debt (% of GDP) |
| `FP.CPI.TOTL` | Consumer price index (2010=100) |

**Social / Demographic:**
| Indicator Code | Description |
|---|---|
| `SP.POP.TOTL` | Total population |
| `SP.POP.GROW` | Population growth rate (%) |
| `SP.DYN.IMRT.IN` | Infant mortality rate (per 1,000 live births) — **top predictor in Goldstone 2010, Fox 2021** |
| `SE.ADT.LITR.ZS` | Adult literacy rate (%) |
| `SI.POV.GINI` | Gini coefficient (income inequality) |
| `SP.URB.TOTL.IN.ZS` | Urban population (% of total) |
| `SP.URB.GROW` | Urban population growth rate |
| `SH.XPD.CHEX.GD.ZS` | Current health expenditure (% of GDP) |
| `SE.PRM.ENRR` | School enrollment, primary (% gross) |
| `SP.POP.0014.TO.ZS` | Population ages 0–14 (% of total) — youth bulge indicator |
| `SP.POP.1564.TO.ZS` | Population ages 15–64 (% of total) |

**Technology / Infrastructure:**
| Indicator Code | Description |
|---|---|
| `IT.CEL.SETS.P2` | Mobile phone subscriptions (per 100 people) — **top SHAP predictor IMF study** |
| `IT.NET.USER.ZS` | Individuals using the internet (% of population) |
| `IT.NET.BBND.P2` | Fixed broadband subscriptions (per 100 people) |

**Governance:**
| Indicator Code | Description |
|---|---|
| `CC.EST` | Control of Corruption (WGI estimate) |
| `GE.EST` | Government Effectiveness (WGI estimate) |
| `PV.EST` | Political Stability and Absence of Violence (WGI) |
| `RL.EST` | Rule of Law (WGI estimate) |
| `RQ.EST` | Regulatory Quality (WGI estimate) |
| `VA.EST` | Voice and Accountability (WGI estimate) |

---

#### FAO Food Price Monitoring and Analysis (FPMA)
**Provider:** FAO | **Coverage:** Global | **Frequency:** Monthly | **Open:** Yes

| Variable | Description |
|---|---|
| `food_price_index` | FAO Food Price Index (monthly, 2014–2016=100) |
| `cereal_price_index` | Cereal Price Index |
| `meat_price_index` | Meat Price Index |
| `dairy_price_index` | Dairy Price Index |
| `sugar_price_index` | Sugar Price Index |
| `oils_price_index` | Vegetable Oil Price Index |
| `domestic_food_price_level` | Country-level food price deviation from trend (monthly, per country) |

**Why this matters:** Food price inflation is the **#1 SHAP-ranked predictor** in the IMF Barrett et al. (2021) social unrest model. Price spikes in cereals are strongly associated with protest waves.

---

#### Freedom House (Annual Country Reports)
**Provider:** Freedom House | **Coverage:** Global, 1972–present | **Frequency:** Annual | **Open:** Yes

| Variable | Description | Range |
|---|---|---|
| `PR` | Political Rights score | 1–7 (1=most free, 7=least free) |
| `CL` | Civil Liberties score | 1–7 (1=most free, 7=least free) |
| `Status` | Aggregate status category | Free / Partly Free / Not Free |
| `internet_freedom` | Internet freedom score | 0–100 (100=most free) |
| `electoral_democracy` | Binary electoral democracy flag | Yes/No |

---

#### Ethnic Power Relations (EPR)
**Provider:** Cederman, Wimmer, Min (ETH Zurich) | **Coverage:** Global, 1946–present | **Frequency:** Annual | **Open:** Yes

| Variable | Description |
|---|---|
| `gwgroupid` | Ethnic group identifier |
| `year` | Year |
| `status` | Ethnic group's access to political power (Monopoly / Dominant / Senior Partner / Junior Partner / Powerless / Discriminated / Self-exclusion / Irrelevant) |
| `size` | Estimated group population share |
| `reg_aut` | Regional autonomy flag |
| `downcoded` | Flag indicating recent loss of political access |

**Key derived variable for ML:** `ethnic_exclusion` = binary flag if a large ethnic group (>10% of population) has `status` = "Powerless" or "Discriminated." Strong predictor in ethnic war models.

---

#### PRIO-GRID
**Provider:** PRIO (Peace Research Institute Oslo) | **Coverage:** Global | **Frequency:** Annual/static | **Open:** Yes

A unified spatial data structure at 0.5°×0.5° resolution. Key grid-level variables:

| Variable | Description |
|---|---|
| `cell` | Grid cell ID |
| `gwno` | Gleditsch-Ward country code |
| `xcoord` | Longitude of cell centroid |
| `ycoord` | Latitude of cell centroid |
| `pop_gpw_sum` | Grid-cell population (Gridded Population of the World) |
| `excluded_group` | Excluded ethnic group present flag (from EPR) |
| `nlights_calib_mean` | Mean nightlight intensity (DMSP-OLS; proxy for economic activity) |
| `bdist1` | Distance to nearest international border (km) |
| `bdist2` | Distance to nearest coast (km) |
| `mountains_mean` | Mean terrain ruggedness (% mountainous) |
| `forest_gc_per` | Forest cover percentage |
| `petroleum_y` | Petroleum extraction flag |
| `drug_y` | Drug crop cultivation flag |
| `wdi_gdp` | Grid-interpolated GDP |
| `rainfall_sd` | Rainfall variability (std dev from GPCP) |
| `temp_sd` | Temperature variability (std dev) |

**Used in:** ViEWS, Muchlinski (2013), Fearon et al. (2021), common in subnational civil conflict models as spatial predictors.

---

#### Twitter / Social Media Signals
**Derived features from Twitter API used in EMBERS, Chitengu (2025), and others:**

| Feature | Description |
|---|---|
| `protest_keyword_count` | Daily count of tweets containing protest-related keywords per country |
| `sentiment_score` | Aggregate tweet sentiment (VADER, XLM-RoBERTa) |
| `sentiment_volatility` | Standard deviation of daily sentiment over rolling window |
| `hashtag_frequency` | Count of political hashtag mentions |
| `retweet_amplification_ratio` | Retweet rate as proxy for content spread/organization |
| `network_clustering_coefficient` | Louvain community detection in hashtag co-occurrence networks |
| `tor_requests` | Country-level TOR anonymity network request volume (proxy for censorship/surveillance fear) — used in EMBERS |
| `influencer_engagement_rate` | Engagement on posts from accounts with >10k followers |

---

#### LDA Newspaper Topics (Mueller & Rauh methodology)
**Derived from:** Global newspaper text (ProQuest, LexisNexis, GDELT URLs)

| Feature | Description |
|---|---|
| `topic_k` | Monthly proportion of country's news assigned to topic k (k=1..K, where K=50–100) |
| `topic_k_delta` | Month-over-month change in topic proportion (within-country variation — the key signal) |
| `conflict_topic_proportion` | Aggregate proportion of conflict-related topics |
| `stabilizing_topic_proportion` | Aggregate proportion of topics negatively associated with conflict |

**Construction:** Latent Dirichlet Allocation (LDA) trained on country-year newspaper corpora. Topic proportions vary within country over time and serve as leading indicators of conflict onset because they capture the contemporaneous framing of events before conflict is coded.

---

## PART 2: DEPENDENT VARIABLE TAXONOMY

This section catalogs every outcome variable (the ML prediction target / label) used across the reviewed studies, organized by instability domain.

---

### 2.1 Social Unrest / Mass Protest Labels

| Label Variable | Operationalization | Source | Studies |
|---|---|---|---|
| `unrest_binary` | 1 = RSUI score exceeds threshold in country-year (approx. top quartile) | RSUI (Barrett et al.) | IMF WP 2021/263 |
| `protest_monthly` | 1 = any ACLED Protests event in country-month | ACLED | Chitengu 2025 |
| `riot_monthly` | 1 = any ACLED Riots event in country-month | ACLED | Chitengu 2025 |
| `protest_count` | Integer count of ACLED protest events per country-month | ACLED | EMBERS, ViEWS |
| `protest_uptick` | 1 = protest count in month exceeds 3-month rolling average by threshold | ACLED, EMBERS GSR | EMBERS |
| `unrest_event` | 1 = Wikipedia-recognized large-scale civil unrest event in county-month | Wikipedia + GDELT | Combs et al. 2018 |
| `tweet_unrest_class` | Binary: tweet indicates high vs. low unrest potential | Twitter | Ayetiran 2023 (SVM) |
| `unrest_type_gov` | Sub-type: government-related unrest | RSUI | IMF WP (sub-analysis) |
| `unrest_type_election` | Sub-type: election-related unrest | RSUI | IMF WP (sub-analysis) |
| `unrest_type_violent` | Sub-type: violent unrest (hardest to predict) | RSUI | IMF WP (sub-analysis) |

**Forecasting horizons used:**
- 1 month ahead (EMBERS, Chitengu 2025)
- 3 months ahead (EMBERS suppression window)
- 12 months ahead (IMF WP)

---

### 2.2 Coup d'État Labels

| Label Variable | Operationalization | Source | Studies |
|---|---|---|---|
| `coup_attempt` | 1 = any coup attempt (successful or failed) in country-month | Powell-Thyne | CoupCast, Ward/Beger |
| `coup_success` | 1 = successful coup in country-year | Powell-Thyne | INSCR coups list |
| `ilc_binary` | 1 = Irregular Leadership Change (coup, rebellion, or protest leading to leader removal) in country-month | Archigos + Powell-Thyne | Beger/Dorff/Ward 2014, 2017 |
| `adverse_regime_change` | 1 = PITF-coded adverse regime change onset in country-year | PITF | Goldstone 2010, Fox 2021 |

**Forecasting horizons:**
- 1 month ahead (CoupCast monthly probability)
- 6 months ahead (Ward/Beger ILC 6-month rolling forecasts)
- 12–24 months ahead (PITF-based models)

---

### 2.3 Civil War / Armed Conflict Labels

| Label Variable | Operationalization | Source | Studies |
|---|---|---|---|
| `civil_war_onset` | 1 = new UCDP-coded armed conflict (≥25 battle deaths/year) begins in country-year | UCDP | Muchlinski 2016, Blair & Sambanis 2020 |
| `civil_war_incidence` | 1 = ongoing armed conflict this country-year | UCDP | Hegre et al. 2013, 2021 |
| `conflict_state` | 3-class: 0=no conflict, 1=minor conflict, 2=major conflict | UCDP | Hegre 2013 (multinomial) |
| `battle_deaths` | Continuous: log(best estimate of battle deaths) per country-month | UCDP-GED | ViEWS (regression target) |
| `delta_fatalities` | Change in battle deaths vs. prior period (escalation label) | UCDP-GED, ACLED | ViEWS Escalation Competition |
| `conflict_onset_ucdp` | 1 = new UCDP conflict episode starts (≥25 BRD threshold crossed) | UCDP | Most civil war ML studies |
| `acled_battle_onset` | 1 = first ACLED Battles event in district in rolling window | ACLED | Muchlinski 2013, Fearon 2021 |
| `ethnic_war_onset` | 1 = PITF ethnic war onset in country-year | PITF | Goldstone 2010, Fox 2021 |
| `revolutionary_war_onset` | 1 = PITF revolutionary war onset in country-year | PITF | Goldstone 2010, Fox 2021 |
| `conflict_probability` | Continuous [0,1]: model output probability of conflict | — | ViEWS, ConflictForecast.org |
| `violence_against_civilians` | 1 = ACLED one-sided violence event in country-month | ACLED | ViEWS, CEHA |

**Forecasting horizons:**
- 1–3 months ahead (EMBERS, ViEWS candidate)
- 6 months ahead (ViEWS standard output)
- 12 months ahead (IMF WP, Goldstone 2010)
- 24 months ahead (Fox 2021 two-year horizon)
- Long-run (2010–2050): Hegre 2013 simulation

---

### 2.4 Regime Change / State Failure Labels

| Label Variable | Operationalization | Source | Studies |
|---|---|---|---|
| `state_failure` | 1 = any PITF instability event (combined) onset in country-year | PITF | King & Zeng 2001, Goldstone 2010 |
| `polity_decline` | 1 = Polity2 score declines by ≥3 points in year-over-year | Polity V | Democratic backsliding studies |
| `democratic_breakdown` | 1 = transition from democracy (polity2 ≥6) to non-democracy in country-year | Polity V | ERT, V-Dem studies |
| `autocratization_episode` | 1 = country is in a V-Dem-coded autocratization episode | V-Dem ERT | Maerz et al. 2024 |
| `regime_transition` | Ordinal: 0=stable, 1=democratic transition, 2=autocratic transition, 3=breakdown | V-Dem ERT | Maerz et al. 2024 |
| `ilc_binary` | 1 = irregular leadership change (see coup section) | Archigos | Beger/Ward |
| `state_fragility_score` | Continuous fragility index | CSP State Fragility Index | Fragility prediction studies |

---

### 2.5 Genocide / Politicide Labels

| Label Variable | Operationalization | Source | Studies |
|---|---|---|---|
| `geno_onset` | 1 = Harff/PITF-coded genocide/politicide onset in country-year | PITF Genocide data | Harff 2003, Goldsmith 2013 |
| `tmk_episode_onset` | 1 = TMK-coded targeted mass killing episode begins | TMK Dataset | Butcher et al. 2020 |
| `mass_killing_onset` | 1 = state-sponsored mass killing begins (USHMM EWP definition) | SSMK / EWP | Early Warning Project |
| `geno_risk_score` | Continuous [0,1]: probability of genocide onset in country-year | — | AFP model outputs |
| `one_sided_violence` | 1 = UCDP one-sided violence (≥25 civilian deaths) | UCDP-GED | ViEWS |

---

## PART 3: INDEPENDENT VARIABLE (PREDICTOR) TAXONOMY

This section catalogs all predictor variables used across the literature, organized by thematic domain. Each variable is tagged with the studies that used it.

---

### 3.1 Regime Type & Governance (Political Structural Predictors)

These are the most consistently powerful predictors across all five instability domains.

| Predictor Variable | Description | Source | Key Studies Using It |
|---|---|---|---|
| **Regime type (5-category PITF)** | Full Democracy / Partial Democracy / Anocracy / Partial Autocracy / Full Autocracy | Polity V | Goldstone 2010 — **most powerful predictor** |
| **Polity2 score** | −10 (autocracy) to +10 (democracy) continuous scale | Polity V | Almost all studies |
| **Anocracy flag** | Binary: polity2 ∈ [−5, +5] (partial democracy / partial autocracy) | Polity V | Goldstone 2010, Fox 2021, IMF WP |
| **Polity durability** | Years since last regime change (time-since-last-instability) | Polity V / PITF | Fox 2021 — **3rd of 3 key predictors** |
| **Executive constraints (xconst)** | Constraints on executive decision-making (1–7) | Polity V | Civil war, coup studies |
| **Executive recruitment openness** | How openly leaders are recruited to office (xropen, xrreg) | Polity V | ILC models |
| **Competitiveness of participation** | Degree to which political competition is permitted (parcomp) | Polity V | Goldstone 2010, coup models |
| **Liberal Democracy Index** | V-Dem composite index of liberal democratic norms | V-Dem | ViEWS, regime change studies |
| **Judicial independence** | V-Dem: courts' independence from executive (v2juncind) | V-Dem | Backsliding early warning |
| **Freedom of expression** | V-Dem: media freedom and alternative information access | V-Dem | Autocratization detection |
| **CSO repression** | V-Dem: civil society organization repression score | V-Dem | MLP project, backsliding |
| **Electoral integrity** | V-Dem: freedom and fairness of elections (v2elfrfair) | V-Dem | Regime change prediction |
| **Political rights score** | Freedom House PR score (1–7) | Freedom House | IMF WP, ILC models |
| **Civil liberties score** | Freedom House CL score (1–7) | Freedom House | IMF WP, regime change |
| **Political Terror Scale** | Government repression index (1–5) | PTS | ViEWS structural models |
| **State fragility index** | Composite fragility score (CSP) | CSP | State failure models |
| **Polity fragmentation** | Flag: state cannot exercise authority over ≥50% of territory | Polity V | State failure, civil war |
| **Irregular exit probability (lag)** | Leader's lagged probability of being removed irregularly | Archigos | ILC/coup models |
| **Military leader** | Binary: leader has military background | Archigos | CoupCast predictor |
| **Leader tenure length** | Years current leader has been in power | Archigos | ILC models |
| **Election scheduled** | Binary: national election scheduled within 6 months | Electoral calendars | IMF WP, ILC models |

---

### 3.2 Economic Conditions (Macro-Economic Predictors)

| Predictor Variable | Description | Source | Key Studies |
|---|---|---|---|
| **GDP per capita (log)** | Log of GDP per capita (constant USD) | WDI | All studies; most basic structural predictor |
| **GDP growth rate** | Annual % change in real GDP | WDI / IMF | IMF WP, civil war models |
| **GDP per capita growth rate** | Annual % change in per capita GDP | WDI | Fearon-Laitin basis |
| **Food price inflation** | Annual % change in domestic food price index | FAO FPMA / WDI | IMF WP — **#1 SHAP predictor for unrest** |
| **Food Price Index** | FAO monthly global food price index | FAO FPMA | EMBERS (as predictor), IMF WP |
| **Inflation (CPI)** | Consumer price index annual % change | WDI | IMF WP, ILC models |
| **Unemployment rate** | % of labor force unemployed | WDI | IMF WP |
| **Currency exchange rate** | Domestic currency vs. USD (daily) | Central banks / EMBERS | EMBERS predictor |
| **Trade openness** | Imports + exports as % of GDP | WDI | Harff 2003 (genocide predictor — inversely) |
| **Oil/natural resource rents** | Resource rents as % of GDP | WDI | Civil war studies (resource curse) |
| **FDI net inflows** | Foreign direct investment (% of GDP) | WDI | ILC models |
| **Government debt** | Central government debt (% GDP) | WDI / IMF | IMF WP |
| **Current account balance** | External balance (% GDP) | WDI / IMF | IMF WP |
| **Nighttime light intensity** | DMSP-OLS/VIIRS nightlight (proxy for GDP at subnational level) | NOAA | ViEWS grid-level models |

---

### 3.3 Demographic & Social Predictors

| Predictor Variable | Description | Source | Key Studies |
|---|---|---|---|
| **Infant mortality rate** | Deaths per 1,000 live births under age 1 | WDI | Goldstone 2010, Fox 2021 — **#2 of 3 key predictors** |
| **Population (log)** | Log of total population | WDI | Most civil war studies |
| **Population growth rate** | Annual % population increase | WDI | IMF WP |
| **Youth bulge** | % of population aged 15–29 | WDI | IMF WP, demographic stress models |
| **Urban population %** | Percent of population in urban areas | WDI | IMF WP, protest models |
| **Ethnic fractionalization** | Probability two random individuals have different ethnicity | Fearon & Laitin 2003 basis | Civil war onset models |
| **Ethnic group exclusion flag** | Large ethnic group (>10%) politically excluded | Ethnic Power Relations | Ethnic war, genocide models |
| **Gini coefficient** | Income inequality (0=equal, 100=maximal) | WDI | IMF WP |
| **HDI / Education** | Human Development Index / Adult literacy | UNDP / WDI | Regime change models |
| **Refugee flows** | Incoming or outgoing refugees (% of population) | UNHCR | Regional spillover predictor |
| **State-led discrimination** | Government discriminates against ethnic groups | EPR / Harff | Harff 2003 genocide model — **key predictor** |

---

### 3.4 Military & Security Predictors

| Predictor Variable | Description | Source | Key Studies |
|---|---|---|---|
| **Military spending** | % of GDP on military | WDI / SIPRI | ILC models, coup prediction |
| **Military personnel** | Active military personnel (per capita) | WDI / SIPRI | Coup prediction |
| **Paramilitary / irregular forces** | Presence of paramilitaries / pro-government militias | ACLED actor types | ViEWS |
| **Number of active rebel groups** | Count of ACLED-coded rebel organizations active | ACLED | Civil war escalation |
| **Territorial control changes** | ACLED: territory gained/lost by non-state actors | ACLED sub-event | ViEWS |
| **Prior conflict episodes** | Count of prior PITF/UCDP conflict episodes | PITF / UCDP | All studies — **conflict trap** predictor |
| **Conflict history (years since last)** | Years since last PITF/UCDP conflict onset | PITF / UCDP | Fox 2021 — **#3 of 3 key predictors** |
| **Journalist killings** | Number of journalists killed in prior year | CPJ / RSF | Gohdes & Carey 2017 (canary in coal mine) |
| **Political prison population** | Estimates of political prisoners | Amnesty / PTS | Genocide early warning models |

---

### 3.5 Geographic / Physical Predictors

| Predictor Variable | Description | Source | Key Studies |
|---|---|---|---|
| **% mountainous terrain** | Proportion of country with rough terrain (log) | Fearon-Laitin basis | Civil war onset studies |
| **Distance to capital** | Distance from grid cell to national capital (km) | PRIO-GRID | Subnational conflict models |
| **Distance to border** | Distance from grid cell to nearest international border | PRIO-GRID | Conflict diffusion models |
| **Forest cover** | % of land area forested | PRIO-GRID | Civil war onset (rebels can hide) |
| **Land area (log)** | Log of country territory in sq km | WDI | Fearon-Laitin controls |
| **Coastline / access to sea** | Binary flag | WDI | Civil war onset controls |
| **Landlocked** | Binary: landlocked country flag | WDI | Trade/conflict controls |
| **Rainfall variability** | Std dev of annual rainfall (climate stress proxy) | PRIO-GRID / GPCP | ViEWS, conflict-climate models |
| **Temperature anomaly** | Deviation from historical temperature mean | PRIO-GRID / Berkeley | Climate-conflict studies |
| **Natural disaster frequency** | Count of natural disasters in prior year | EM-DAT | State failure, coup risk after shocks |

---

### 3.6 Spatial / Neighborhood / Contagion Predictors

| Predictor Variable | Description | Source | Key Studies |
|---|---|---|---|
| **Regional conflict incidence** | Weighted average conflict probability of neighboring countries | UCDP / ACLED | Goldstone 2010 (**#4 predictor**); ViEWS |
| **Number of neighbors at war** | Count of contiguous countries experiencing armed conflict | UCDP | Most civil war models |
| **Spatial lag of ACLED events** | Weighted ACLED event count in contiguous grid cells | ACLED + PRIO-GRID | ViEWS grid models |
| **Cultural/linguistic sphere spillover** | Shared language/religion contagion measure | COW + EPR | Hegre 2021 (gap identified after Arab Spring) |
| **Refugee flows from neighbors** | Outflows from neighboring countries as instability signal | UNHCR | Contagion models |
| **UN peacekeeping presence** | Binary: UN PKO operating in country | UN DPPA | ViEWS (moderating variable) |
| **Coup contagion** | Regional coup attempt in prior 3 months | Powell-Thyne | Coup prediction |

---

### 3.7 Event-Derived / Dynamic Predictors (from ACLED, GDELT, ICEWS)

These are derived from the event datasets and serve as dynamic, high-frequency predictors in monthly or near-real-time models.

| Predictor Variable | Description | Source | Key Studies |
|---|---|---|---|
| **Rolling protest count (7/30/90 day)** | ACLED protest events in rolling window | ACLED | EMBERS, ViEWS |
| **Rolling battle count** | ACLED battle events in rolling window | ACLED | ViEWS, Muchlinski 2013 |
| **Rolling fatality count** | ACLED fatalities in rolling window | ACLED | ViEWS, civil war models |
| **Delta fatalities (trend)** | Change in rolling fatality count vs. prior period | ACLED | ViEWS Escalation Competition |
| **Actor diversity index** | H-index of armed actor types active | ACLED inter codes | Sub-conflict complexity indicator |
| **Geographic dispersion** | Entropy of conflict event locations | ACLED | Conflict intensity predictor |
| **Goldstein Scale (GDELT)** | Monthly mean/volatility of GDELT Goldstein scores | GDELT | EMBERS, IMF WP, ViEWS |
| **AvgTone (GDELT)** | Monthly mean news sentiment toward country | GDELT | EMBERS, ConflictForecast.org |
| **Conflict news count (Chadefaux)** | Weekly count of conflict-related news items | Historical newspapers | Chadefaux 2014 |
| **LDA topic proportions** | Country-month topic distribution from news corpus | Global newspapers | Mueller & Rauh 2018, 2022 |
| **Topic proportion delta** | Within-country month-over-month change in topic mix | Global newspapers | Mueller & Rauh (key insight) |
| **TOR traffic volume** | Country-level TOR anonymity network usage | TOR metrics | EMBERS predictor |
| **CAMEO verbal conflict events** | GDELT QuadClass=3 event count | GDELT | EMBERS, ViEWS |
| **CAMEO material conflict events** | GDELT QuadClass=4 event count | GDELT | EMBERS, ViEWS |
| **Protest keyword frequency (Twitter)** | Country-level protest keyword tweet count | Twitter API | EMBERS, Chitengu 2025 |
| **Sentiment score (Twitter)** | Aggregate political sentiment from tweets | Twitter API | Chitengu 2025 — **top SHAP predictor** |
| **Sentiment volatility (Twitter)** | Rolling std dev of daily tweet sentiment | Twitter API | Chitengu 2025 |
| **Day of week / holiday flag** | Binary for weekends, public holidays | Calendar | Chitengu 2025 — **top SHAP predictor** |
| **Election event flag** | Binary: national election occurring in month | Electoral calendars | IMF WP, ILC models |
| **Mobile phone penetration** | Subscriptions per 100 people (monthly) | WDI / ITU | IMF WP — **top SHAP predictor** |

---

### 3.8 Genocide-Specific Predictors (Harff model and derivatives)

These variables are specific to genocide/politicide prediction and not typically used in other instability models:

| Predictor Variable | Description | Source | Key Studies |
|---|---|---|---|
| **Prior genocide/politicide** | Binary: country has experienced genocide in past | PITF Genocide | Harff 2003, Goldsmith 2013 |
| **Exclusionary ideology** | Ruling elite has ethnic/religious exclusionary ideology (coded by Harff) | Expert coding | Harff 2003 — **key predictor** |
| **Ethnic character of elite** | Ruling elite ethnically homogeneous | Harff / expert coding | Harff 2003 |
| **Autocratic history** | Years under autocratic rule | Polity V | Harff 2003 |
| **State failure onset** | Country experiencing PITF state failure event | PITF | Goldsmith 2013 (Stage 1 predictor) |
| **Trade openness (inverse)** | Low trade openness = high genocide risk | WDI | Harff 2003 (surprising protective effect of trade) |
| **Political terror scale** | Government repression score | PTS | Harff 2003 |
| **High-level assassinations** | Count of senior official assassinations | PITF / news | Harff 2003 |
| **Armed group territorial control** | Non-state group controls territory | ACLED | TMK/mass atrocity models |
| **Peacekeeping force presence** | UN PKO active in country | UN DPPA | Surprising positive correlation with genocide risk (Rost 2013) |

---

## PART 4: VARIABLE USAGE MATRIX BY STUDY

This matrix shows which predictor categories each key study used. ✓ = used; (✓) = partially or in some model variants; — = not used.

| Predictor Category | IMF WP 2021 | EMBERS 2014 | Goldstone 2010 | Muchlinski 2016 | ViEWS 2019 | Mueller-Rauh 2018 | CoupCast | Fox 2021 | Harff 2003 | Goldsmith 2013 |
|---|---|---|---|---|---|---|---|---|---|---|
| Regime type / Polity | ✓ | — | ✓ | ✓ | ✓ | — | ✓ | ✓ | ✓ | ✓ |
| V-Dem indicators | — | — | — | — | ✓ | — | ✓ | — | — | — |
| GDP / economic macro | ✓ | (✓) | — | ✓ | ✓ | — | ✓ | — | ✓ | ✓ |
| Food prices | ✓ | ✓ | — | — | — | — | — | — | — | — |
| Infant mortality | ✓ | — | ✓ | — | — | — | — | ✓ | — | — |
| Demographic (youth, pop) | ✓ | — | — | ✓ | ✓ | — | — | — | — | — |
| Ethnic / EPR | ✓ | — | (✓) | ✓ | ✓ | — | — | — | ✓ | (✓) |
| Military / coup-proofing | — | — | — | — | — | — | ✓ | — | — | — |
| Geography / terrain | — | — | — | ✓ | ✓ | — | — | — | — | — |
| Neighborhood / contagion | ✓ | — | ✓ | — | ✓ | — | — | — | — | — |
| ACLED event history | — | (✓) | — | ✓ | ✓ | — | — | — | — | — |
| GDELT / tone | ✓ | ✓ | — | — | (✓) | — | — | — | — | — |
| ICEWS event data | — | ✓ | — | — | — | — | ✓ | — | — | — |
| LDA newspaper topics | — | — | — | — | — | ✓ | — | — | — | — |
| Twitter / social media | — | ✓ | — | — | — | — | — | — | — | — |
| Mobile penetration | ✓ | ✓ | — | — | — | — | — | — | — | — |
| TOR traffic | — | ✓ | — | — | — | — | — | — | — | — |
| Freedom House | ✓ | — | — | — | — | — | — | — | ✓ | ✓ |
| Leader characteristics | — | — | — | — | — | — | ✓ | — | — | — |
| Years since last instability | — | — | — | — | — | — | ✓ | ✓ | — | ✓ |
| Prior genocide | — | — | — | — | — | — | — | — | ✓ | ✓ |
| Exclusionary ideology | — | — | — | — | — | — | — | — | ✓ | ✓ |
| Trade openness | ✓ | — | — | — | — | — | — | — | ✓ | — |
| PRIO-GRID spatial | — | — | — | ✓ | ✓ | — | — | — | — | — |

---

## PART 5: CROSS-CUTTING NOTES ON VARIABLE USAGE

### What Serves as Both Label and Predictor
Several variables appear on *both* sides of the model depending on the study:

- **Polity2 / regime type:** Most often a predictor; occasionally the label (regime change studies)
- **ACLED event counts:** Used as labels (social unrest prediction) AND as lagged predictors (conflict onset models)
- **Prior instability event flag:** Always a predictor (in the form of a lag/history), never a concurrent label
- **UCDP conflict incidence:** Sometimes label (civil war onset), sometimes predictor (lagged conflict history)

### The Temporal Lag Structure
All variables used as predictors must be lagged relative to the label to prevent data leakage:
- **Structural variables (Polity, WDI):** Typically lagged 1–2 years
- **Event variables (ACLED, GDELT):** Typically lagged 1–6 months or aggregated over a prior window
- **Social media signals:** Typically rolling 7–30 day windows ending before the forecast date
- **LDA topic features:** Prior month or quarter proportion

### The Class Imbalance Problem Across All Labels
Every instability label in the literature is severely imbalanced:

| Label | Approximate Base Rate | Effective % of country-years positive |
|---|---|---|
| Social unrest (annual) | ~30–40% | Moderate imbalance |
| Coup attempt (monthly) | ~0.1–0.3% | Extreme imbalance |
| Civil war onset (annual) | ~1–2% | Severe imbalance |
| Adverse regime change (annual) | ~0.8% | Severe imbalance |
| Genocide/politicide onset (annual) | ~0.2–0.5% | Extreme imbalance |

All ML studies must address this through SMOTE, cost-sensitive learning, threshold tuning, or stratified sampling. AUPRC (not AUROC) is the correct evaluation metric for severe imbalance cases.

---

*Compiled April 2026 from primary source codebooks, IMF WP 2021/263 supplementary materials, ACLED Codebook 2024, GDELT Event Codebook v2.0, Polity V Manual, V-Dem Codebook v15, PITF State Failure Problem Set documentation, Powell-Thyne Coup Dataset codebook, and detailed methods sections of reviewed papers.*
