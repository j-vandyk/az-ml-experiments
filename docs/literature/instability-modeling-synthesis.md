# Refining the Target

## What Instability Models Predict, On What Time Scales, and Where Their Labels Come From

*A Synthesis for Practitioners Building Social Unrest Prediction Systems*

**Compiled April 2026**

-----

## Table of Contents

- [Executive Summary](#executive-summary)
- [Part I — What Instability Models Actually Predict](#part-i--what-instability-models-actually-predict)
  - [The Seven Families of Dependent Variables](#the-seven-families-of-dependent-variables)
  - [The Threshold Choice as Hidden Theory](#the-threshold-choice-as-hidden-theory)
  - [Construct Validity: What Each Label Actually Captures](#construct-validity-what-each-label-actually-captures)
- [Part II — Time Scale: The Most Consequential Architectural Choice](#part-ii--time-scale-the-most-consequential-architectural-choice)
  - [The Horizon Spectrum](#the-horizon-spectrum)
  - [How Horizon Constrains Signal Choice](#how-horizon-constrains-signal-choice)
  - [The Calibration vs. Horizon Trade-off](#the-calibration-vs-horizon-trade-off)
  - [Lead Time vs. Recall: The Operational Trade-off](#lead-time-vs-recall-the-operational-trade-off)
- [Part III — Where Labels Come From: Off-the-Shelf vs. Custom-Labeled Data](#part-iii--where-labels-come-from-off-the-shelf-vs-custom-labeled-data)
  - [The Five Sources of Labels](#the-five-sources-of-labels)
  - [The Cost-Benefit of Custom Labeling](#the-cost-benefit-of-custom-labeling)
  - [Sourcing Decision Matrix](#sourcing-decision-matrix)
- [Part IV — Synthesis: Practical Implications for System Design](#part-iv--synthesis-practical-implications-for-system-design)
  - [Three Decisions That Cascade Through Everything](#three-decisions-that-cascade-through-everything)
  - [Common Patterns from the Reviewed Systems](#common-patterns-from-the-reviewed-systems)
  - [What the Field Has Not Yet Solved](#what-the-field-has-not-yet-solved)
- [Appendix — New Sources Surfaced in This Synthesis](#appendix--new-sources-surfaced-in-this-synthesis)

-----

## Executive Summary

This synthesis answers three operational questions that determine what an instability prediction system can credibly deliver: what exactly is the target outcome, on what temporal horizon are we forecasting it, and where is the labeled training data coming from. Each of these choices cascades through every downstream architectural decision — model choice, feature engineering, evaluation metric, and ultimately the system’s policy utility.

The literature reveals a field that has converged on a small set of canonical practices but where critical methodological tensions remain unresolved. On the question of what is predicted, the field has fragmented into at least seven distinct dependent variable families, ranging from binary onset events (did a coup happen?) to continuous fatality counts (how many died?) to ordinal escalation classes (how intense was it?). On time scale, operational systems span from real-time tweet-stream forecasts (EMBERS, hourly to daily) through monthly forecasts at six to thirty-six month horizons (ViEWS, CoupCast) to annual structural risk assessments (IMF, Goldstone PITF) — and the choice of horizon fundamentally constrains which signals are predictive. On data sourcing, perhaps surprisingly, the dominant paradigm is not custom labeling but rather creative repurposing of off-the-shelf event datasets — though the most innovative recent work breaks this pattern in instructive ways.

The recurring methodological insight across these dimensions is that **label construction is the most consequential and least scrutinized decision in the entire ML pipeline**. A 25-fatality threshold versus a 100-fatality threshold for civil war onset produces fundamentally different prediction problems. A binary shock label versus a continuous count produces different model classes. A label drawn from automated CAMEO event coding versus expert-annotated UCDP-GED data produces datasets with documented systematic biases that propagate into model predictions. These are not minor calibration choices; they are the defining acts of constructing the prediction problem itself.

This document synthesizes findings from the prior artifacts in this thread (literature review, dataset/variable taxonomy, model type comparison) and extends them with targeted snowball searches into three new directions: how forecast horizon is operationalized, how dependent variables are constructed and validated, and the off-the-shelf versus custom-label sourcing decision. The goal is to give practitioners building Azure ML or similar production systems a clear-eyed view of the choices that have already been made for them by their data sources — and the choices they still need to make explicitly.

-----

## Part I — What Instability Models Actually Predict

The phrase “predict instability” obscures a fundamental fragmentation in dependent variable construction. Across the reviewed literature, at least seven distinct families of dependent variables are operationalized, each producing different model architectures, evaluation metrics, and policy implications.

### The Seven Families of Dependent Variables

#### Family 1 — Binary Onset Events

The dominant operationalization. The label is 1 if a specified event occurs in a given country-period and 0 otherwise. Examples: civil war onset (Muchlinski et al. 2016, using ≥25 battle deaths/year UCDP threshold), coup attempts (CoupCast and Powell-Thyne data), genocide/politicide onset (Harff 2003, Goldsmith et al. 2013), adverse regime change (PITF). The unit can be country-year, country-month, or grid-month. Binary onset framing is preferred because political scientists fundamentally care about whether something *happens*, and because it produces interpretable probability outputs that policymakers can act on. The cost is information loss: a single battle death and a thousand are coded identically once the threshold is crossed.

#### Family 2 — Continuous Fatality Counts

Used in escalation prediction and the more recent ViEWS prediction competitions. The label is the actual count of battle deaths in a country-month or grid-month, often log-transformed to handle the heavy-tailed distribution. This is the framing for Hegre et al.’s 2023/2024 and 2024/2025 ViEWS prediction challenges and for the Bin-Conditional Conformal Prediction work (Randahl et al. 2024). The advantage is that magnitude is preserved; the disadvantage is that the heavy tail (most country-months have zero fatalities, a few have thousands) is notoriously hard to model and produces brittle point predictions.

#### Family 3 — Continuous Risk/Intensity Indices

A continuous score in [0,1] or similar bounded range, often produced as the model output of a binary classifier and then interpreted as a risk index. Examples: the Reported Social Unrest Index (Barrett et al. 2020) used as both label and predictor in the IMF WP, and the Political Instability Risk Index produced by Fernandes (2026) GEWS as a 0–1 score with four categorical risk bands (Stable / Moderate / High / Critical). These indices serve a different policy function than binary onsets — they are designed for continuous monitoring and ranking of countries, not for triggering specific event-driven interventions.

#### Family 4 — Ordinal Intensity Classes

The label is one of K ordered categories of conflict intensity. Hegre et al. (2013) use a three-class scheme (no conflict / minor conflict / major conflict). Ettensperger (2020) constructs a categorical conflict intensity variable from socioeconomic and political indicators across seven categories. Ordinal framing splits the difference between binary onset (which loses magnitude) and continuous count (which is statistically unstable for rare extremes). The trade-off is that the cut-points between classes are themselves arbitrary modeling decisions that drive results.

#### Family 5 — Discontinuous Change Detection (Shock Detection)

A distinctive operationalization developed by Chen, Springman & Wibbels in the MLP project. The label is whether the month-over-month change in event activity exceeds a threshold, defined as the top 20% of changes between month X and month X+k. This converts a continuous activity measure into a binary shock/no-shock classification, but in a way that focuses the model on detecting *discontinuities* rather than levels. The Italian-MFA-affiliated work by Macis et al. (2024) uses a related framing via autoencoder anomaly detection — here the label is implicit in the unsupervised model’s reconstruction error rather than explicitly defined.

#### Family 6 — Composite Multi-Type Aggregations

The label combines multiple instability event types into a single binary indicator. The Fernandes (2026) GEWS uses a unified label that is 1 if any of {protests, riots, armed conflict, coup, government collapse} occur in a country-year, producing an 11.4% positive rate across 5,830 country-year observations. The advantage is sample size (more positive cases to learn from); the disadvantage is conceptual heterogeneity (a peaceful protest and a coup are coded identically). Basuchoudhary et al. (2018) take a similar approach using PITF, combining political wars, ethnic wars, regime change, genocides, and politicides.

#### Family 7 — Spatial Location Prediction

A radically different framing introduced by Warnke & Runfola (2024). The label is binary — was this 1 km² satellite imagery tile a riot location or not? The model predicts where unrest occurs in space, conditional on knowing it occurred somewhere. This decouples the temporal and spatial dimensions of prediction, producing a system that answers “if unrest occurs in this country, where will it cluster?” rather than “will unrest occur here?” — useful for resource pre-positioning but distinct from event forecasting.

### The Threshold Choice as Hidden Theory

Every binary onset label requires a threshold, and these thresholds are theoretically loaded in ways the literature rarely makes explicit. The UCDP standard of 25 battle-related deaths per calendar year as the criterion for an “armed conflict” is the field’s most consequential threshold choice. UCDP justifies this threshold as the minimum at which lethal organized violence becomes systematically recordable in news sources. Once the threshold is met, the conflict-dyad is included in subsequent years even if it dips below 25 deaths — this means the dependent variable embeds a path-dependent assumption that conflicts “persist” administratively after they cease lethally.

The implication for ML practitioners is that two studies both predicting “civil war” can produce wildly different results because they use different thresholds and persistence rules. Muchlinski et al. (2016) using UCDP/PRIO at the 25-death threshold produces a base rate around 1.6% of country-years. Hegre et al.‘s ViEWS at the 100-death-month threshold produces a different base rate. Mueller & Rauh’s “hard onset” definition (conflicts emerging after at least five years of peace) produces yet another, with PR-AUC scores typically below 0.20 even for state-of-the-art models — demonstrating that the threshold choice is the single biggest determinant of apparent model performance.

Similarly, the IMF’s RSUI binarizes a continuous newspaper-mention index into “unrest occurred” using a country-specific threshold. The Chen/MLP shock label uses the top-20% of month-over-month increases. The Goldsmith genocide model uses Harff’s 1955 binary classification of state-led mass killing onset. None of these thresholds is obviously correct; each encodes a substantive theoretical claim about what counts as “the event” we are trying to predict. A practitioner adopting any of these labels is implicitly adopting that theoretical claim.

### Construct Validity: What Each Label Actually Captures

The table below summarizes what each major label in the literature is actually measuring, the construct it is intended to capture, and the gap between the two.

|Label                          |What It Measures (Operationally)                                                                        |What It Is Intended to Capture                                            |Construct Gap                                                                                                      |
|-------------------------------|--------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------|
|**UCDP Battle Deaths ≥25/year**|News-reported lethal violence with identifiable perpetrators and dates, summed per calendar year        |Organized armed conflict involving the state or organized non-state actors|Excludes violence below threshold; biased toward media-covered conflicts; depends on Factiva search quality        |
|**ACLED Protest Event**        |News-reported public demonstration with non-violent intent                                              |Citizen mobilization in public space against state or other authority     |Misses unreported small protests; over-represents urban, English-language coverage; conflates permitted/spontaneous|
|**Powell-Thyne Coup**          |Expert-coded successful or attempted illegal seizure of executive by military or part of state apparatus|Sudden irregular leadership change by elites within the state             |Boundary cases (failed plots, civilian-led, gradual takeovers) handled inconsistently across versions              |
|**PITF Adverse Regime Change** |Expert-coded transition to less democratic regime within 3 years, scored against Polity                 |Democratic backsliding leading to authoritarian consolidation             |Lagged label (3-year window); biased toward Polity’s institutional emphasis; misses informal/de facto changes      |
|**Harff Genocide/Politicide**  |Expert-coded sustained state-led killing of ≥1,000 civilians targeted by group identity                 |State-perpetrated mass atrocity intended to destroy a group               |High threshold misses lower-intensity atrocities; intent coding requires post-hoc historical assessment            |
|**RSUI Binary**                |Country-month newspaper article count above country-specific threshold for unrest keywords near “unrest”|Significant social unrest events drawing media attention                  |Country-specific threshold; depends entirely on Factiva keyword matching; English-language bias                    |
|**MLP Civic Space Shock**      |Top 20% of month-over-month increases in NLP-classified news article share                              |Discontinuous deterioration in citizen rights or civil society space      |Measures media reporting on events, not events themselves; threshold (20%) is design choice not theory             |
|**V-Dem Polyarchy Drop**       |Year-over-year decline in Liberal Democracy Index ≥0.05 sustained for 2+ years                          |Episode of democratic backsliding                                         |Index changes can reflect coding updates as much as actual democratic change; smoothed measure                     |

-----

## Part II — Time Scale: The Most Consequential Architectural Choice

The forecast horizon is the single most determinative choice in instability prediction system design. It dictates which signals are predictive (real-time event flows for short horizons; structural conditions for long horizons), which models are appropriate (sequence models for short horizons, tabular models for long horizons), how labels can be constructed (high-frequency labels exist only for some outcomes), and what policy actions the predictions can support.

### The Horizon Spectrum

Operational systems in the literature span seven orders of magnitude in temporal granularity, from real-time hourly forecasts to multi-decade structural projections. Understanding where on this spectrum a system operates is the precondition for understanding everything else about it.

|Horizon                |System / Study                                |What’s Predicted                                              |Dominant Signals                                                   |Use Case                                        |
|-----------------------|----------------------------------------------|--------------------------------------------------------------|-------------------------------------------------------------------|------------------------------------------------|
|**Hours – Days**       |EMBERS (2014, 2016)                           |Daily protest events in 10 Latin American countries           |Twitter, news, food prices, currency, TOR                          |Tactical situational awareness; analyst alerting|
|**1 Month**            |CoupCast; ViEWS one-step-ahead                |Country-month coup probability; conflict probability          |ICEWS event flows + Polity + WDI structural                        |Near-term operational risk briefing             |
|**3 Months**           |MLP (Chen et al. 2022); Macis 2024 autoencoder|Civic space shocks; sudden unrest outbreaks                   |MLP NLP event counts; ACLED + GDELT anomaly signals                |Civil society strategic preparation             |
|**6–12 Months**        |ViEWS standard horizon; Fernandes GEWS 2026   |Conflict onset; multi-type instability                        |Structural (V-Dem, WDI) + lagged event history                     |Diplomatic and humanitarian planning            |
|**1–2 Years**          |IMF WP (Barrett et al. 2021); Fox 2021        |Country-year unrest; adverse regime change                    |WDI + IMF macro + RSUI history + Polity                            |Annual risk reports; foreign aid prioritization |
|**2 Years**            |Goldstone et al. 2010 PITF model              |Adverse regime change, ethnic war, revolutionary war, genocide|Polity + infant mortality + state-led discrimination + neighborhood|Strategic intelligence assessments              |
|**3 Years (36 months)**|ViEWS extended horizon                        |State-based, non-state, one-sided violence at PRIO-GRID level |UCDP-GED history + structural; MC dropout for uncertainty          |Long-horizon risk monitoring                    |
|**5 Years**            |O’Brien 2002 FASE; Basuchoudhary 2018         |Country-instability intensity classes                         |Structural macro + governance indicators                           |Long-range strategic planning                   |
|**Decades (to 2050)**  |Hegre et al. 2013 ISQ                         |Long-run conflict incidence trajectories                      |Demographic/development trends; Monte Carlo simulation             |Theory testing; long-run aid policy             |

### How Horizon Constrains Signal Choice

A core insight from the field that is underappreciated by newcomers: **the relevant predictors at different horizons are essentially disjoint**. This is because the signals operate on different timescales of causation.

At the **hours-to-days** horizon, the only useful signals are real-time event indicators — what is being said on Twitter, what news headlines are appearing, what Tor traffic spikes look like, how food prices changed yesterday. Structural indicators like GDP per capita are nearly useless because they don’t move on this timescale. EMBERS’ architecture reflects this: each constituent sub-model is trained on a real-time data stream, with the PSL fusion layer combining them into a daily protest forecast.

At the **monthly** horizon, lagged event flows become the dominant signal — last month’s ACLED counts predict this month’s, with the autocorrelation typically explaining most variance. The IMF WP finding that lagged RSUI values are top-10 SHAP predictors reflects this. ViEWS’ one-step-ahead modeling explicitly time-shifts predictors with respect to outcomes for each forecast step (t+1, t+2, … t+36), training a separate model link function for each horizon — recognition that the relevant signal structure changes as the horizon extends.

At the **annual** horizon, structural conditions begin to dominate. Goldstone et al.’s four PITF predictors (regime type, infant mortality, state-led discrimination, neighborhood instability) are all slow-moving structural variables that produce 80%+ accuracy at the 2-year horizon. The Fernandes (2026) finding that population size and GDP per capita are the top SHAP predictors at the country-year level reflects the same pattern.

At the **multi-year** horizon, only the slowest-moving demographic and economic trends remain useful — and even these become stochastic enough that models converge toward simulating distributions of futures rather than predicting specific events. Hegre et al.’s 2013 model uses dynamic multinomial logit with Monte Carlo simulation precisely because point prediction at 30+ year horizons is not credible; the model produces probability distributions over conflict incidence trajectories.

### The Calibration vs. Horizon Trade-off

All forecasting systems exhibit performance degradation as horizon extends. The HydraNet paper (2025) provides quantitative evidence from ViEWS data: state-based violence forecasts show consistent temporal degradation across all metrics, with Mean Squared Error and Brier Score rising sharply after the first year and AUC gradually losing discriminative strength. Non-state violence is comparatively more stable but declines in the latter half of the 36-month window. One-sided violence exhibits persistent volatility throughout.

This produces a fundamental design tension: short-horizon systems are more accurate but less actionable (by the time you have high-confidence prediction of unrest in 48 hours, prevention windows have closed); long-horizon systems are more actionable but less accurate (you can act on a 2-year warning, but the prediction itself carries large uncertainty). Most operational systems address this by producing multi-horizon outputs simultaneously — ViEWS publishes 1-month through 36-month forecasts in the same release; CoupCast publishes monthly probabilities aggregated to running 6-month and 12-month windows — leaving the user to choose the horizon appropriate to their decision context.

### Lead Time vs. Recall: The Operational Trade-off

Muthiah et al.’s 2016 retrospective on four years of EMBERS operation introduced the most important practical concept in this domain: the distinction between **cause uncertainty** (why will it happen?) and **timing uncertainty** (when exactly?). EMBERS achieved its strongest performance on planned, organized events because organizational signals (call-to-action posts, mobilization keywords on social media) appear days to weeks ahead. It performed worst on spontaneous, reactive events because no organizational signal exists by definition. This trade-off is structural to the prediction problem and cannot be eliminated by better models — it can only be navigated by choosing the right combination of horizon and outcome type.

-----

## Part III — Where Labels Come From: Off-the-Shelf vs. Custom-Labeled Data

The dominant paradigm in this literature is not custom labeling but creative repurposing of pre-existing event datasets. This shapes everything: the systematic biases of the source datasets propagate into models, the geographic and temporal coverage of the source dataset constrains the model’s coverage, and the conceptual definitions baked into the source dataset’s coding rules become the implicit theoretical commitments of the model. Understanding the off-the-shelf vs. custom-label decision is therefore central to understanding what any instability ML system can credibly claim.

### The Five Sources of Labels

#### Source 1 — Expert Hand-Coded Datasets (Off-the-Shelf, Researcher-Led)

These are the gold standard for academic conflict research. ACLED, UCDP-GED, the Powell-Thyne Coup Dataset, the PITF State Failure Problem Set, the Harff Genocide/Politicide list, and the Archigos leadership data all share a common methodology: trained human coders read source materials (news articles, academic histories, NGO reports) and apply explicit coding rules to produce structured records. Inter-coder reliability is reported; coding disagreements are resolved by senior staff; codebooks are public.

**Strengths of this source:** high construct validity (the coding scheme is theory-driven), explicit definitional choices (you can read the codebook to see exactly what counts as a coup), reasonable temporal coverage (UCDP from 1989, ACLED from 1997, Powell-Thyne from 1950), and academic legitimacy.

**Weaknesses:** limited geographic coverage in early years (ACLED’s Africa coverage from 1997, expansion to global only in 2018), source-language bias (English-dominant news sources under-cover conflict in non-English-speaking regions), event-not-process focus (ACLED records discrete events, missing slow-burn dynamics), and substantial lag between event and inclusion in the dataset.

Eck (2012) and Raleigh & Kishi (2020) have published systematic comparisons showing that UCDP and ACLED, despite both being expert-coded, produce significantly different event counts for the same country-period — UCDP being more conservative and ACLED more inclusive of low-intensity events. This is consequential: a model trained on UCDP-derived labels will learn different patterns than one trained on ACLED-derived labels, even when ostensibly predicting the “same” outcome.

#### Source 2 — Automated Event Coding (Off-the-Shelf, Algorithmic)

GDELT and ICEWS represent the alternative paradigm: automated NLP pipelines extract structured (Actor1, Event, Actor2) triples from news articles in near-real-time using the CAMEO event taxonomy. GDELT processes news in 100+ languages globally with 15-minute latency; ICEWS uses a more conservative pipeline with daily updates.

**Strengths:** massive scale (GDELT contains over 3.2 trillion data points), real-time availability, and global coverage including non-English sources. The Goldstein Scale assigns a numeric impact score to each CAMEO event type, providing a built-in continuous variable for model use.

**Weaknesses are substantial and well-documented:** the Nature Humanities & Social Sciences Communications study by Raleigh, Kishi & Linke (2023) demonstrates systematic miscategorization in both GDELT and ICEWS, including bias toward excessively violent designations (over-inflation of −10 Goldstein scores), event duplication, and false positives. ICEWS reportedly recorded 25 events in the United States in June 2019 between the US government and Iran rated at the −10 nuclear-conflict Goldstein level — events that were actually a verbal exchange. This is not an isolated coding error; it is structural to automated event extraction.

The implication for ML practitioners is critical: GDELT and ICEWS are excellent for measuring trends in news attention to conflict, but they should not be treated as ground truth for actual conflict events. Using them as labels (rather than predictors) in supervised learning will train models to predict news coverage patterns, not actual underlying instability — and the systematic biases in coverage will propagate into systematic biases in predictions.

#### Source 3 — Structural Indicator Indices (Off-the-Shelf)

V-Dem, Polity, Freedom House, the Worldwide Governance Indicators, and the Political Terror Scale are not event datasets — they are expert-coded annual measures of political institutions and civil liberties. They are typically used as predictors but occasionally as labels: V-Dem polyarchy declines of 0.05+ sustained over multiple years are used to define democratic backsliding episodes (the ERT dataset). The 2024 V-Dem release covers 200+ countries, 470+ indicators, back to 1789.

These indices are built from country-expert assessments aggregated via item response theory or similar measurement models. The strengths are theoretical grounding and conceptual clarity (V-Dem’s seven democracy components are explicitly theorized). The weaknesses are coding lag (annual updates, several months after year-end), index volatility from coding updates rather than substantive change, and the difficulty of using slow-moving indices to predict fast-moving events.

#### Source 4 — Newspaper Mention Indices (Off-the-Shelf, Derivative)

The Reported Social Unrest Index (Barrett et al. 2020, IMF) and the Conflict News Index (Chadefaux 2014) are constructed by counting newspaper articles matching specific keyword patterns in Dow Jones Factiva or LexisNexis. RSUI counts articles mentioning a country alongside protest/riot/revolution keywords near “unrest”; Chadefaux’s index counts conflict-related news items in a century-long historical archive.

These are arguably the cleanest middle ground between automated event coding (high noise) and expert hand-coding (slow, expensive). They are reproducible by anyone with Factiva access, they capture media-attention-to-instability rather than instability itself (which can be a feature: media attention is often what triggers policy response), and they provide consistent monthly updates. The cost is the language and source bias of the underlying news aggregator. RSUI’s English-language bias means it may systematically under-detect unrest in non-English-language media environments.

#### Source 5 — Custom-Labeled Datasets (Project-Specific)

This is the smallest category but contains some of the most innovative recent work. Three distinct subcategories appear in the literature:

**First, expert annotation of NLP outputs:** the CEHA dataset (2024) provides human-annotated conflict event detection labels for the Horn of Africa, used to fine-tune BERT, RoBERTa, T5, and LLM models. This is custom labeling in service of building better automated coding pipelines.

**Second, custom event classification on text streams:** the Machine Learning for Peace project at DevLab@Penn built a custom NLP pipeline that scrapes 70+ million articles from 100+ international, regional, and domestic online news sources in 22 languages, and trained custom classifiers to identify reporting on 20 civic space event types and 22 RAI (relevant actor interaction) event types. This is a hybrid approach: the underlying news text is off-the-shelf, but the event categorization is custom-trained for civil-society-relevant outcomes that off-the-shelf datasets do not capture.

**Third, satellite imagery + ACLED hybrid labeling:** Warnke & Runfola (2024) use ACLED-coded riot events as positive labels and DEGURB-classified urban areas without ACLED events as negative labels. The ResNet18 CNN learns from raw satellite pixels to distinguish them. The labels are off-the-shelf (ACLED), but the construction of contrastive negative cases via DEGURB urbanization data is a custom design choice that fundamentally shapes what the model learns.

### The Cost-Benefit of Custom Labeling

The reason custom labeling is rare in this field is that it is enormously expensive. Conflict event annotation requires domain expertise (you cannot crowdsource it on Mechanical Turk), produces low throughput (a trained annotator may process only dozens of articles per hour), and demands rigorous inter-annotator agreement protocols. The MLP project’s 70-million-article corpus is impossible to fully hand-annotate; it relies on training NLP classifiers on smaller hand-annotated subsets and applying them at scale. The CEHA paper notes the high cost as the reason most conflict NLP work uses off-the-shelf event datasets despite their known limitations.

However, three trends are reducing this cost:

1. **Large language model pre-labeling with human verification** — the workflow now standard at platforms like Label Studio, Prodigy, and Amazon SageMaker Ground Truth.
1. **Active learning approaches** that prioritize the most informative cases for human review — the Häffner & von der Maase work at ViEWS uses this for rare-event detection in education attacks.
1. **Synthetic data augmentation using LLMs** — early-stage but promising.

The economic case for custom labeling will improve substantially over the next few years, and the field will likely move toward more outcome-specific custom datasets rather than the current pattern of forcing all questions into the off-the-shelf event taxonomies.

### Sourcing Decision Matrix

The table below summarizes the trade-offs across the five sourcing approaches for practitioners making this choice.

|Source Type                                       |Cost                                      |Latency          |Best For                                                      |Worst For                                                              |
|--------------------------------------------------|------------------------------------------|-----------------|--------------------------------------------------------------|-----------------------------------------------------------------------|
|**Expert hand-coded** (ACLED, UCDP, PITF)         |Free (academic) / Subscription (real-time)|Days to months   |Standard binary onset prediction; academic benchmarking       |Real-time alerting; non-English coverage; outcomes outside the codebook|
|**Automated event coding** (GDELT, ICEWS)         |Free (GDELT) / restricted (ICEWS)         |15 min – 24 hours|Near-real-time signal extraction; trend detection             |Ground-truth labels; outcomes requiring conceptual nuance              |
|**Structural indices** (V-Dem, Polity, FH, WGI)   |Free                                      |6–12 months      |Long-horizon structural risk; predictor variables             |Short-horizon prediction; event-driven analysis                        |
|**Newspaper mention indices** (RSUI, Chadefaux)   |Subscription (Factiva)                    |1 month          |Country-month unrest detection; long historical analysis      |Distinguishing actual events from media coverage                       |
|**Custom labeling** (MLP, CEHA, satellite hybrids)|High (custom annotation)                  |Project-specific |Outcomes outside off-the-shelf taxonomy; novel data modalities|Quick prototypes; small teams without annotation infrastructure        |

-----

## Part IV — Synthesis: Practical Implications for System Design

### Three Decisions That Cascade Through Everything

The three dimensions analyzed in this document — what is predicted, on what horizon, with what label source — are not independent design choices. They form a tightly coupled system where each decision constrains the others. The practical synthesis can be expressed as three coupled decisions that any new system must make explicitly:

- **Decision 1 — Outcome family.** Choose between binary onset, continuous count, ordinal class, change detection, multi-type aggregation, or spatial location prediction. This decision determines the entire downstream architecture: model class (classification vs. regression), evaluation metric (AUPRC vs. MSE vs. coverage), and policy use case.
- **Decision 2 — Forecast horizon.** Choose where on the spectrum from real-time to multi-decade you operate. This decision determines which signals are useful (real-time event flows for short horizons; structural indicators for long horizons), which models are appropriate (sequence models for short horizons; tabular models for long horizons), and what you can credibly claim (point predictions for short horizons; probability distributions for long horizons).
- **Decision 3 — Label source.** Choose between off-the-shelf event datasets (with their inherited biases), automated coding (with their reliability problems), structural indices (with their coverage gaps), newspaper indices (with their language bias), or custom labeling (with its cost). This decision determines what you are actually predicting, regardless of what you call it.

### Common Patterns from the Reviewed Systems

The most successful operational systems make these three decisions in specific consistent combinations:

- **EMBERS pattern:** short-horizon (hours-days) + binary protest event + automated event coding (GDELT) + custom Twitter/news mining. *Use case:* tactical analyst alerting in a specific regional context.
- **ViEWS pattern:** medium-horizon (1-36 months) + continuous fatality count + UCDP-GED expert hand-coded labels + V-Dem and WDI structural predictors. *Use case:* humanitarian and diplomatic planning.
- **CoupCast pattern:** medium-horizon (monthly probability) + binary coup attempt + Powell-Thyne expert hand-coded labels + Polity and Archigos structural predictors. *Use case:* diplomatic and intelligence briefing.
- **IMF/Fernandes pattern:** long-horizon (annual) + binary multi-type instability + RSUI/PITF/aggregate event coding + WDI/IMF macro structural predictors. *Use case:* strategic risk reports for international financial institutions.
- **MLP pattern:** medium-horizon (3-month) + binary shock detection + custom NLP-classified civic space events + custom news + economic predictors. *Use case:* civil society organization strategic preparation.
- **Warnke-Runfola pattern:** temporally-decoupled spatial prediction + binary riot location + ACLED hand-coded labels + custom satellite imagery negative cases. *Use case:* pre-positioning of humanitarian or security resources.

### What the Field Has Not Yet Solved

Three substantial gaps remain unresolved as of 2026.

**First, the contagion problem:** Hegre, Nygård & Landsverk’s (2021) retrospective of nine years of ViEWS forecasts demonstrates that conflict diffusion across cultural and linguistic spheres (the Arab Spring being the canonical example) is poorly predicted by current models. GNN approaches are emerging but have not been fully validated at scale.

**Second, the rare-event escalation problem:** predicting which low-intensity unrest will escalate to full conflict remains poorly solved. The 25-fatality UCDP threshold creates a binary divide where most country-months sit at zero and a few sit at thousands; modeling the escalation transition itself, rather than the threshold crossing, is the frontier where Bayesian deep learning and conformal prediction methods are being developed.

**Third, the construct validity problem:** as discussed at length in Part I, the labels we use to train models embed substantive theoretical claims about what counts as instability. The field has not produced a systematic comparison showing how model rankings change when the same predictors are used against different label definitions of the “same” outcome. Until this is done, claims about which models are “best” should be heavily caveated.

-----

## Appendix — New Sources Surfaced in This Synthesis

The following sources were added during the targeted snowball search for this document and were not in the prior literature review artifacts. They are organized by which dimension of the analysis they support.

### On Time Scale and Horizon

- **Hegre, H., Allansson, M., Basedau, M., Colaresi, M., Croicu, M., Fjelde, H., Hoyles, F., Hultman, L., Högbladh, S., Jansen, R., Mouhleb, N., Muhammad, S.A., Nilsson, D., Nygård, H.M., Olafsdottir, G., Petrova, K., Randahl, D., Rød, E.G., Schneider, G., von Uexkull, N., Vestby, J. (2019).** “ViEWS: A political violence early-warning system.” *Journal of Peace Research* 56(2): 155–174. — definitive description of the one-step-ahead modeling approach for 1-36 month horizons.
- **HydraNet (2025).** “Next-Generation Conflict Forecasting: Unleashing Predictive Patterns through Spatiotemporal Learning.” arXiv:2506.14817. — provides quantitative documentation of forecast performance degradation across the 36-month horizon, with Monte Carlo Dropout for uncertainty quantification.
- **ShapeFinder / “The geometry of conflict” (2026).** arXiv:2604.21067. — documents 6-month rolling evaluation periods used as the practical compromise between forecast horizon and evaluation observation count.
- **ViEWS Forecasting Project** (viewsforecasting.org). — current operational platform publishing 1-36 month country-level and PRIO-GRID-level forecasts globally.

### On Dependent Variable Construction

- **Mueller, H. & Rauh, C. (2022).** “The Hard Problem of Prediction for Conflict Prevention.” *Journal of the European Economic Association* 20(6): 2440–2467. — defines “hard onset” as conflict emerging after at least 5 years of peace; demonstrates that PR-AUC drops to 0.127–0.363 even for state-of-the-art models on this harder definition.
- **Randahl, D. et al. (2024).** “Bin-Conditional Conformal Prediction of Fatalities from Armed Conflict.” *Political Analysis*. — provides framework for prediction intervals on continuous fatality outcomes, addressing the brittleness of point predictions on heavy-tailed distributions.
- **UCDP Methodology Documentation (2024).** “UCDP Methodology.” Department of Peace and Conflict Research, Uppsala University. — definitive source on the 25-battle-deaths threshold and the path-dependent persistence rule for conflict-dyad inclusion.
- **Beyond Closed Doors (2025).** “An Open-Source AI Framework for Forecasting Armed Conflict.” — documents the use of TabPFN, AutoGluon, and conformal methods on the hard onset prediction problem; achieves PR-AUC of 0.363 with AutoGluon on the 5-year-peace-threshold definition.

### On Label Source and Data Provenance

- **Raleigh, C., Kishi, R. & Linke, A. (2023).** “Political instability patterns are obscured by conflict dataset scope conditions, sources, and coding choices.” *Humanities and Social Sciences Communications* 10:74. — demonstrates that GDELT and ICEWS systematically miscategorize events (bias toward −10 Goldstein scores), with documented examples including ICEWS coding US-Iran verbal exchanges as nuclear-level conflict events. **Critical reading for anyone using automated event coding as labels.**
- **Eck, K. (2012).** “In Data We Trust? A Comparison of UCDP GED and ACLED Conflict Events Datasets.” *Cooperation and Conflict* 47(1): 124–141. — foundational comparison showing UCDP and ACLED, despite both being expert-coded, produce significantly different event counts for the same country-period.
- **Dietrich, N. & Eck, K. (2020).** “Known unknowns: media bias in the reporting of political violence.” *International Interactions*. — documents systematic media bias affecting all news-derived event datasets; foundational for understanding label construct validity.
- **CAMEO Codebook v0.9b6 (2009).** Pennsylvania State University Event Data Project. — definitive specification of the Conflict and Mediation Event Observations taxonomy underlying both GDELT and ICEWS event classification.
- **Häffner, S. & von der Maase, S.P. (2025).** ISA presentation at University of Pittsburgh HAIL seminar. — documents an active learning approach for custom labeling of rare-event datasets (attacks on education), combining LLMs, synthetic data, and active learning for efficient annotation. Indicative of where custom labeling methodology is heading.

-----

*Synthesizes prior literature review, dataset/variable, and model-type artifacts (this thread) with targeted snowball search across ViEWS publications, UCDP methodology documentation, IMF working papers, GDELT/ICEWS critical literature, and CAMEO codebook references.*