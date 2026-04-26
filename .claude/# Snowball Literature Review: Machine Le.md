# Snowball Literature Review: Machine Learning for Political Instability Prediction
## Social Unrest · Coups · Civil & Revolutionary War · Regime Collapse · Politicide/Genocide

**Compiled April 2026 | Snowball expanded from initial review**

---

## HOW TO READ THIS DOCUMENT

Sources are organized into five thematic sections matching the priority order specified. Within each section, entries are ranked by influence tier:

- **Tier 1 — Foundational/Canonical:** Work that established the subfield, is widely cited as a benchmark, or introduced a dominant methodology.
- **Tier 2 — Significant Contribution:** Strong empirical or methodological advance; widely cited in subsequent work.
- **Tier 3 — Targeted/Applied:** Single-country or domain-specific contributions; useful for practitioners.

A final cross-cutting section covers methodological papers applicable across all domains.

---

## SECTION 1: SOCIAL UNREST & MASS PROTEST PREDICTION

### Tier 1 — Foundational

**[SU-1] Ramakrishnan, N. et al. (2014). "Beating the news with EMBERS: Forecasting Civil Unrest using Open Source Indicators." *KDD '14, ACM SIGKDD*. arXiv:1402.7035**

The foundational deployed system in the field. EMBERS (Early Model Based Event Recognition using Surrogates) is a 24×7 automated pipeline forecasting civil unrest across 10 Latin American countries using tweets, news, blogs, food prices, currency exchange rates, TOR traffic, and Wikipedia edits. A multi-model architecture with probabilistic soft logic (PSL) fusion layer combines five sub-models. Operated under IARPA's Open Source Indicators program, independently evaluated by MITRE. Successfully forecast the June 2013 Brazilian Spring. The benchmark architecture for real-time open-source unrest forecasting.

- **Data:** Twitter, RSS news, GDELT-adjacent news, food price indices, currency exchange rates, TOR traffic, Wikipedia
- **Models:** LASSO logistic regression per data source; PSL-based model fusion; maximum entropy suppression
- **Key finding:** Lead time was easiest to optimize; planned events identifiable weeks ahead via organizational signals in social media

---

**[SU-2] Muthiah, S. et al. (2016). "EMBERS at 4 years: Experiences operating an Open Source Indicators Forecasting System." *KDD '16, ACM SIGKDD*. arXiv:1604.00033**

Retrospective on four years of 24×7 EMBERS operation. Introduces the distinction between cause uncertainty (why?) and timing uncertainty (when?). Documents lessons from a forecasting tournament across 10 countries and three languages. Essential for anyone building a production system.

- **Key finding:** Recall on spontaneous, unplanned events was the greatest system weakness; planned events forecasted well through organizational signals

---

**[SU-3] Barrett, P., Appendino, M., Nguyen, T., & de Leon Miranda, J. (2021). "Forecasting Social Unrest: A Machine Learning Approach." *IMF Working Paper 2021/263*.**

Globally-scoped study producing a social unrest risk index for 125 countries from 1996–2020 using 340+ macro-financial, socioeconomic, and political indicators. Ground truth is the Reported Social Unrest Index (RSUI). SHAP values reveal food price inflation, mobile penetration, and prior unrest as top drivers. ~67% accuracy forecasting unrest onset one year ahead.

- **Data:** World Bank WDI, IMF databases, FAO food prices, RSUI, Polity IV, Freedom House
- **Models:** Gradient Boosted Trees; SHAP explainability; Logistic Regression (baseline)
- **Key finding:** Coups and violent unrest are systematically harder to predict than other unrest types and carry larger macroeconomic consequences

---

**[SU-4] Deng, S. & Ning, Y. (2021). "A Survey on Societal Event Forecasting with Deep Learning." *arXiv:2112.06345*.**

Comprehensive taxonomy of deep learning methods applied to civil unrest and crime prediction. Covers all major event data sources (ACLED, GDELT, ICEWS, GSR, RSUI), structural indicators, and model architectures (LSTM, BiLSTM, GNN, Transformer). The single most complete methodological reference for the field.

- **Key finding:** Deep learning improves recall on rare event types but introduces interpretability problems unacceptable in policy contexts; hybrid structured+unstructured approaches are state-of-the-art

---

### Tier 2 — Significant Contribution

**[SU-5] Macis, L., Tagliapietra, M., & Meo, R. (2024). "Breaking the trend: Anomaly detection models for early warning of socio-political unrest." *Technological Forecasting and Social Change*. ScienceDirect.**

Novel anomaly detection framing developed in collaboration with the Italian Ministry of Foreign Affairs. Detects sudden deviations in behavioral patterns rather than direct classification. Combines ACLED and GDELT; AUC of 86.6–93.7% across three geopolitical case studies. Introduces XAI for diplomatic audit trails.

- **Data:** ACLED, GDELT
- **Models:** Isolation Forest-style anomaly detection; XAI explanations
- **Key finding:** Anomaly detection outperforms classification for detecting sudden *outbreaks* in low-baseline contexts

---

**[SU-6] Chitengu, R., Verkijika, S.F., & Mamabolo, K.E. (2025). "Forecasting Civil Unrest in South Africa Using Social Media Data: A Hybrid Machine Learning Approach." *Social Science Computer Review*. SAGE.**

Applies CRISP-DM methodology to 18,487 tweets (2019–2024) with ACLED ground truth. Hybrid Bi-LSTM + XGBoost model. Classification accuracy: 92% for protests, 86.2% for riots. R² of 33% for protest regression. SHAP identifies sentiment, engagement, region, day-of-week, holidays as top predictors.

- **Data:** Twitter/X API, ACLED
- **Models:** Bi-LSTM + XGBoost ensemble; SHAP
- **Key finding:** Temporal social media sentiment + event data delivers high classification accuracy; predicting magnitude (regression) remains much harder

---

**[SU-7] Combs, V. et al. (2018). "Predicting Social Unrest Using GDELT." In Stahlbock et al. (eds.) *Applied Data Mining for Business and Industry*. Springer.**

Uses GDELT to predict large-scale civil unrest in the US at county and state levels. Demonstrates that negative-sentiment news volume increases in weeks following triggering events. Tests Random Forest, Gradient Boosting, and Neural Networks.

- **Data:** GDELT (free, updated every 15 minutes, global coverage from 1979)
- **Models:** Random Forest, Gradient Boosting, Neural Networks
- **Key finding:** GDELT negative tone is a lagging but meaningful signal; subnational prediction feasible with careful feature aggregation

---

**[SU-8] Cadena, J. et al. (2015). "Forecasting Social Unrest using Twitter Data." Referenced in IMF WP 2021/263.** Studies the ability of Twitter data to forecast newspaper-identified unrest events in Brazil, Mexico, and Venezuela. One of the earliest papers demonstrating Twitter's predictive value for civil unrest events without requiring structural indicators.

---

**[SU-9] Timoneda, J.C. & Wibbels, E. (2022). "Detecting Democratic Backsliding Using Google Trends."** Uses Google Trends metadata—a purely open, near-real-time data source—as a predictor of protest dynamics and civic space constriction. Demonstrates that citizen search behavior contains latent signals of political discontent.

---

**[SU-10] Spaiser, V. et al. (2017). "Social Media and Regime Change: The Strategic Use of Twitter in the 2011–12 Russian Protests." *Journal of Information Technology & Politics*.**
Case study analysis of how Twitter coordination signals change in the weeks preceding mass protest events, specifically the 2011–12 Bolotnaya protests. Demonstrates the role of strategic framing and amplification in coordinating protest waves.

---

### Tier 3 — Targeted/Applied

**[SU-11] Ayetiran, E.F. (2023). "Social Unrest Prediction Through Sentiment Analysis on Twitter Using SVM: #EndSARS." *Open Information Science*. De Gruyter.**
Nigeria case study. Twitter binary SVM classifier for protest potential detection. Good single-country baseline; limited generalizability.

**[SU-12] Grill, G. (2021). "Future Protest Made Risky: Examining Social Media Based Civil Unrest Prediction Research and Products." *CSCW*. Springer.**
Critical review of 53 papers. Examines framing, motivations, and ethical risks of protest prediction systems. Essential reading for responsible deployment.

---

## SECTION 2: COUP D'ÉTAT PREDICTION

### Tier 1 — Foundational

**[CO-1] Powell, J. & Chacha, M. / University of Central Florida. CoupCast (2016–present).**

The leading operational coup forecasting system. Predicts monthly coup probability for every national leader worldwide. Training data: Powell-Thyne Coup Dataset (1950–present), GDP, infant mortality, election schedules, regime longevity, leader age and military background. Random Forest + ensemble autoregressive regression models trained in log-probability space, combined via Generalized Additive Model. Autoregressive training approach (retrain each year, incorporating previous predictions). Only ~600 positive examples globally — the canonical small-data challenge in political ML. Successfully predicted the 2021 upheavals in Chad and Mali.

- **Data:** Powell-Thyne Coup Dataset, World Bank WDI, Polity IV/V-Dem, ICEWS
- **Models:** Random Forest, Ensemble Autoregressive Regression, GAM
- **Key finding:** Combining RF and regression ensemble via GAM substantially outperforms either alone; coup prediction benefits equally from structural and event-flow signals

---

**[CO-2] Beger, A., Dorff, C.L., & Ward, M.D. (2014). "Ensemble Forecasting of Irregular Leadership Changes." *Research & Politics* 1(3).**

First publication of an ensemble split-population duration model for Irregular Leadership Changes (ILCs) — coups, successful rebellions, and protest-led leader removals — treated as a unified outcome class. Uses ICEWS event data, Archigos leadership data, and structural indicators for monthly forecasts across 168 countries. Shows Ukraine, Bosnia, Yemen, Egypt, and Thailand as highest-risk for mid-2014 ILCs.

- **Data:** ICEWS, Archigos Data (leader tenure/characteristics), Polity IV, World Bank WDI
- **Models:** Ensemble split-population duration regression; EBMA (Ensemble Bayesian Model Averaging)
- **Key finding:** Treating diverse forms of irregular leadership change as a unified class with a common ensemble model improves forecasting over specialized single-type models

---

**[CO-3] Ward, M.D. & Beger, A. (2017). "Lessons from near real-time forecasting of irregular leadership changes." *Journal of Peace Research* 54(2): 141–156.**

Systematic retrospective on the ILC forecasting program. Documents methodology for six-month rolling forecasts, discusses handling of missing data, temporal cross-validation, and calibration of ensemble outputs. Also develops tools for communicating probabilistic forecasts to non-technical analysts. The definitive operational lessons paper for the coup/ILC forecasting literature.

- **Data:** ICEWS, Polity IV, PITF, World Bank WDI
- **Models:** Seven thematic split-population duration models; EBMA fusion

---

### Tier 2 — Significant Contribution

**[CO-4] Beger, A., Dorff, C.L., & Ward, M.D. (2016). "Irregular Leadership Changes in 2014: Forecasts using ensemble, split-population duration models." *International Journal of Forecasting* 32(1): 98–111.**
Extended technical report on the 2014 ILC forecasts. Provides the most detailed methodology description for the split-population duration approach.

**[CO-5] Morgan, R., Beger, A., & Glynn, A. (2019). "Varieties of Forecasts: Predicting Adverse Regime Transitions." *V-Dem Working Paper 2019:89*.**
Applies V-Dem's high-dimensional governance indicators to predict adverse regime transitions (closest V-Dem analog to coups and state collapse). Uses Random Forest and gradient boosting. First paper systematically combining V-Dem's 400+ variables with temporal forecasting.

**[CO-6] Fox, S., Verhagen, M., & Cunningham, J. (2021). "Explainable models for forecasting the emergence of political instability." *PLOS ONE*. PMC8321219.**
Builds minimal logistic regression models using PITF data (1976–2017) to predict Adverse Regime Change, Revolutionary War, Ethnic War, and GenoPoliticide. Three variables — Polity Code, Infant Mortality, Years Since Last Instability — achieve competitive AUPRC of 0.108–0.115. Monte Carlo confidence intervals. The strongest argument in the literature for interpretable over black-box models in high-stakes policy contexts.

- **Data:** PITF State Failure Problem Set, Polity V, World Bank WDI
- **Key finding:** Policy makers are more likely to act on explainable predictions; minimal models can match complex ones

---

**[CO-7] Ward Lab / Predictive Heuristics (predictiveheuristics.com). ILC Forecasting Blog & arXiv writeups.**
Practitioner blog documenting six-month ILC forecasts as produced, including Polity transition matrix analysis after coups. Shows that 41% of successful coups result in status quo, 40% in worsening, only ~19% in democratization. Invaluable for modeling post-coup regime outcomes.

---

## SECTION 3: CIVIL WAR & ARMED CONFLICT PREDICTION

### Tier 1 — Foundational

**[CW-1] Ward, M.D., Greenhill, B.D., & Bakke, K.M. (2010). "The Perils of Policy by P-Value: Predicting Civil Conflicts." *Journal of Peace Research* 47(4): 363–375.**

The turning-point paper that reoriented conflict research from explanatory statistics to predictive validation. Demonstrates that many prominent civil war models (Fearon-Laitin; Collier-Hoeffler) perform poorly out-of-sample despite high in-sample statistical significance. Established out-of-sample prediction as a mandatory standard in conflict ML. Cited by virtually every subsequent conflict forecasting paper.

- **Key finding:** GDP per capita and population predict conflict where they occurred before, not when or where they will occur next; temporal prediction requires fundamentally different modeling approaches

---

**[CW-2] Muchlinski, D., Siroky, D., He, J., & Kocher, M. (2016). "Comparing Random Forest with Logistic Regression for Predicting Class-Imbalanced Civil War Onset Data." *Political Analysis* 24(1): 87–103.**

Landmark paper demonstrating that Random Forest substantially outperforms logistic regression (including rare-event and regularized variants) for civil war onset prediction. Civil war base rate ~1.6%; logistic regression achieves 98.4% accuracy by predicting peace everywhere. Random Forest (AUC 0.91) vs. Fearon-Laitin logit (AUC 0.77). Triggered a literature on ML methods for imbalanced political data. Note: subsequent replications by Wang (2018) and Neunhoeffer & Sternberg (2018) identified cross-validation errors; the core finding (RF > logistic) was upheld after correction, with gradient boosting outperforming both.

- **Data:** Sambanis civil war dataset (UCDP/PRIO basis)
- **Models:** Random Forest vs. Logistic Regression (3 variants)
- **Key finding:** Algorithmic methods substantially outperform logistic regression for rare-event binary classification in conflict data

---

**[CW-3] Colaresi, M. & Mahmood, Z. (2017). "Do the Robot: Lessons from Machine Learning to Improve Conflict Forecasting." *Journal of Peace Research* 54(2): 193–214.**

Introduces "Box's Loop" (build, compute, critique, think) as an iterative ML research design for conflict forecasting. Provides software tools for model criticism plots that visualize where and why models fail. Shows that iterative model criticism simultaneously improves out-of-sample performance and generates theoretical insights. Winner of JPR Best Visualization Award 2017.

- **Models:** Random Forest + Logistic Regression, iterative ensemble
- **Key finding:** Iterative model criticism — examining what the model gets wrong and why — is the most underutilized tool in conflict research and improves both prediction and theory

---

**[CW-4] Hegre, H. et al. (2019). "ViEWS: A Political Violence Early-Warning System." *Journal of Peace Research* 56(2): 155–174.**

The institutional gold standard for conflict forecasting. ViEWS produces monthly forecasts at country and subnational (PRIO-GRID) levels for 36 months ahead, covering all three UCDP types of organized violence. Large ensemble of thematic models (conflict history, economy, political institutions, geography) combined into ensemble outputs. Maximally transparent and publicly available.

- **Data:** UCDP-GED, PRIO-GRID, World Bank WDI, Polity, V-Dem, Ethnic Power Relations
- **Models:** Random Forest, Gradient Boosting, Neural Networks (ensemble)
- **Key finding:** Subnational (PRIO-GRID) ensemble models outperform country-level models; geographic disaggregation is essential for policy-relevant early warning

---

**[CW-5] Hegre, H., Nygård, H.M., & Landsverk, P. (2021). "Can We Predict Armed Conflict? How the First 9 Years of Published Forecasts Stand Up to Reality." *International Studies Quarterly* 65(3): 660–668.**

The only peer-reviewed evaluation of a conflict model's published forecasts against actual realized outcomes over nine years (2010–2018). Hegre et al. (2013) predictions held up for major conflict but performed worse for low-level incidence and failed to anticipate regional contagion effects (Arab Spring spread). Identifies the diffusion of conflict within cultural/linguistic spheres as a model gap.

- **Key finding:** Conflict models predict where conflict has been before; regional spillover modeling and contagion dynamics remain unsolved problems

---

**[CW-6] Mueller, H. & Rauh, C. (2018). "Reading Between the Lines: Prediction of Political Violence Using Newspaper Text." *American Political Science Review* 112(2): 358–375.**

Landmark NLP methodology paper. Uses LDA topic modeling on 3.8 million newspaper articles (1975–2015, 185 countries) to extract interpretable topic distributions. Panel regressions on within-country topic variation predict conflict onset one year ahead. Critical insight: within-country variation avoids the systematic bias of predicting conflict only where it has occurred before, enabling first-onset early warning.

- **Data:** Global newspaper text, UCDP-GED, Polity IV
- **Models:** LDA (unsupervised), Panel regression (supervised); Random Forest in follow-up
- **Key finding:** NLP topic features predict first-onset conflicts that all structural country-level data misses

---

**[CW-7] Mueller, H. & Rauh, C. (2022). "The Hard Problem of Prediction for Conflict Prevention." *Journal of the European Economic Association*. (Working paper: SSRN 3395185, 2019.)**

Follow-up targeting "hard cases" — first conflict onsets in previously peaceful countries, the most policy-relevant but hardest-to-predict events. LDA features fed into a Random Forest, which evaluates topics conditionally on conflict history via tree branching. Identifies stabilizing topics (negatively associated with onset) as key early warning signals.

- **Key finding:** Random Forest's conditional structure lets topic features function differently based on conflict history; supervised ML helps even with small samples for rare events

---

**[CW-8] Mueller, H., Rauh, C., & Seimon, B. (2024). "Introducing a global dataset on conflict forecasts and news topics." *Data & Policy* 6. conflictforecast.org**

Documents the open dataset of monthly country- and grid-level conflict forecasts from 2010–present, updated in real time at conflictforecast.org. Funded by FCDO. Provides methodology for the live operational system drawing on LDA newspaper topics + ACLED/UCDP data. The practitioner-accessible version of the Mueller-Rauh methodology.

---

**[CW-9] Chadefaux, T. (2014). "Early Warning Signals for War in the News." *Journal of Peace Research* 51(1): 5–18.**

Analyzes a century of newspaper articles (1900–present) across all interstate and intrastate conflicts. Derives a weekly conflict risk index from news volume alone. Conflict-related news item counts increase dramatically prior to war onset, providing up to 85% accuracy within three months. Simpler than Mueller-Rauh but historically earlier and covers interstate wars.

- **Data:** Historical newspaper corpus (LexisNexis-style), Correlates of War war list
- **Models:** Simple count-based risk index; logistic regression on news count + structural controls
- **Key finding:** News volume alone can predict war onset within weeks; contemporaries' own interpretation of events contains genuine predictive signal

---

**[CW-10] Anon. (2025). "Common Indicators Hurt Armed Conflict Prediction." *arXiv:2503.00265*.**

Critical methodological challenge to standard practice. Using ACLED's ~1 million conflict events (1997–2024) in Africa, demonstrates that adding structural indicators (GDP, population, terrain) to event-history-based models *degrades* out-of-sample performance. Conflict events are non-independent chains; treating them as such is essential.

- **Data:** ACLED (Africa), World Bank WDI
- **Key finding:** Standard structural indicators are redundant — and harmful — once ACLED event-history is included in the model

---

**[CW-11] Anon. (2024). "The Promise of Machine Learning in Violent Conflict Forecasting." *Data & Policy* (Cambridge Core).**

Comprehensive 2024 state-of-the-field review. Catalogs the full open-source data ecosystem, benchmarks historical performance, and documents emerging transformer/LLM approaches. The single most current comprehensive overview.

---

### Tier 2 — Significant Contribution

**[CW-12] Beger, A., Morgan, R.K., & Ward, M.D. (2021). "Reassessing the Role of Theory and Machine Learning in Forecasting Civil Conflict." *Journal of Conflict Resolution*.**
Re-examines the Blair-Sambanis (2020) claim that theory-based models outperform atheoretical ML. Finds that the result was conditional on using parametrically smoothed ROC curves (non-standard). Using empirical ROC curves, the advantage disappears. Important methodological corrective.

**[CW-13] Blair, R. & Sambanis, N. (2020). "Forecasting Civil Wars: Theory and Structure in an Age of Big Data and Machine Learning." *Journal of Conflict Resolution* 64(10): 1885–1915.**
Argues that theory-informed feature selection produces better-generalizing models than purely data-driven approaches. Even if the headline finding was later contested (see CW-12), the normative argument for theory-guided ML remains important.

**[CW-14] Hegre, H., Vesco, P., & Colaresi, M. (2022). "Lessons from an Escalation Prediction Competition." *International Interactions* 48(4): 521–554.**
Documents a formal forecasting competition where 15 international teams predicted changes in state-based violence (not just binary onset) for true future events. Ensemble methods from multiple teams outperformed any single team's model. Extends forecasting focus from onset to escalation dynamics.

**[CW-15] Vesco, P. et al. (2022). "United They Stand: Findings from an Escalation Prediction Competition." *International Interactions* 48(4): 860–896.**
Follow-on analysis from the escalation prediction competition. Documents the winning strategies: diverse feature sets (event data, text, structural), ensemble methods, and subnational spatial modeling.

**[CW-16] D'Orazio, V., Honaker, J., Prasady, R., & Shoemate, M. (2019). "Modeling and Forecasting Armed Conflict: AutoML with Human-Guided Machine Learning." *IEEE International Conference on Big Data*.**
Applies AutoML (automated machine learning pipeline search) to the ViEWS conflict forecasting problem. Shows that AutoML with minimal human guidance approaches the performance of carefully tuned human-designed models. Important for future platform automation.

**[CW-17] Muchlinski, D. (2013). "Machine Learning and Conflict Prediction: A Use Case." *Stability: International Journal of Security and Development*.**
Early applied study comparing supervised ML classifiers (RF, SVM, Neural Network) to logistic regression on ACLED sub-national African district data. Demonstrates superiority of Random Forest for imbalanced conflict data at sub-national granularity.

**[CW-18] Fearon, F. et al. (2021). "Prevention Is Better Than Cure: Machine Learning Approach to Conflict Prediction in Sub-Saharan Africa." *Sustainability* 13(13): 7366. MDPI.**
Grid-level (0.5°×0.5°) prediction across 48 sub-Saharan African countries. Combines ACLED, SPAM crop data, socioeconomic and geographic features. Random Forest outperforms logistic regression. Argues for including smaller-scale violence (riots, protests) as civil war precursors.

**[CW-19] Xue, Y., Schincariol, T., Chadefaux, T. et al. (2025). "Using machine learning to forecast conflict events for use in forced migration models." *Scientific Reports* 15, 28202.**
Hybrid ML + agent-based modeling pipeline that links conflict prediction to displacement forecasting. Shows that ML-based conflict forecasts at higher spatial-temporal resolution can substantially improve refugee movement predictions.

**[CW-20] Anon. (2025). "Do Large Language Models Know Conflict? Investigating Parametric vs. Non-Parametric Knowledge of LLMs for Conflict Forecasting." *arXiv:2505.09852*.**
Tests GPT-4/LLaMA for conflict forecasting with and without RAG (ACLED + GDELT context). RAG-augmented LLMs improve but still lag specialized ML models. LLMs excel at narrative reasoning, not quantitative magnitude forecasting.

**[CW-21] Hegre, H. et al. (2022). "Forecasting Fatalities in Armed Conflict: Forecasts for April 2022–March 2025." PRIO Working Paper.**
ViEWS operational forecasts using the updated system (ViEWS2020). Documents performance on the 2022 Ukraine escalation — a major out-of-sample test. Highlights the limits of purely data-driven approaches in predicting geopolitical shocks.

**[CW-22] Anon. (2024). "CEHA: A Dataset of Conflict Events in the Horn of Africa." *arXiv:2412.13511*.**
Introduces a new expert-annotated NLP benchmark dataset combining ACLED and GDELT for conflict event detection in the Horn of Africa. Tests BERT, RoBERTa, T5, and LLMs (GPT-4o, Mixtral). Valuable for transfer learning and domain adaptation in low-resource conflict NLP.

**[CW-23] Chadefaux, T. (2021). "A shape-based approach to conflict forecasting." *International Interactions*.**
Novel temporal pattern-matching approach that models time-series shapes of conflict escalation rather than point-in-time probabilities. Captures the trajectory of conflict buildup rather than just instantaneous risk levels.

---

## SECTION 4: REGIME CHANGE & STATE FAILURE PREDICTION

### Tier 1 — Foundational

**[RC-1] Goldstone, J.A., Bates, R.H., Epstein, D.L., Gurr, T.R. et al. (2010). "A Global Model for Forecasting Political Instability." *American Journal of Political Science* 54(1): 190–208.**

Developed under the CIA-funded Political Instability Task Force (PITF). Logistic regression model using four variables predicts political instability (including adverse regime change) two years ahead with 80%+ out-of-sample accuracy. Variables: regime type (partial democracy = highest risk), infant mortality, state-led ethnic discrimination, and regional neighborhood instability. The "anocracy gap" finding — that partial democracies are dramatically more unstable than full democracies or autocracies — is the field's most robust empirical result. Benchmark for all subsequent ML-based approaches.

- **Data:** PITF State Failure Problem Set, Polity IV, World Bank WDI
- **Models:** Logistic Regression (4 variables)
- **Key finding:** The anocracy gap; regime type is the single most powerful predictor of instability

---

**[RC-2] Hegre, H. et al. (2013). "Predicting Armed Conflict, 2010–2050." *International Studies Quarterly* 57(2): 250–270.**

Long-horizon simulation model forecasting armed conflict incidence globally to 2050 using dynamic multinomial logit. Transitions between "no conflict," "minor conflict," and "major conflict" states modeled jointly. The forecasts were then evaluated nine years later (see CW-5), demonstrating genuine predictive validity.

- **Data:** UCDP, World Bank WDI, Polity IV
- **Models:** Dynamic multinomial logit; Monte Carlo simulation for long-run projections
- **Key finding:** Conflict will decline globally in the long run primarily due to economic development, education, and democratic consolidation — but regional hot spots will persist

---

**[RC-3] Fox, S., Verhagen, M., & Cunningham, J. (2021). "Explainable models for forecasting the emergence of political instability." *PLOS ONE*. (Dual-listed with Section 2)**

Three-variable logistic regression (Polity, infant mortality, years of stability) covers adverse regime change, revolutionary war, ethnic war, and politicide. AUPRC 0.108–0.115 on PITF data (1976–2017). The definitive argument for minimal, explainable models in high-stakes policy prediction.

---

**[RC-4] King, G. & Zeng, L. (2001). "Improving Forecasts of State Failure." *World Politics* 53(4).**

Seminal application of rare-event logistic regression to state failure prediction. Introduces the concept of correcting for case-control sampling bias in rare-event data — the foundational statistical contribution to conflict and instability forecasting methodology. Precedes the ML wave but is cited as the statistical baseline virtually all ML papers benchmark against.

- **Data:** PITF State Failure Problem Set
- **Models:** Rare-event logistic regression (ReLogit)
- **Key finding:** Standard logistic regression underestimates the probability of rare events; rare-event correction is essential

---

**[RC-5] V-Dem Project (Coppedge et al., University of Gothenburg). Varieties of Democracy dataset and associated applied research. v15 (2025).**

The most comprehensive open democracy measurement dataset: 400+ governance indicators for 200+ countries from 1789–present. Multiple V-Dem publications apply Random Forest and gradient boosting to predict democratic backsliding, autocratization episodes, and regime transitions. The 2024 "State of the World" report introduces a "watchlist" initiative explicitly calling for early warning ML systems based on V-Dem indicators.

- **Data:** V-Dem (fully open, annual)
- **Key finding:** Erosion of judicial independence and free media precede autocratization by 2–5 years and are more reliable early warning signals than election outcomes

---

**[RC-6] Maerz, S.F., Edgell, A.B., Wilson, M.C., Hellmeier, S., & Lindberg, S.I. (2024). "Episodes of Regime Transformation." *Journal of Peace Research* 61(6): 967–984.**

Introduces the Episodes of Regime Transformation (ERT) dataset: a new conceptualization coding democratization and autocratization as obverse processes with 10 distinct pattern types and outcomes. Provides a more granular and theoretically grounded regime-change ground truth than Polity IV, enabling ML models to distinguish gradual backsliding from sudden collapse.

- **Data:** V-Dem (basis for ERT)
- **Key finding:** Only 32% of autocratization/democratization episodes result in full regime transition; most arrest or reverse — ML models need to capture episode trajectories, not just endpoints

---

### Tier 2 — Significant Contribution

**[RC-7] Ward, M.D. et al. (2012). "Improving Predictions Using Ensemble Bayesian Model Averaging." *Political Analysis* 20(3): 271–291.**
Early ensemble Bayesian model averaging (EBMA) application to regime/conflict predictions. Foundational for the ensemble methodology later used in ViEWS and the ILC project.

**[RC-8] Machine Learning for Peace Project (MLP) / DevLab @ Penn (2021–present). mlp.devlab.upenn.edu**
USAID-funded system forecasting 6-month changes in political conditions and civic space for 40+ countries. Uses local news data (in-country sources, not just international), ML models with interpretable SHAP outputs, and quarterly policy-facing dashboards. Distinctive for using locally produced information and incorporating local civil society validation.

- **Data:** Local and international online news (pipeline), economic indicators
- **Models:** Interpretable ML (LASSO, gradient boosting with SHAP)
- **Key finding:** Including locally produced news (not just international wire services) substantially improves early warning in autocratizing contexts

**[RC-9] Cederman, L.E. & Weidmann, N.B. (2017). "Predicting Armed Conflict: Time to Adjust Our Expectations?" *Science* 355(6324): 474–476.**
Commentary arguing that the inherent stochasticity and complexity of political violence imposes fundamental limits on conflict prediction accuracy, regardless of model sophistication. A necessary counterweight to overconfident forecasting claims.

---

## SECTION 5: GENOCIDE & POLITICIDE PREDICTION

### Tier 1 — Foundational

**[GP-1] Harff, B. (2003). "No Lessons Learned from the Holocaust? Assessing Risks of Genocide and Political Mass Murder Since 1955." *American Political Science Review* 97(1): 57–73.**

The founding quantitative forecasting model for genocide and politicide. Case-control design on 126 instances of internal war/regime collapse (1955–1997); 35 led to genocide/politicide. Six-factor logistic regression (regime type, prior genocide, exclusionary ideology, ethnic character of ruling elite, international trade openness, autocratic history) achieves 74% accuracy in distinguishing onset from non-onset. Cited by the US Genocide Prevention Taskforce as a central policy tool; reportedly used by the US Atrocities Prevention Board.

- **Data:** PITF Genocide/Politicide data, case-control sample
- **Models:** Logistic Regression (conditional case-control)
- **Key finding:** The combination of state failure + exclusionary elite ideology + autocracy + prior mass violence predicts genocide onset with 74% accuracy; economic openness is a surprising protective factor

---

**[GP-2] Goldsmith, B.E., Butcher, C.R., Semenovich, D., & Sowmya, A. (2013). "Forecasting the Onset of Genocide and Politicide: Annual Out-of-Sample Forecasts on a Global Dataset, 1988–2003." *Journal of Peace Research* 50(4): 437–452.**

First published *unconditional* (global, not case-control) annual out-of-sample genocide/politicide forecasts. Two-stage probit model: Stage 1 predicts political instability; Stage 2 predicts genocide given instability. Out-of-sample forecasts for 1988–2003 predict 90.9% of genocide onsets correctly while predicting 79.2% of non-onset years correctly. Identifies six of 11 genocide/politicide onsets within the top 5% of at-risk countries. A landmark advance over Harff (2003).

- **Data:** PITF State Failure Problem Set, Harff genocide/politicide data, World Bank WDI
- **Models:** Two-stage probit; Generalized Additive Model for ensemble
- **Key finding:** A global unconditional model can identify most genocide onsets within the top 5% of country-year risk assessments; two-stage design substantially outperforms single-stage

---

**[GP-3] Goldsmith, B.E. & Butcher, C.R. (2018). "Genocide Forecasting: Past Accuracy and New Forecasts to 2016–2020." *Journal of Genocide Research* 20(1): 90–107.**

Retrospective evaluation of the Atrocity Forecasting Project's (AFP) 2011–2015 predictions and new forecasts to 2020. Compares AFP accuracy against UN warnings and Genocide Watch alerts. Demonstrates that quantitative ML-assisted forecasts identified the Central African Republic and Myanmar as high-risk before international warnings were issued.

- **Key finding:** AFP correctly identified CAR and Myanmar as high-risk years before conventional diplomatic warnings; quantitative models provide complementary early warning value

---

**[GP-4] Butcher, C., Goldsmith, B.E., Nanlohy, S., Sowmya, A., & Muchlinski, D. (2020). "Introducing the Targeted Mass Killing Data Set for the Study and Forecasting of Mass Atrocities." *Journal of Conflict Resolution* 64(7–8): 1524–1547.**

Introduces the TMK (Targeted Mass Killing) dataset: 201 episodes (1946–2017) that improve on PITF data by clarifying intent measurement, enabling customizable thresholds (genocide = severity level 4+), explicitly identifying both state and non-state perpetrators, and providing start/end dates. The new benchmark dataset for ML-based genocide/mass atrocity prediction.

- **Data:** Combines PITF, Harff, and new coding; 201 TMK episodes vs. PITF's 44
- **Key finding:** PITF data substantially underestimates mass atrocity episodes; TMK enables more granular and reproducible ML models

---

**[GP-5] Anon. (2025). "Artificial Intelligence for Peace: An Early Warning System for Mass Violence." *Springer Nature / Book Chapter*. (ResearchGate: publication 395111459)**

Presents what the authors claim is the first RAG-augmented LLM-assisted genocide early warning system. Combines structured data (PITF, V-Dem, ACLED) with unstructured text inputs processed through a neural network classifier. Introduces AI-generated explanations for flagged country-years to make predictions auditable by prevention practitioners.

- **Data:** PITF, V-Dem, ACLED, news text
- **Models:** Neural network classifier + LLM narrative generation (RAG)
- **Key finding:** AI-generated explanations substantially improved analyst uptake of model predictions in simulation exercises

---

**[GP-6] Early Warning Project (USHMM). U.S. Holocaust Memorial Museum (2011–present). earlywarningproject.ushmm.org**

Operational annual statistical risk assessments for state-sponsored mass killing for all countries globally. Ensemble Bayesian approach combining three statistical models: (1) Harff model, (2) Goldstone et al. PITF model, and (3) Rost model. Adds expert opinion pool via structured analytic techniques. Published annually; cited in US government genocide prevention policy.

- **Data:** PITF, Polity IV, World Bank WDI, expert elicitation
- **Models:** Ensemble Bayesian forecast combining three base models + opinion pool
- **Key finding:** Combining quantitative models with expert opinion consistently outperforms either alone for rare mass atrocity events

---

### Tier 2 — Significant Contribution

**[GP-7] Verdeja, E. (2016). "Predicting Genocide and Mass Atrocities." *Genocide Studies and Prevention* 9(3).**
Comprehensive literature review of genocide prediction models from Harff (2003) through 2015. Maps the full landscape of quantitative approaches, data sources, and methodological evolution. Essential reading before entering the subfield.

**[GP-8] Ulfelder, J. & Valentino, B. (2008). "Assessing Risks of State-Sponsored Mass Killing." Working Paper, Political Instability Task Force.**
Non-conditional global risk assessment model for state-sponsored mass killing (broader than genocide). Employs cross-validation rather than in-sample accuracy. Notable for being among the first genocide-related papers to explicitly use out-of-sample validation.

**[GP-9] Bagozzi, B.E. & Koren, O. (2017). "Using Machine Learning Methods to Identify Atrocity Perpetrators." *SSRN Working Paper*.**
Applies supervised ML (Random Forest, SVM) specifically to the task of identifying which armed actor (state military, paramilitaries, rebel groups) is most likely to escalate to mass atrocity. Disaggregates atrocity prediction by perpetrator type.

**[GP-10] Leveraging a Multi-Method Approach to Improve Mass Atrocity Forecasting (2024). *Genocide Studies and Prevention* 18(1): 54–83.**
Sponsored by PITF/CIA. Combines PITF Mass Killings data (1972–2022) and TMK data (1972–2022) with multiple ML methods. Tests ensemble approaches vs. single models. Documents how combining two atrocity datasets with complementary coding schemes improves predictive performance.

---

## SECTION 6: CROSS-CUTTING METHODOLOGICAL CONTRIBUTIONS

These papers address methodological problems applicable across all five prediction domains.

---

**[M-1] Hegre, H., Metternich, N.W., Nygård, H.M., & Wucherpfennig, J. (2017). "Introduction: Forecasting in Peace Research." *Journal of Peace Research* 54(2): 113–124.**

The special issue introduction that synthesized the state of forecasting in peace and conflict research as of 2017. Reviews all major methodological approaches, data sources, and validation strategies. Defines best practices for temporal cross-validation, calibration, and uncertainty communication. Required reading for any new researcher entering the field.

---

**[M-2] Schrodt, P.A. (2014). "Seven Deadly Sins of Contemporary Quantitative Political Analysis." *Journal of Peace Research* 51(2): 287–300.**

Identifies the methodological failures endemic to conflict studies that limit forecasting utility: in-sample prediction, AUROC as the only metric, failure to address class imbalance, no temporal cross-validation, not computing confidence intervals, ignoring spatial dependence, and ignoring temporal autocorrelation. The checklist against which any forecasting system should be evaluated.

---

**[M-3] Kennedy, R. (2015). "Making Useful Conflict Predictions: Methods for Addressing Skewed Classes and Implementing Cost-Sensitive Learning in the Study of State Failure." *Journal of Peace Research* 52(5): 649–664.**

Systematic comparison of methods for handling class imbalance in rare-event political data: SMOTE, cost-sensitive learning, undersampling, threshold tuning. The definitive practical guide for handling the class imbalance problem central to all five prediction domains.

---

**[M-4] Montgomery, J.M., Hollenbach, F.M., & Ward, M.D. (2012). "Improving Predictions Using Ensemble Bayesian Model Averaging." *Political Analysis* 20(3): 271–291.**

Introduces Ensemble Bayesian Model Averaging (EBMA) to conflict and instability prediction. Demonstrates that EBMA consistently outperforms any single model across multiple political science forecasting tasks. The foundational ensemble methodology paper for the field.

---

**[M-5] Cranmer, S.J. & Desmarais, B.A. (2017). "What Can We Learn from Predictive Modeling?" *Political Analysis* 25(2): 145–166.**

Theoretical piece arguing for the coequal scientific status of prediction alongside explanation in political science. Addresses the "folk criticism" that forecasting is atheoretical. Argues that prediction uniquely tests models against future data and generates theoretically informative failures.

---

**[M-6] Rød, E.G., Gåsste, T., & Hegre, H. (2024). "A Review and Comparison of Conflict Early Warning Systems." *International Journal of Forecasting*.**

Systematic comparison of major operational EWS platforms (ViEWS, ACLED forecasts, ConflictForecast, CrisisWatch, USHMM Early Warning Project) across common benchmarks. Documents which systems perform better on onset vs. escalation vs. resolution prediction.

---

**[M-7] Saito, T. & Rehmsmeier, M. (2015). "The Precision-Recall Plot Is More Informative Than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets." *PLoS ONE*.**

Technical paper establishing that AUPRC (Area Under Precision-Recall Curve) is the correct evaluation metric for imbalanced classification problems like conflict and instability prediction. Frequently cited in conflict ML papers after 2016 as motivation for reporting AUPRC alongside AUROC.

---

**[M-8] Beck, N., King, G., & Zeng, L. (2000). "Improving Quantitative Studies of International Conflict: A Conjecture." *American Political Science Review* 94(1): 21–36.**

Pre-ML classic that introduced neural networks to conflict research. Demonstrated that neural networks could outperform logistic regression on Militarized Interstate Dispute data. The paper that launched a decade of ML interest in IR before the field's rediscovery in the 2010s. Also introduced the conjecture that predictive validation should be standard practice.

---

## OPEN-SOURCE DATA SOURCES: COMPREHENSIVE UPDATED TABLE

| Dataset | Coverage | Frequency | Primary Use | Open? |
|---|---|---|---|---|
| **ACLED** | Global, 1997–present | Real-time | Social unrest, armed conflict event ground truth | ✅ |
| **GDELT** | Global, 1979–present | 15-minute | News event coding, sentiment, tone (CAMEO coding) | ✅ |
| **UCDP-GED** | Global, 1989–present | Monthly/Annual | Battle deaths, organized violence ground truth | ✅ |
| **ICEWS** | Global, 1995–present | Daily | Diplomatic/political event coding (CAMEO) | ✅ (partial) |
| **ViEWS** | Global | Monthly | Conflict forecasting benchmark + fatality forecasts | ✅ |
| **PITF State Failure Problem Set** | Global, 1948–present | Annual | Political instability, revolutionary war, genocide | ✅ |
| **Polity V** | Global, 1800–2018 | Annual | Regime type, democratic institutions, anocracy | ✅ |
| **V-Dem (v15)** | Global, 1789–present | Annual | 400+ governance/democracy indicators | ✅ |
| **ERT Dataset** | Global, 1900–present | Annual | Episodes of regime transformation (autocratization/democratization) | ✅ |
| **Powell-Thyne Coup Dataset** | Global, 1950–present | Annual | Coup events (successful, attempted, alleged) | ✅ |
| **Archigos** | Global, 1875–present | Annual | Leader tenure, characteristics, exit type | ✅ |
| **TMK Dataset** | Global, 1946–2022 | Annual | Targeted mass killings, genocide/politicide | ✅ |
| **PITF Genocide/Politicide** | Global, 1955–present | Annual | Genocide/politicide episodes | ✅ |
| **Early Warning Project** | Global, 1946–2023 | Annual | State-sponsored mass killings | ✅ |
| **World Bank WDI** | Global, 1960–present | Annual | Macro-economic and social indicators | ✅ |
| **IMF Financial Statistics** | Global | Monthly/Annual | GDP, inflation, exchange rates | ✅ |
| **RSUI** | Global, 1996–present | Monthly | Newspaper-based unrest index | ✅ |
| **Twitter/X API** | Global, 2006–present | Real-time | Social media sentiment, organization signals | Conditional |
| **FAO FPMA** | Global | Monthly | Food price indices | ✅ |
| **Freedom House** | Global, 1972–present | Annual | Political rights, civil liberties | ✅ |
| **Correlates of War** | Global, 1816–2016 | Annual | Interstate wars, alliances, capabilities | ✅ |
| **Global Terrorism Database** | Global, 1970–2020 | Annual | Terrorism incidents | ✅ |
| **PRIO-GRID** | Global, 1946–present | Annual/Monthly | Spatial grid-level socioeconomic/geographic data | ✅ |
| **Ethnic Power Relations** | Global, 1946–present | Annual | Ethnic group political access and discrimination | ✅ |
| **Political Terror Scale** | Global, 1976–present | Annual | State repression and human rights | ✅ |
| **ConflictForecast.org** | Global, 2010–present | Monthly | LDA-based conflict risk scores (Mueller-Rauh) | ✅ |
| **Harff Genocide/Politicide** | Global, 1955–2001 | Annual | Original genocide/politicide episodes | ✅ |

---

## CITATION INDEX: PAPERS BY PUBLICATION TYPE

**Journal Articles (peer-reviewed):** SU-1 (KDD), SU-3 (IMF WP), SU-5 (ScienceDirect), SU-6 (SAGE), CW-1 (JPR), CW-2 (PA), CW-3 (JPR), CW-4 (JPR), CW-5 (ISQ), CW-6 (APSR), CW-9 (JPR), CO-2 (R&P), CO-3 (JPR), CO-6 (PLOS ONE), RC-1 (AJPS), RC-2 (ISQ), RC-6 (JPR), GP-1 (APSR), GP-2 (JPR), GP-3 (JGR), GP-4 (JCR), M-1 (JPR), M-2 (JPR), M-3 (JPR), M-4 (PA), M-5 (PA)

**arXiv Preprints:** SU-2 (arXiv), SU-4 (arXiv:2112.06345), CW-10 (arXiv:2503.00265), CW-20 (arXiv:2505.09852)

**Conference Papers:** SU-1/SU-2 (ACM KDD), CW-16 (IEEE Big Data)

**Operational Systems / Platforms:** SU-1/SU-2 (EMBERS), CO-1 (CoupCast), CW-4 (ViEWS), CW-8 (ConflictForecast), RC-8 (MLP), GP-6 (Early Warning Project/USHMM)

**Working Papers / Government Reports:** GP-6 (USHMM), CO-5 (V-Dem WP), GP-8 (PITF/OEF), GP-10 (PITF/CIA-funded)

---

*Document compiled April 2026 via systematic snowball search from the initial literature review. Core seed papers: IMF WP 2021/263, EMBERS (arXiv:1402.7035), Mueller & Rauh APSR 2018, Deng & Ning arXiv:2112.06345, Goldstone et al. AJPS 2010, CoupCast (UCF), Hegre et al. ViEWS JPR 2019. Secondary snowball chains followed citation networks through Semantic Scholar, Google Scholar, PRIO publication databases, arXiv cs.SI and cs.LG, and conflictforecast.org.*