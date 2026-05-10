# Model Types in Political Instability Prediction
## A Synthesis Across Social Unrest, Coup, Civil War, Regime Change & Genocide/Politicide Literature

---

## Overview

The field has passed through three methodological generations, and most high-performance operational systems today combine all three in ensemble architectures. This document maps each model type to the instability domains where it is used, the studies that use it, and its key strengths and weaknesses for rare-event political prediction.

The most consistent finding across all reviewed systems — EMBERS, ViEWS, CoupCast, the IMF Working Paper, and the Ward/Beger ILC forecasting program — is that **no single model type dominates**. Ensemble methods combining multiple model types trained on diverse data sources outperform any individual model, typically by 5–15 AUPRC points.

---

## Generation 1: Statistical Baselines (pre-2010)

### Logistic Regression

The canonical baseline that every subsequent ML paper benchmarks against. Dominated the field from the founding of the CIA-funded Political Instability Task Force (PITF) in 1994 through roughly 2010. Every study in the literature that introduces a new ML model tests it against at least one logistic regression variant.

**Domains used:** All five instability domains (unrest, coups, civil war, regime change, genocide)

**Representative studies:**
- Goldstone et al. (2010) — PITF political instability model; 4-variable logistic regression, 80%+ out-of-sample accuracy
- Harff (2003) — genocide/politicide model; 6-variable logistic regression, 74% accuracy
- King & Zeng (2001) — state failure prediction; introduced rare-event logistic regression (ReLogit)
- Mueller & Rauh (2018) — panel logistic regression on LDA topic features (fixed-effects variant)

**Variants in the literature:**

**Firth rare-event logistic regression (ReLogit)**
King & Zeng (2001) introduced this correction for the systematic downward bias in probability estimates for rare events in standard logit. It was the standard correction for class imbalance before ML methods arrived and remains the statistical baseline in most benchmark comparisons. Muchlinski et al. (2016) show that even this corrected form underperforms Random Forest for civil war onset.

**L1-regularized (LASSO) logistic regression**
Used by EMBERS as the sub-model for each individual data source (one model per data stream: Twitter, news, blogs, food prices, currency). LASSO performs automatic variable selection by shrinking uninformative coefficients to zero, producing sparse, interpretable models well-suited to high-dimensional text-derived features.

**Panel logistic regression (fixed-effects)**
Used by Mueller & Rauh (2018) for LDA newspaper topic features. Country fixed-effects absorb between-country structural differences so that only within-country temporal variation in topic proportions drives the prediction — the core methodological insight of that paper. This variant is specifically designed to predict *timing* of conflict onset rather than which countries are structurally at risk.

**Strengths:**
- Coefficients directly interpretable as odds ratios
- Well-understood statistical properties
- Computationally fast
- Widely accepted by policy audiences

**Weaknesses:**
- Performs poorly on class-imbalanced data (predicts majority class everywhere)
- Linear decision boundary by default; manual feature engineering required for nonlinear relationships
- Poor out-of-sample predictive performance even when in-sample significance is high — the central empirical finding of Ward, Greenhill & Bakke (2010)
- Sensitive to multicollinearity among predictors

---

### Split-Population Duration Models

A specialized statistical model used almost exclusively by the Ward/Beger/Duke group for coup and irregular leadership change (ILC) prediction. The model acknowledges that the population of countries is divided into two latent groups: those that are *at risk* of ever experiencing a coup (the "at risk" group) and those that are structurally immune (the "never" group). Two equations are estimated jointly: one predicting membership in the at-risk group, one predicting timing conditional on being in the at-risk group. This handles the structural excess of zeros in rare-event data more elegantly than standard logistic regression.

**Domains used:** Coups, irregular leadership changes

**Representative studies:**
- Beger, Dorff & Ward (2014) — first ILC ensemble using split-population duration models
- Beger, Dorff & Ward (2016) — extended technical report on 2014 ILC forecasts
- Ward & Beger (2017) — operational lessons from six-month rolling ILC forecasts
- CoupCast (Powell/UCF) — ensemble regression component uses autoregressive split-population structure

**Strengths:**
- Correctly models the two-stage structure of rare political events (structural immunity vs. timing)
- Handles excess zeros without resampling
- Ensemble of seven thematic models (each thematic focus: economy, politics, geography, events, etc.) combined via EBMA
- Produces calibrated probability outputs with interpretable lead-time estimates

**Weaknesses:**
- Complex to specify and estimate
- Requires assumption about latent at-risk group structure
- Less flexible than tree-based methods for nonlinear interactions
- Computationally intensive for large country-month panels

---

## Generation 2: Ensemble Tree Methods (2013–present, now dominant)

### Random Forest

The workhorse of modern conflict and instability prediction. Muchlinski et al. (2016) provided the landmark demonstration that Random Forest substantially outperforms all logistic regression variants on civil war onset (AUC 0.91 vs. 0.77 for the Fearon-Laitin logistic regression, after cross-validation correction). Every subsequent benchmark study includes Random Forest as either the primary model or a required comparison.

**Domains used:** All five instability domains; most common across civil war onset and conflict prediction

**Representative studies:**
- Muchlinski, Siroky, He & Kocher (2016) — definitive civil war onset comparison study (*Political Analysis*)
- CoupCast (Powell/UCF) — RF as one of two primary model architectures, combined with regression ensemble via GAM
- Mueller & Rauh (2019, "Hard Problem") — LDA newspaper topics fed into RF for first-onset civil war prediction; conditional structure via tree branching handles the hard case of peaceful countries
- Colaresi & Mahmood (2017) — iterative RF + logistic regression via Box's Loop
- ViEWS (Hegre et al. 2019) — RF as one of several thematic ensemble members
- Fearon et al. / Musumba et al. (2021) — RF across 48 African countries using ACLED + SPAM crop data
- Morgan, Beger & Glynn (2019) — RF for adverse regime transitions using V-Dem indicators
- Prevention Is Better Than Cure (MDPI 2021) — RF outperforms logit for sub-Saharan Africa conflict grids
- Goldsmith/Atrocity Forecasting Project (AFP) — RF in later model versions

**How it handles class imbalance:** The `class_weight="balanced"` parameter (scikit-learn) or `classwt` parameter (R `randomForest`) inversely weights minority class observations by their frequency. In extreme imbalance cases, SMOTE oversampling is applied to the training set before RF is fit.

**Key hyperparameters:**
- `n_estimators`: Number of trees (typically 100–1,000; more trees = lower variance, diminishing returns)
- `max_features`: Number of features considered at each split (typically `sqrt(p)` for classification)
- `max_depth`: Maximum tree depth (deep trees overfit; often left unlimited with min_samples_leaf constraint)
- `min_samples_leaf`: Minimum samples per leaf node (key regularizer for rare-event data)
- `class_weight`: Inverse-frequency weighting for class imbalance

**Strengths:**
- Handles class imbalance via weighting
- Nonlinear decision boundaries without manual feature engineering
- Handles mixed variable types (continuous, binary, categorical)
- Robust to correlated features and irrelevant predictors
- Produces variable importance rankings readable to non-technical analysts
- Does not require feature scaling
- Handles missing values reasonably via surrogate splits

**Weaknesses:**
- Black-box predictions require SHAP or permutation importance for explanation
- Computationally slower than gradient boosting on large datasets
- Less accurate than gradient boosting on tabular data when carefully tuned
- Variable importance can be biased toward high-cardinality features

---

### Gradient Boosted Trees (XGBoost / LightGBM / CatBoost)

The state-of-the-art single base learner for structured tabular political data as of 2025. Where Random Forest builds trees in parallel on bootstrap samples, gradient boosting builds trees sequentially — each tree corrects the residual errors of all previous trees. In practice, XGBoost and LightGBM consistently outperform Random Forest on tabular data with careful hyperparameter tuning. The IMF WP (2021) uses gradient boosting as its primary model over RF.

**Domains used:** All five instability domains; now the default primary model in the most recent generation of work

**Representative studies:**
- Barrett et al. / IMF WP (2021) — gradient boosted trees primary model; 340+ predictors; SHAP explainability
- Chitengu, Verkijika & Mamabolo (2025) — XGBoost as classification head in hybrid architecture
- ViEWS (Hegre et al. 2019, 2021) — gradient boosting as ensemble member
- Wang (2018), comment on Muchlinski (2016) — AdaBoosted trees and gradient boosted trees outperform RF once cross-validation errors are corrected

**Key hyperparameters:**
- `n_estimators` / `num_boost_round`: Number of boosting rounds
- `learning_rate` / `eta`: Step size shrinkage (smaller = more regularization, more trees needed)
- `max_depth`: Maximum tree depth (typically 3–6 for conflict data; shallower than RF)
- `subsample`: Row sampling rate per tree (stochastic boosting; typically 0.5–0.8)
- `colsample_bytree`: Column sampling rate per tree
- `scale_pos_weight` (XGBoost): Ratio of negative to positive class — primary class imbalance handle
- `min_child_weight`: Minimum sum of instance weight in a child (key regularizer for rare events)
- `reg_alpha` / `reg_lambda`: L1 and L2 regularization

**Strengths:**
- Best tabular prediction performance in benchmark comparisons
- Native handling of missing values (important for WDI/PITF data gaps in fragile states)
- Built-in regularization (L1, L2, subsampling) prevents overfitting
- `scale_pos_weight` for class imbalance
- Fast training enabling extensive hyperparameter search
- SHAP values integrate seamlessly — TreeSHAP is exact and fast
- LightGBM especially fast on high-dimensional data (leaf-wise growth)

**Weaknesses:**
- Requires more hyperparameter tuning than Random Forest
- Can overfit if learning rate is too high or trees too deep
- Less interpretable than logistic regression without SHAP
- Sensitive to outliers in the target variable (relevant for fatality counts)

---

### Decision Trees

Appear as baselines or in ensemble contexts. Rarely used as standalone models due to severe overfitting on political data. Used in early PITF-era work as exploratory tools for finding nonlinear thresholds in regime type variables (the original discovery of the "anocracy gap" involved inspecting decision tree splits on the Polity score). In modern work, decision trees appear only as the base learner within RF or GBT ensembles.

---

## Generation 3: Neural Networks & Deep Learning (2018–present, supplementary)

### LSTM and Bidirectional LSTM (Bi-LSTM)

The most common deep learning architecture in this literature, applied when political instability is modeled as a sequential time-series problem rather than a point-in-time classification. The key advantage over tree methods is that LSTMs capture long-range temporal dependencies — for example, a protest wave three months ago feeding into a coup risk this month — without requiring the researcher to manually specify lag structures.

**Domains used:** Social unrest, armed conflict escalation, subnational conflict sequences

**Representative studies:**
- Chitengu, Verkijika & Mamabolo (2025) — Bi-LSTM as temporal feature extractor feeding XGBoost; 92% protest classification accuracy (South Africa)
- Deng & Ning (2021) survey — identifies Bi-LSTM as consistently outperforming unidirectional LSTM across societal event prediction tasks
- Multiple ViEWS prediction competition entries (2021, 2022) — LSTM-based models for escalation forecasting

**Two common configurations in this literature:**

**Standalone LSTM** on country-month event sequences (ACLED fatality counts, GDELT Goldstein scores over time). Takes a sequence of T historical monthly observations and outputs a prediction for month T+1 or T+k.

**Hybrid Bi-LSTM + tree model:** LSTM or Bi-LSTM processes the time-series (event data) component while XGBoost or Random Forest processes structural (annual) features. Outputs from both branches are concatenated before the final classification layer. This is the architecture of Chitengu (2025) and several ViEWS competition-winning entries. It combines the temporal pattern recognition of LSTMs with the tabular data efficiency of gradient boosting.

**Key architectural parameters:**
- `hidden_size`: Number of LSTM units (hidden state dimensionality; typically 64–256)
- `num_layers`: Depth of stacked LSTM layers (typically 1–3)
- `dropout`: Dropout rate between LSTM layers (regularization; typically 0.2–0.5)
- `bidirectional`: Boolean flag for Bi-LSTM (reads sequence forward and backward)
- `sequence_length`: Number of historical timesteps fed as input (typically 12–36 months)
- `batch_size`: Training batch size
- Optimizer: Adam with learning rate ~1e-3 to 1e-4

**Strengths:**
- Captures temporal autocorrelation and long-range dependencies automatically
- No manual lag specification required
- Bi-LSTM reads context from both past and future within the training window
- Handles variable-length sequences
- Can process raw event sequences before aggregation

**Weaknesses:**
- Less interpretable than tree methods without SHAP adaptation (KernelSHAP is approximation)
- Requires more training data than tree methods
- Sensitive to sequence length and architecture choices
- Training is slower and requires GPU for large-scale models
- Vanishing gradient issues with very long sequences (though LSTMs mitigate this vs. vanilla RNN)

---

### Graph Neural Networks (GNNs)

Appear in specialized work on conflict diffusion, spatial propagation, and actor-network modeling. Well-suited to modeling how conflict spreads between spatially adjacent grid cells or ethnically and linguistically linked countries — the contagion problem that Hegre et al. (2021) identified as the main gap in ViEWS after it failed to anticipate the Arab Spring's regional spread.

**Domains used:** Civil conflict diffusion, actor network analysis, subnational spatial prediction

**Representative work:**
- "Glean" system (described in Deng & Ning 2021 survey) — uses temporal event knowledge graphs built from (Actor, Event, Actor) triples to predict concurrent conflict events across multiple event types; combines graph aggregation modules with context-aware embedding fusion
- ViEWS grid-level models — spatial lag features (weighted neighbor event counts) are a simplified non-GNN approach to the same problem; full GNN architectures appear in competition entries

**Key concepts:**
- **Nodes:** Countries, grid cells, or actors (armed groups, governments)
- **Edges:** Geographic adjacency, shared ethnicity/language, historical interaction patterns, or ACLED/GDELT dyadic event links
- **Message passing:** Each node aggregates information from its neighbors, updated iteratively
- **Temporal GNN:** Graph structure evolves over time as new conflict events create/destroy actor relationships

**Strengths:**
- Explicitly models spatial and network dependencies that tabular models treat as independent observations
- Can represent actor heterogeneity (different node types for states, rebels, militias)
- Naturally handles the non-independence of neighboring conflict events

**Weaknesses:**
- Graph construction requires strong domain knowledge and assumptions
- Computationally intensive for large graphs (global country networks)
- Limited labeled training data for conflict domains
- Less mature tooling than tabular or sequence models

---

### Transformers and Attention Mechanisms

The newest and most experimental architecture in this literature, appearing in two distinct forms.

**Text transformers (BERT / XLM-RoBERTa)**

Used for multilingual sentiment extraction from news and social media. XLM-RoBERTa is the most common choice because it covers 100+ languages and performs well zero-shot on political text. It replaces older VADER or TextBlob sentiment tools in recent high-quality systems.

**Representative studies:**
- Chitengu et al. (2025) — XLM-RoBERTa for multilingual tweet sentiment extraction (South Africa)
- CEHA dataset paper (2024, arXiv:2412.13511) — benchmarks BERT, RoBERTa, T5-base, and LLMs (GPT-4o, Mixtral, Mistral-large) for conflict event detection in the Horn of Africa; BERT and RoBERTa outperform larger LLMs in the low-resource setting with fine-tuning
- MLP (Machine Learning for Peace) project — uses transformer-based NLP pipeline for political event tracking across 40+ countries in local languages

**Time-series transformers (TimesFM, PatchTST)**

Google's TimesFM and similar transformer-based time-series foundation models are mentioned in the 2024 Cambridge Core review as nascent approaches being tested on conflict data in zero-shot mode. Their advantage is that they are pretrained on massive time-series corpora and may generalize to political data without task-specific fine-tuning. Performance relative to XGBoost ensembles on conflict data is not yet established in peer-reviewed work.

**Strengths (text transformers):**
- State-of-the-art multilingual text understanding
- Contextual representations capture nuanced political language
- Pre-trained on massive corpora — fine-tuning requires relatively little labeled data

**Weaknesses:**
- Require GPU infrastructure
- Black-box without attention visualization
- Large models (BERT-base: 110M parameters) may be overkill for aggregate monthly country-level sentiment
- Fine-tuning on small conflict-domain datasets risks overfitting

---

### Large Language Models (LLMs) with Retrieval-Augmented Generation (RAG)

Represent the frontier as of 2025. LLMs function as reasoning and narrative generation modules rather than primary classifiers.

**Representative studies:**
- arXiv:2505.09852 (2025) — tests GPT-4, LLaMA-2 on conflict escalation and fatality prediction in the Horn of Africa and Middle East; parametric-only LLMs are unreliable; RAG-augmented LLMs (with ACLED + GDELT context injected at inference) improve substantially but still underperform specialized ML models on quantitative accuracy
- USHMM Early Warning Project — uses LLM narrative generation to explain statistical model predictions to genocide prevention practitioners
- ConflictForecast.org — Mueller/Rauh system generates natural-language risk summaries from topic model outputs

**Current consensus on LLM role:**
LLMs are not suitable as *primary predictive classifiers* for instability onset due to quantitative inaccuracy, knowledge staleness, and hallucination risks. They are valuable as:
- **Narrative generation layer:** Given SHAP feature attributions + recent GDELT/ACLED context, an LLM generates a human-readable explanation of why the model flagged a country — improving analyst uptake
- **Zero-shot baseline:** Useful for establishing a qualitative performance floor in data-sparse contexts
- **Event summarization:** Condensing large volumes of news or social media into structured country-situation summaries that feed downstream classifiers

---

## Ensemble and Fusion Methods

Every high-performance system in the literature uses ensemble fusion. The architecture of the fusion layer distinguishes the major operational systems.

---

### Probabilistic Soft Logic (PSL)

EMBERS's fusion mechanism. Each thematic sub-model (Twitter, news, food prices, TOR traffic, etc.) produces its own probability of unrest. PSL combines these using a set of logic-like rules with learned weights, effectively treating sub-model outputs as "soft" evidence. Rules express statements like "if both Twitter and food prices indicate high risk, the combined probability is much higher than either alone" — without hard boolean logic. PSL is a continuous relaxation of Markov Logic Networks.

**Used in:** EMBERS (Ramakrishnan et al. 2014, Muthiah et al. 2016)

---

### Ensemble Bayesian Model Averaging (EBMA)

The Ward/Beger approach for ILC and conflict forecasting. Each thematic model's predictions are combined using weights proportional to their historical out-of-sample predictive accuracy. Models that have been more accurate in the past receive more weight; models whose predictions are highly correlated with other models are penalized (since correlated models provide redundant information). Montgomery, Hollenbach & Ward (2012) demonstrated EBMA consistently outperforms any individual constituent model.

**Used in:** Ward & Beger ILC forecasting system; early ViEWS prototype (Colaresi, Hegre & Nordkvelle 2016)

---

### Stacking (Meta-Learning)

Trains a second-level model — often a simple logistic regression or shallow gradient boosting — on the out-of-fold predictions of first-level models. The meta-learner learns which first-level models are reliable in which contexts (e.g., tree models may be more reliable for structural country risk; LSTM may be more reliable for acute escalation). This is the approach used in several ViEWS prediction competition winning entries (Hegre, Vesco & Colaresi 2022).

**Used in:** ViEWS escalation prediction competition entries; some EMBERS-adjacent systems

---

### Generalized Additive Model (GAM) Fusion

CoupCast's fusion mechanism for combining the Random Forest and regression ensemble outputs. Smooth nonlinear functions (splines) of each sub-model's output are combined additively. The advantage over stacking is that the contribution of each sub-model to the final prediction is interpretable as a smooth monotonic curve. The GAM fusion layer can be inspected to understand, for example, how the RF output maps to final coup probability.

**Used in:** CoupCast (Powell/UCF)

---

### Simple Averaging / Weighted Averaging

Used as the baseline ensemble fusion method. Calibrated probabilities from multiple models are averaged (equal weights) or weighted by historical out-of-sample AUPRC. Performs surprisingly well — often within 3–5 AUPRC points of EBMA or stacking — and is the recommended starting point before investing in more complex fusion. The IMF WP (2021) uses a simpler variant of this across candidate model specifications.

---

## Summary Table

| Model Type | Generation | Instability Domains | Primary Strength | Primary Weakness | Representative Studies |
|---|---|---|---|---|---|
| Logistic regression (standard) | 1 | All | Interpretable, fast, accepted by policy audiences | Poor on imbalanced data; linear only | Goldstone 2010, Harff 2003 |
| Logistic regression (rare-event / Firth) | 1 | All | Corrects rare-event probability bias | Still linear, still underperforms ML | King & Zeng 2001 |
| LASSO logistic regression | 1 | Unrest, civil war | Automatic feature selection; sparse model | Still linear | EMBERS sub-models |
| Panel logistic regression (FE) | 1 | Civil war, unrest | Identifies within-country temporal signal | Assumes linearity; absorbs structural variation | Mueller & Rauh 2018 |
| Split-population duration model | 1 | Coups, ILC | Correctly handles structural zeros; interpretable | Complex; specialized; less flexible than ML | Ward-Beger 2014/2017; CoupCast |
| Two-stage probit | 1 | Genocide | Models conditional onset structure explicitly | Parametric; in-sample only in early versions | Goldsmith et al. 2013 |
| Random Forest | 2 | All | Nonlinear; handles imbalance; variable importance | Black-box; slower than GBT | Muchlinski 2016; Mueller-Rauh 2019; ViEWS |
| Gradient Boosted Trees (XGBoost/LightGBM) | 2 | All | Best tabular performance; SHAP-native; handles missing data | Requires tuning; less robust than RF to noisy data | IMF WP 2021; Chitengu 2025; ViEWS |
| LSTM | 3 | Unrest, conflict escalation | Captures temporal dependencies without manual lag spec | Needs sequence data; less interpretable | ViEWS competition entries |
| Bi-LSTM | 3 | Unrest, conflict escalation | Bidirectional context; outperforms LSTM in benchmarks | Requires GPU; more parameters | Chitengu 2025; Deng & Ning 2021 |
| Hybrid Bi-LSTM + XGBoost | 3 | Unrest | Best of tabular + sequential architectures | Complex pipeline; harder to maintain | Chitengu 2025 |
| Graph Neural Network (GNN) | 3 | Conflict diffusion, actor networks | Models spatial/network dependencies explicitly | Graph construction requires domain knowledge; computationally intensive | Glean (Deng-Ning survey) |
| BERT / XLM-RoBERTa (text) | 3 | NLP signal extraction | Multilingual; state-of-the-art text understanding | GPU required; overkill for aggregate sentiment | CEHA 2024; MLP project |
| Time-series transformer (TimesFM) | 3 | Conflict escalation | Zero-shot; no task-specific training needed | Performance vs. XGBoost unproven on political data | Cambridge Core review 2024 |
| LLM + RAG | 3 | Narrative explanation | Human-readable explanations; flexible reasoning | Not reliable for quantitative forecasts; hallucination risk | arXiv:2505.09852 |
| PSL ensemble | Cross-gen | Unrest (EMBERS) | Logic-driven fusion of heterogeneous sub-models | System-specific; hard to generalize | EMBERS 2014/2016 |
| EBMA ensemble | Cross-gen | ILC, conflict | Accuracy-weighted; penalizes correlated models | Requires held-out calibration data | Ward-Beger; early ViEWS |
| Stacking | Cross-gen | Civil war, conflict escalation | Learns context-dependent model weights | Can overfit if meta-learner is complex | ViEWS competition entries |
| GAM fusion | Cross-gen | Coups | Interpretable nonlinear combination of sub-models | Less flexible than stacking | CoupCast |
| Simple/weighted averaging | Cross-gen | All | Simple; surprisingly competitive; easy to explain | Does not learn optimal weights per context | IMF WP; multiple systems |
| Isolation Forest (anomaly detection) | Cross-gen | Unrest outbreak detection | Detects sudden regime shifts; no class threshold needed | Less accurate on precise timing | Macis et al. 2024 |

---

## Key Methodological Takeaways

**On model selection:** For structured country-year or country-month tabular data, gradient boosted trees (XGBoost/LightGBM) are now the empirically best single base learner. For sequential event data (monthly ACLED counts, GDELT Goldstein time series), Bi-LSTM or LSTM is the best single architecture. Combining both in a stacking or hybrid ensemble is the current state of the art.

**On class imbalance:** Every instability outcome is a rare event. Standard logistic regression is the worst model for this setting — it predicts peace everywhere and achieves deceptively high accuracy. The correct approaches are: (1) cost-sensitive learning via `scale_pos_weight` or `class_weight`; (2) SMOTE or ADASYN oversampling on the training set; (3) threshold calibration on a held-out validation set; (4) evaluating with AUPRC, not AUROC, as the primary metric.

**On ensembles:** No single model consistently wins. EMBERS, ViEWS, CoupCast, and the IMF study all demonstrate that ensembles — whether PSL, EBMA, stacking, or weighted averaging — outperform any individual model. The minimum viable ensemble for a new system combines a gradient boosted tree model on structural features with an LSTM on event-sequence features, fused via weighted averaging of calibrated probabilities.

**On explainability:** SHAP TreeExplainer is the universal explainability tool across the tree-based methods and integrates directly with XGBoost, LightGBM, and Random Forest. For LSTM and neural models, KernelSHAP provides an approximation. LLMs (GPT-4o via API) can generate narrative explanations from SHAP feature rankings and recent event context, substantially improving analyst uptake of model predictions without replacing the underlying quantitative model.

**On temporal validation:** All models must use strict temporal train/test splits — no shuffling, no k-fold across time. The consensus standard is: train on years T₀ to T-2, validate on T-1, test on T (or rolling origin retraining). Models evaluated only on in-sample or shuffled splits are unreliable for deployment.

---

*Compiled April 2026 from: Ramakrishnan et al. (arXiv:1402.7035); Muthiah et al. (arXiv:1604.00033); Barrett et al. (IMF WP 2021/263); Deng & Ning (arXiv:2112.06345); Macis et al. (ScienceDirect 2024); Chitengu et al. (SAGE 2025); Ward, Greenhill & Bakke (JPR 2010); Muchlinski et al. (Political Analysis 2016); Colaresi & Mahmood (JPR 2017); Hegre et al. (JPR 2019); Hegre, Nygård & Landsverk (ISQ 2021); Mueller & Rauh (APSR 2018); Mueller & Rauh (SSRN 2019); Chadefaux (JPR 2014); Beger, Dorff & Ward (2014/2016/2017); Fox, Verhagen & Cunningham (PLOS ONE 2021); Goldstone et al. (AJPS 2010); King & Zeng (World Politics 2001); Harff (APSR 2003); Goldsmith et al. (JPR 2013); Montgomery, Hollenbach & Ward (Political Analysis 2012); Cambridge Core review (Data & Policy 2024); arXiv:2505.09852 (2025); CEHA dataset (arXiv:2412.13511 2024).*
