# Plan: 4-Month Azure ML Social Instability Classifiers

## Requirements Summary

**Project Goal:** Build 2-3 fine-tuned classifiers (Random Forest, XGBoost, Ensemble) to predict social instability events using ACLED + existing annotated protest datasets.

**Key Decisions:**
- **Data:** ACLED + existing annotated datasets (NOT PEA repo integration)
- **Models:** XGBoost, Random Forest, Ensemble
- **Labeling:** Custom workflow (Label Studio)
- **Tracking:** MLflow + Azure ML (dual integration)
- **Deployment:** Azure Container Instances (REST endpoints)
- **Timeline:** M1 data+labeling, M2-3 tuning, M4 deployment

---

## Plan: Phase I (Month 1) — Data Procurement & Annotation Setup

### Discovery & Procurement (Weeks 1-2)

1. **Identify datasets:**
   - Fetch ACLED events dataset (free API, register at acleddata.com; ~25K Africa events annually)
   - Source existing annotated protest datasets (e.g., Protest Radicals, MMAD, GFD, POLI; record licensing)
   - Document feature alignment (date, location, event type, actors, claims, outcome)

2. **Define predictive schema:**
   - Target variable: instability indicator (binary: protest/riot/violent vs. non-event OR multi-class: riot/protest/demonstration/looting/violence-escalation)
   - Features: narrative text, date, location, reported crowd size, state response type (if available), actors
   - Document class balance expectations and ratios

3. **EDA + data cleaning:**
   - Notebook: `exploratory_analysis.ipynb` (load ACLED + annotated datasets, inspect missingness, class distribution, text length)
   - Identify clean subset for v1 labeling sprint (target: 3,000–5,000 events) 
   - Record data lineage: source URL, file hash, download date

**Outputs:** `data/raw/acled_raw.csv`, `data/raw/annotated_datasets_compiled.csv`, `exploratory_analysis.ipynb`, `schema_definition.md`

### Annotation Workflow Setup (Weeks 2-3)

4. **Label Studio environment:**
   - Docker Compose: spin up Label Studio locally + persist DB
   - Labeling interface XML: define tasks (event classification: protest/riot/violence/non-event, confidence level, notes field)
   - Create 2-3 projects: (1) Cold-start; (2) Active learning tier 1 (uncertain model predictions); (3) Tier 2 (high confidence check)

5. **Annotation team & inter-rater reliability:**
   - Define team (1-3 annotators? note assumption in plan)
   - Codebook: detailed definitions + 10 edge cases per event type
   - Cohen's kappa calibration: 10-event overlap, target κ ≥ 0.75
   - Review protocol: spot-check 10% of labels weekly

6. **Active learning framework (optional, for M3+ speed-up):**
   - Train initial baseline classifier (150–200 hand-labeled seed events)
   - Log predictions to MLflow; flag low-confidence predictions for re-labeling first
   - Implement uncertainty sampling callback

**Outputs:** 
- Label Studio project (via docker-compose.yml)
- `labeling_config.xml`, `annotator_codebook.md`
- Initial 200–500 labeled events
- `inter_rater_reliability_report.json` (κ scores per annotator pair)

### Infrastructure Setup (Weeks 3-4)

7. **Azure ML Workspace provisioning:**
   - Bicep template: Workspace, Storage, Compute (CPU cluster for tuning)
   - Azure CLI script: deploy infrastructure, capture workspace details into `.env`
   - Verify: authenticate via `az login`, confirm Workspace connectivity

8. **Repository structure & Git workflow:**
   - Migrate `rf_finetune_azureml.ipynb` → `notebooks/` directory
   - Create directories: `data/{raw,processed,splits}`, `models/`, `configs/`, `src/{training,inference,monitoring}`, `experiments/`
   - Initialize `requirements-core.txt`, `requirements-dev.txt` with all dependencies
   - Add `.env.template`, `.gitignore` (exclude `.env`, `models/*.pkl`, `data/raw/`)
   - Create README: setup instructions + workspace registration

9. **Connect MLflow backend to Azure ML:**
   - Install `mlflow` + `azureml-mlflow` packages
   - Configure MLflow tracking URI to Azure ML Workspace: `azureml://<workspace-name>`
   - Test: verify MLflow experiment runs appear in Azure portal

**Outputs:**
- Azure ML Workspace provisioned (via Bicep + CLI)
- Git repository organized (all dirs, requirements, README)
- `.env` populated with workspace ID, resource group
- Test run: one dummy MLflow experiment logged successfully

---

## Plan: Phase II (Months 2–3) — Model Selection & Fine-Tuning

### Feature Engineering & Data Splits (Weeks 5–6)

10. **Expand labeled dataset:**
    - Continue annotation (target: cumulative 3,000–5,000 events by end of Week 6)
    - Monitor inter-rater reliability, recompute κ weekly
    - Handle disagreements: third-pass annotation by senior annotator if κ dips below 0.70

11. **Feature engineering notebook:**
    - TF-IDF vectorization of narrative text (max_features=500)
    - Temporal features: day-of-week, month, season
    - Geographic features: region encoding, urbanicity (if available)
    - Aggregate features: event-type prevalence in prior 30 days
    - Store feature matrix as `data/processed/feature_matrix.pkl`

12. **Train/test/validation splits:**
    - Stratified by event type + country (to avoid country leakage)
    - Train: 70%, Val: 15%, Test: 15%
    - Export: `data/splits/{train,val,test}_indices.json`

**Outputs:** 3,000–5,000 labeled events, `feature_engineering.ipynb`, feature matrix + splits

### Baseline & Model Selection (Weeks 7–8)

13. **Baseline classifier (simple decision rule):**
    - Rule-based precursor (e.g., IF keywords="violence" OR actors="military" THEN violent_instability=1)
    - Log metrics to MLflow: accuracy, precision, recall, F1 per event type
    - Record baseline as benchmark for model comparison

14. **XGBoost baseline fine-tune:**
    - Notebook: `xgboost_baseline.ipynb`
    - Hyperparameters: initial grid (n_estimators=[50,100,200], max_depth=[3,6,10], learning_rate=[0.01,0.05,0.1])
    - Local RandomizedSearchCV (10 iterations) with stratified 5-fold CV
    - Log each run to MLflow: `mlflow.log_params()`, `mlflow.log_metrics()` (accuracy, F1, precision, recall)
    - Register best run as artifact: `mlflow.sklearn.log_model()`

15. **Random Forest fine-tune:**
    - Comparable hyperparameter grid (n_estimators, max_depth, min_samples_split)
    - Same logging + artifact registration
    - Compare performance to XGBoost in MLflow UI

16. **Ensemble (stacking or voting):**
    - Meta-learner: logistic regression on XGBoost + RF predictions
    - Evaluate ensemble vs. individual models
    - Log as separate run + model

**Outputs:** 
- MLflow experiments with 20–30 runs logged (baselines + tuning)
- 3 candidate models (XGBoost, RF, Ensemble) registered in MLflow registry
- Comparison table: `models/model_comparison.csv` (metrics by model)

### Distributed Hyperparameter Tuning (Weeks 9–10) — *Optional, depends on data size*

17. **Scale tuning to Azure ML HyperDrive (if needed):**
    - If test set is > 10K events or model training time > 5 min per iteration:
      - Prepare training script: `src/training/train_xgboost.py`
      - Define hyperparameter search space (Azure ML Uniform, Choice distributions)
      - Submit HyperDrive job via Azure ML SDK (20–50 iterations)
      - Early stopping: Bandit policy (slack_factor=0.15, eval_interval=2)
    - Log HyperDrive run to MLflow + Azure ML (dual tracking)

**Outputs:** Accelerated tuning results, best hyperparameters registered

### Error Analysis & Iteration (Weeks 10–11)

18. **Confusion matrix + error analysis:**
    - Per model: compute confusion matrix by event type
    - Identify systematically misclassified events (e.g., "peaceful demonstrations mis-labeled as violent")
    - Create `error_analysis.ipynb`: visualizations + category breakdown
    - Log to MLflow as HTML/PNG artifacts

19. **Data augmentation / re-labeling:**
    - If specific event types have poor recall (e.g., false negatives), flag for re-annotation
    - Collect hard negatives (false pos that almost passed); re-label to refine boundary
    - Active learning: retrain baseline, collect low-confidence predictions, re-label top-N

20. **Final model lock:**
    - Choose champion model (best F1 or best precision depending on use case)
    - Document decision: metrics, error profile, production constraints
    - Stage model in MLflow Registry with "Staging" status + production readiness notes

**Outputs:**
- Error analysis notebook
- 200–500 re-labeled hard negatives (added to training dataset v2)
- Champion model registered as "Staging"

---

## Plan: Phase III (Month 4) — Production Deployment

### Model Packaging & Testing (Weeks 13–14)

21. **Package model as Python artifact:**
    - Create `src/inference/model_loader.py`: load trained model from MLflow, apply same feature pipeline
    - Create `src/inference/predictor.py`: input validation, inference, confidence thresholding
    - Unit tests: `tests/test_predictor.py` (mock inputs, verify output schema)
    - Integration test: end-to-end test with 10 real events from test set

22. **Create inference Docker image:**
    - Dockerfile: base Python 3.9, install `scikit-learn`, `xgboost`, `mlflow`, `flask` (or FastAPI)
    - API endpoints:
      - `POST /predict` — input: JSON (narrative, date, location), output: JSON (event_type, confidence, instability_score)
      - `GET /health` — liveness check
    - Test locally: `docker build -t instability-predictor:v1 . && docker run -p 5000:5000`

**Outputs:** 
- `src/inference/model_loader.py`, `predictor.py`
- Unit + integration tests
- `Dockerfile` + tested locally

### Azure Container Instances Deployment (Weeks 14–15)

23. **Push image to Azure Container Registry (ACR):**
    - Create ACR resource in Azure (once per project)
    - Push image: `az acr build -r <registry-name> -t instability-predictor:v1 .`
    - Verify: image appears in ACR via `az acr repository list`

24. **Deploy to Azure Container Instances:**
    - Create container instance resource / deploy via Azure CLI:
      ```
      az container create --resource-group <rg> --name instability-predictor --image <acr-url>/instability-predictor:v1 \
        --ports 5000 --cpu 1 --memory 1 \
        --environment-variables MLFLOW_TRACKING_URI=<workspace-uri> MODEL_VERSION=v1
      ```
    - Capture public IP / FQDN
    - Test via curl: `curl http://<FQDN>:5000/health`

25. **Wire MLflow model registry:**
    - Pass production model URI to container (env var or config)
    - Model loader auto-fetches from MLflow @ container startup
    - Log inference requests back to MLflow (optional, for monitoring)

**Outputs:**
- Image in ACR
- Live endpoint at `http://<FQDN>:5000`
- Health check succeeds

### Monitoring & Observability (Week 15–16)

26. **Application Insights + logging:**
    - Configure container to send logs to Azure Application Insights
    - Log structured events: timestamp, input features, prediction, inference latency
    - Set up dashboard: request rate, response time P50/P95/P99, error rate

27. **Data drift monitoring:**
    - Collect inference request features monthly
    - Compare to training set feature distributions (Kolmogorov-Smirnov test)
    - Alert if KS statistic > threshold (0.15)
    - Document retraining trigger in README

28. **Versioning & promotion:**
    - Document model version mapping (v1 → Champion from M3)
    - Create rollback playbook: how to revert to previous model if performance degrades
    - Setup: A/B test framework (optional, for future rollouts)

**Outputs:**
- Application Insights dashboard
- Monitoring queries + alert rules
- `docs/monitoring_guide.md`, `docs/rollback_playbook.md`

### Documentation & Handoff (Week 16)

29. **Production run book:**
    - How to update model (rerun training, bump version, push new image, update container)
    - How to monitor / alert on failures
    - Troubleshooting guide (common errors + fixes)

30. **MLflow + Azure ML audit trail:**
    - Verify all experiments + model versions are logged
    - Export experiment summary report: `mlflow experiments` + artifact tree
    - Document: which run ID is production, why it was chosen

31. **Final validation:**
    - Shadow-run: compare predictions on randomly sampled 100 events (manual review by stakeholder)
    - Confirm output schema matches downstream consumers
    - Record sign-off from stakeholder (if applicable)

**Outputs:**
- Production playbook + troubleshooting guide
- MLflow audit report
- Go-live checklist

---

## Critical Architecture & Patterns

### Model Training Pipeline
- **Baseline:** `src/training/train_baseline.py` (rule-based + simple models)
- **Tuning:** `src/training/train_xgboost.py`, `src/training/train_rf.py`
- **Ensemble:** `src/training/train_ensemble.py` (meta-learner pattern)
- **Feature pipeline:** `src/features/engineer.py` (TF-IDF, temporal, geographic)
- **MLflow logging:** Wrapper `src/training/mlflow_utils.py` (standardize log_params, log_metrics, register_model)

### Inference Pipeline
- **Model loader:** `src/inference/model_loader.py` (fetch from MLflow registry, cache locally)
- **Preprocessing:** `src/inference/preprocessor.py` (same feature pipeline as training)
- **API:** `src/inference/app.py` (Flask/FastAPI, health + predict routes)
- **Docker:** `Dockerfile` (multi-stage, ~200MB final image)

### Monitoring & Observability
- **Logging:** Structured logs (JSON) → Application Insights
- **Drift detection:** Monthly batch job (KS test on inference features)
- **Model registry:** Tracked in MLflow + versioned in Git (`models/versions.json`)

### Data Management
- **Versioning:** All datasets tracked by hash in `data/MANIFEST.md`
- **Deduplication:** Handle overlapping events from ACLED + annotated datasets (record source)
- **Splits:** Stratified by geography + event type, exported as indices (reproducible)

---

## Relevant Files (Planned)

**Notebooks:**
- `notebooks/exploratory_analysis.ipynb` — ACLED EDA
- `notebooks/xgboost_baseline.ipynb` — Model tuning + MLflow logging
- `notebooks/error_analysis.ipynb` — Confusion matrix + hard negatives
- `notebooks/features_engineering.ipynb` — Feature matrix generation

**Source Code:**
- `src/training/train_xgboost.py` — XGBoost training script
- `src/training/train_rf.py` — Random Forest training script
- `src/training/train_ensemble.py` — Stacking ensemble
- `src/training/mlflow_utils.py` — MLflow helpers (log_params, register, compare)
- `src/features/engineer.py` — Feature pipeline (TF-IDF, temporal, geo)
- `src/inference/model_loader.py` — Fetch trained model from MLflow
- `src/inference/predictor.py` — Inference logic + input validation
- `src/inference/app.py` — Flask/FastAPI endpoints
- `src/monitoring/drift_detector.py` — Monthly KS test + alert

**Configuration & Infrastructure:**
- `configs/model_config.yaml` — Model hyperparameters, feature settings, thresholds
- `infra/bicep/workspace.bicep` — Azure ML Workspace provisioning
- `infra/setup.sh` — Workspace creation + workspace config download
- `Dockerfile` — Multi-stage build for inference container
- `requirements-core.txt` — Core dependencies
- `requirements-dev.txt` — Dev + notebook dependencies

**Labels & Data:**
- `data/raw/acled_raw.csv` — Downloaded ACLED dataset
- `data/raw/annotated_datasets_compiled.csv` — Compiled public datasets
- `data/processed/feature_matrix.pkl` — Engineered features
- `data/splits/{train,val,test}_indices.json` — Reproducible splits
- `data/labels/events_labeled_v1.jsonl` — Label Studio output

**Tests & Validation:**
- `tests/test_predictor.py` — Unit tests for inference
- `tests/test_feature_engineering.py` — Feature pipeline tests
- `notebooks/mlflow_audit.ipynb` — Export all runs + model versions

**Documentation:**
- `README.md` — Setup + execution instructions
- `docs/schema_definition.md` — Target variable + feature definitions
- `docs/annotation_codebook.md` — Event type definitions + edge cases
- `docs/monitoring_guide.md` — Observability setup
- `docs/rollback_playbook.md` — How to revert models

---

## Verification

### Phase I (Data & Setup)
1. **Week 2:** ACLED + annotated datasets imported, schema defined, EDA notebook runnable
2. **Week 3:** Label Studio running locally, team calibrated (Cohen's κ ≥ 0.75)
3. **Week 4:** Azure ML Workspace provisioned, test MLflow run logged successfully

### Phase II (Training & Tuning)
4. **Week 6:** 3,000–5,000 events labeled, inter-rater reliability stable
5. **Week 8:** XGBoost, RF, and Ensemble models trained and compared in MLflow UI
6. **Week 11:** Champion model registered with detailed error analysis; predictions reviewed by stakeholder

### Phase III (Deployment)
7. **Week 14:** Docker image builds and runs locally; endpoints respond with correct JSON schema
8. **Week 15:** Container deployed to ACI, health check passes, shadow prediction test with 100 events
9. **Week 16:** Production playbook + rollback guide documented; sign-off obtained

---

## Decisions & Scope Boundaries

### Included (In Scope)
- ✅ ACLED + existing annotated datasets (public, free)
- ✅ XGBoost + RF + Ensemble models only (traditional ML, not neural networks)
- ✅ Custom Label Studio annotation workflow
- ✅ MLflow + Azure ML dual tracking
- ✅ Azure Container Instances deployment + REST API
- ✅ Binary instability classification (or multi-class; codebook TBD)
- ✅ Monthly drift monitoring + retraining trigger

### Explicitly Out of Scope (For Future Phases)
- ❌ Neural network fine-tuning (LLMs, DistilBERT)
- ❌ PEA repo data integration
- ❌ Kubernetes deployment (using ACI for simplicity)
- ❌ A/B testing framework (baseline rollout only)
- ❌ Batch inference pipelines (single-record inference via API)
- ❌ Production ACLED API integration (manual data pull in M1)

---

## Critical Assumptions & Risks

### Assumptions
- ACLED API remains accessible & data schema stable (low risk, mature dataset)
- 1–3 annotators available for M1 labeling (high engagement required)
- 3,000–5,000 labeled events sufficient for model training (may underestimate if class imbalance extreme)
- XGBoost/RF models will meet performance needs (true if feature engineering is strong)

### Risks & Mitigations
| Risk | Likelihood | Mitigation |
|------|-----------|-----------|
| Annotation bottleneck in M1 | High | Pre-identify annotators; implement active learning early (W10) to accelerate tier 1 labels |
| Poor model performance (F1 < 0.65) | Medium | A/B test feature engineering variants; consider ensemble of multiple feature sets; collect more hard negatives |
| Azure ML cost overruns | Low | Monitor compute usage weekly; use CPU-only clusters; disable HyperDrive if local tuning sufficient |
| Production endpoint latency > acceptable | Low | Profile locally; optimize feature pipeline; cache model in memory |

---

## Further Considerations

1. **Event type taxonomy:** Decide whether target is binary (instability Y/N) or multi-class (protest, riot, violence, looting, escalation). Recommend multi-class for strategic value but requires split annotation instructions. **Recommendation:** Multi-class; 5 categories. Document in `annotation_codebook.md` before M1 starts.

2. **Active learning tier structure:** If labeling becomes bottleneck, pyramid tiers: (1) Low conf + high relevance first (M2), (2) Medium conf spot-check (M3), (3) High conf sample only. Requires baseline model by W7. **Recommendation:** Plan infrastructure in M1, activate if needed by W9.

3. **Real-time vs. batch inference:** Current plan assumes single-record REST API. If downstream consumer needs batch inferences (e.g., monthly nightly batch), add `POST /batch_predict` endpoint by W14. **Question:** Who will call the endpoint? How often?

---

## Timeline Summary

| Phase | Duration | Key Deliverables |
|-------|----------|------------------|
| **Phase I: Data & Setup** | Month 1 (Weeks 1–4) | Labeled dataset (3k–5k events), Label Studio setup, Azure ML Workspace, MLflow tracking |
| **Phase II: Training & Tuning** | Months 2–3 (Weeks 5–12) | 3 trained models (XGBoost, RF, Ensemble), error analysis, champion model selected |
| **Phase III: Production** | Month 4 (Weeks 13–16) | Live endpoint on ACI, monitoring dashboard, runbooks, go-live checklist |

---

**Status:** Plan ready for review. Clarify any further requirements before handoff to implementation.
