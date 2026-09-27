# Modeling and analysis

The statistical work behind the churn model and recommender: what was tested, what it found, and how each number was produced. See [`report/findings.md`](../report/findings.md) for the business-facing write-up.

## Statistical & analytical depth

Beyond the production pipeline, `notebooks/eda_and_statistical_analysis.ipynb` is a fully executed (not just written) analysis notebook that goes past descriptive segment charts into actual statistical methodology:

- **Significance testing**: chi-square tests (with Cramer's V effect size) for every categorical feature against churn, and Mann-Whitney U tests (with rank-biserial effect size) for every numeric feature. Finding: `contract` and `tenure_bucket` are genuinely associated with churn; `gender`, `education`, `marital_status`, and `payment_method` are **not** statistically significant at all — no demographic "churn persona" is supported by this data.
- **Statistical vs. practical significance**: at 300K+ rows, several features reach p < 0.05 with a practically negligible effect size (e.g. `monthlycharges`, hazard ratio ≈ 0.998 per dollar) — called out explicitly rather than reported as a bare "significant!" p-value.
- **Correlation & multicollinearity**: a correlation heatmap plus Variance Inflation Factors across key numeric features.
- **Survival analysis**: Kaplan-Meier curves (overall and by contract type) and a Cox Proportional Hazards model — modeling *time to churn* directly, which the binary classifier discards. `is_month_to_month` carries a hazard ratio of ~2.77 (holding other factors constant); concordance index 0.617. Kaplan-Meier: 94.7% of customers still active at 12 months tenure, 90.2% at 24, 82.1% at 48.
- **Unsupervised customer segmentation**: K-means clustering on profile features (age, income, tenure, charges, satisfaction, usage, service count), with an elbow/silhouette analysis to choose k and a PCA projection to visualize it. Reported honestly: silhouette scores are modest (~0.14–0.16), meaning the natural cluster structure is soft, not sharply separated — stated plainly rather than overclaimed.
- **SHAP explainability**: TreeExplainer on the production LightGBM model, both as a global summary plot and individual waterfall plots for specific high-risk and low-risk customers. This same SHAP logic is also used live in the **frontend's At-Risk Customers view** (`compute_shap_details` in `src/model/explain_churn.py`) — replacing an earlier global-feature-importance heuristic with real per-customer explanations (bounded to the displayed rows, not the full 1M-row table, for memory reasons).

### Re-running the analysis

The notebooks need libraries the pipeline itself doesn't (statsmodels, lifelines, xgboost, imbalanced-learn, matplotlib, seaborn). Those live in a **separate** requirements file, deliberately: `requirements.txt` is installed into the API image and every CI run, neither of which import any of them.

```bash
pip install -r requirements.txt -r requirements-notebooks.txt
jupyter lab notebooks/
```

Versions there are pinned to the ones the committed outputs were produced with, so a re-run reproduces the numbers in `report/findings.md` rather than approximately reproducing them.

See the notebook itself for full output, and `report/findings.md` for the business-facing summary of these findings.

## Drift monitoring

Every batch that arrives is scored against a fixed baseline batch using the **Population Stability Index**, the standard drift metric in credit-risk and churn modeling:

```
PSI = sum over bins of  (actual% - expected%) * ln(actual% / expected%)

PSI < 0.10          stable        - no action
0.10 <= PSI < 0.25  moderate      - investigate
PSI >= 0.25         significant   - retrain
```

19 features are monitored (14 numeric, 5 categorical). Results are written to `public.feature_drift` by the DAG's `detect_feature_drift` task and surfaced on the frontend's Pipeline Status view.

This turns retraining into a two-trigger decision rather than a fixed cadence:

| Trigger | Condition | Rationale |
| --- | --- | --- |
| Cadence | every 3rd batch | the model never silently goes stale |
| Evidence | any feature PSI ≥ 0.25 | the incoming distribution no longer matches the training distribution |

Three implementation choices worth calling out, because they are where naive PSI implementations go wrong:

- **Bin edges come from the reference distribution only**, never recomputed per batch. Re-binning on the new data makes every batch look identical to itself and hides precisely the shift being measured.
- **NULLs are their own bucket**, not dropped. A feature whose missing rate jumps from 3% to 40% has drifted in the way that matters most operationally; dropping NULLs scores that as perfectly stable.
- **The outer bin edges are open** (`-inf`, `+inf`), so values beyond the reference range — the most obvious kind of drift — land in the end bins instead of being silently discarded as out-of-range.

**The honest result on this dataset: no drift, as expected.** All 13 batches are sequential slices of one pre-shuffled source file, so the maximum PSI observed across every feature and every batch pair is **0.0006** — three orders of magnitude below the "investigate" threshold. That is the correct answer for this data, not a broken detector. Evidence that the detector does fire lives in `tests/test_drift.py` (18 tests), which injects real shifts and asserts they are caught: mean shifts, variance shifts with an unchanged mean, missingness jumps, unseen categorical levels, and out-of-range values.

Run it manually against any ingested batch:

```bash
python -m src.monitoring.drift batch_013.csv
```

## Probability calibration

`class_weight="balanced"` is what makes the churn model's *ranking* work under a ~10%-imbalanced target, but it does so by inflating minority-class scores, so the raw output is not a probability at all — mean predicted score was **0.41** against an actual churn rate of **0.10**, a 4.1x overstatement, and the model's own Brier score (0.196) was *worse* than simply always predicting the base rate (0.090).

`train_churn.py` now fixes this with a three-way split (train / calibration / test) and `CalibratedClassifierCV` (sigmoid / Platt scaling) fit on the held-out calibration split — never the training or test data, so nothing leaks into the reported metrics:

| | Brier score | Mean predicted | Actual rate |
| --- | --- | --- | --- |
| Uncalibrated | 0.196 (worse than the 0.090 baseline) | 0.408 | 0.100 |
| **Calibrated (sigmoid)** | **0.087** (beats the baseline) | **0.100** | 0.100 |

Sigmoid rather than isotonic: both reached the same Brier score in testing, but sigmoid is a strictly monotonic two-parameter fit, so it leaves ROC AUC exactly unchanged (0.6693 either way) and is far less prone to overfitting a ~15K-row calibration split than isotonic's free-form step function.

**Calibration changes what the numbers mean, not what the model does.** Because sigmoid scaling is monotonic, ranking and the operating point are unaffected — precision (0.160) and recall (0.600) at the chosen threshold are the same before and after. What changes is that the decision threshold now sits at **0.105** instead of **0.451**, because it is now a real probability near the ~10% base rate rather than an arbitrary score compressed toward 0.5. One consequence worth knowing before you go looking for it: **a naive 0.5 threshold on the calibrated model now flags zero customers** — not a bug, but exactly the failure mode calibration exists to expose. The full before/after, including the expected-value threshold analysis recomputed on the calibrated scores, is in `report/findings.md` Section 6.

This also broke something non-obvious enough to be worth flagging on its own: `CalibratedClassifierCV` has no `.named_steps`, which the dashboard's SHAP explanations and feature-importance fallback both depend on. See [Deviations](engineering-log.md#deviations-from-the-original-spec-and-why) for how that was caught and fixed.

## Recommender evaluation

Every other model in this project is measured against a baseline on a held-out split; until now the recommender wasn't — its metadata recorded only structural facts (profile count, neighbor count). `src/model/evaluate_recommender.py` closes that gap with a leave-one-out protocol: hide one service a customer genuinely owns, present them to the model as if they didn't own it, and check where the hidden service lands in the ranking of everything else they don't own. Three rankers are scored on identical splits, over customers held **entirely outside** the k-NN reference set (scoring the model on profiles it was fitted on would measure memorization, not generalization):

| Ranker | Hit@1 | Hit@2 | Hit@3 | MRR |
| --- | --- | --- | --- | --- |
| **k-NN (production)** | **0.519** | 0.737 | 0.878 | **0.703** |
| Popularity baseline | 0.483 | 0.732 | 0.891 | 0.686 |
| Random | 0.245 | 0.478 | 0.681 | 0.498 |

**The honest result: k-NN beats popularity, but by a small margin, and popularity is a genuinely hard baseline to clear here** — 8 services with skewed adoption rates (internet 84.8% down to device protection 29.6%) leave little room for a naive ranker to be bad. A table alone can't say whether +2.5% MRR is signal or noise, so the module runs two significance tests rather than reporting the gap and moving on: a paired bootstrap (2,000 resamples, customer-level) puts the MRR gap at **+0.0170, 95% CI [+0.0105, +0.0237]**, and an exact McNemar test on Hit@1 gives **p = 2.3×10⁻⁹**. Both agree the edge is real, not sampling noise — but it is a small edge, reported as one, not rounded up to "the model works great" or down to "personalization doesn't matter here."

```bash
python -m src.model.evaluate_recommender
```

Full detail, including why customers with fewer than 2 candidate services are excluded from evaluation (a guaranteed hit that would inflate every ranker equally), is in `report/findings.md` Section 5a.
