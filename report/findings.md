# Findings — Telecom Customer Churn & Recommendation Platform

> **Data disclosure:** every number in this report comes from a **synthetic**
> dataset (Kaggle: `isandeep06/customer-churn-prediction-dataset-1m`, 1,000,000
> generated records). Nothing here describes a real telecom market or real
> customers — it describes patterns the data generator produced. Treat this
> as a demonstration of the analysis method, not a market research report.

**Snapshot as of:** the churn model below was retrained on the **full
1,000,000-row dataset** (all 13 batches, reconstructed locally while the
Postgres warehouse was temporarily unavailable — see
`notebooks/model_dev_offline.py`). `customer_360` in the live warehouse
will catch up to this row count as the remaining batches are ingested
through Airflow in the normal course of operation. See the main
[README.md](../README.md) for architecture and verified build state.

## 1. Class balance (churn label)

90.08% retained / 9.92% churned (900,773 / 99,227 of 1,000,000 customers)
— stable across every subset checked so far (each individual batch, every
partial `customer_360` snapshot, and the full dataset all land within
90.0–90.1% / 9.9–10.0%). Meaningfully imbalanced, which is why this
report leads with precision/recall/F1 per class rather than accuracy.

## 2. Churn model performance

A dedicated model-improvement investigation was run before settling on a
final configuration — see `notebooks/model_dev_offline.py` for the full
script. Summary of what was tried, on the same 80/20 stratified
train/test split throughout for a fair comparison:

- **Feature audit**: all 38 available `customer_360` columns were already
  in use; the only untested field (raw `signup_date`) was tried as three
  extra features (month/day-of-week/year) and **rejected** — it made AUC
  slightly worse (0.6767 → 0.6760), not better. No feature redundancy
  found beyond `tenure`/`tenure_years` (perfectly correlated, harmless for
  tree models) and no target leakage.
- **Threshold tuning**: the default 0.5 threshold is not optimal for a
  ~10%-imbalanced target. Precision at a few recall targets: recall 0.60
  costs precision 0.159; recall 0.70 costs precision 0.145. A business
  threshold was chosen to **guarantee recall ≥ 0.60** (catching more
  churners is worth some false positives in a retention context) rather
  than maximizing F1 alone.
- **Model comparison** (RandomForest baseline vs. XGBoost vs. LightGBM vs.
  Logistic Regression, same split): LightGBM won by a small but real
  margin, confirmed stable by 5-fold cross-validation (AUC 0.683 ± 0.002
  across folds — not a lucky split).
- **Imbalance handling**: SMOTE oversampling was tested against
  `class_weight="balanced"` and gave a statistically indistinguishable AUC
  (0.684 vs 0.683) while adding real complexity (resampling 800K→1.44M
  rows, needing careful threshold recalibration since SMOTE shifts what
  the raw probability output means). Not adopted.

**Final configuration:** LightGBM, `class_weight="balanced"`, decision
threshold 0.512 (chosen to guarantee recall ≥ 0.60), trained on the full
1,000,000-row dataset:

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Retained (0) | 0.94 | 0.66 | 0.78 |
| Churned (1) | 0.16 | 0.60 | 0.26 |

Accuracy 0.66 (uninformative given the imbalance — included only for
completeness), ROC AUC 0.683.

**Honest verdict:** nothing tried meaningfully beats the original
RandomForest baseline (AUC 0.677, F1 0.252 at the default 0.5 threshold).
LightGBM's edge (AUC 0.683, confirmed by cross-validation) is small,
real, and adopted — mainly because it also trains ~6x faster. The bigger,
practically useful change was threshold tuning: recall went from 0.45–0.50
to a guaranteed 0.60 at the cost of precision dropping to 0.16, a
deliberate trade-off for a retention use case where missing a churner
costs more than one extra false-positive outreach. The ~0.68 AUC ceiling
looks like a genuine property of this synthetic dataset's generated
signal, not a modeling gap — see the README's Limitations section.

## 3. Top churn risk factors

LightGBM feature importances (individual features, not grouped), top
contributors:

1. **Contract type** — by far the largest single driver (`contract=two_year`
   and `contract=one_year` together account for ~31% of total importance;
   see segment breakdown below: month-to-month churns at 4.8x the rate of
   two-year).
2. **Customer satisfaction score** (~11% of importance on its own)
3. **Service call volume** (`num_service_calls`) and **complaint volume**
   (`num_complaints`, `complaints_per_tenure_month`)
4. **Contract type again**, via the `is_month_to_month` derived flag
5. **Monthly charges** and **cost per active service**
6. **Support usage** (`has_tech_support`) and **late payments**
7. **Total charges** and **income-to-spend ratio** (`annual_spend_to_income_ratio`)
8. **Credit score**

These are all sensible, business-plausible churn drivers — no sign the
model is picking up on noise.

## 3b. Statistical validation — is the risk-factor list actually real?

A feature-importance ranking can look convincing without being statistically
sound. `notebooks/eda_and_statistical_analysis.ipynb` tests this directly,
independent of the LightGBM model, using two completely different methods:

**Significance tests** (chi-square for categorical features, Mann-Whitney U
for numeric): `contract` and `tenure_bucket` are genuinely associated with
churn (Cramer's V 0.14 and 0.01 respectively — contract's effect is real
and moderate, tenure's is real but weak). `gender`, `education`,
`marital_status`, and `payment_method` are **not statistically
significant** (all p > 0.05) — there is no supportable "churn persona"
based on demographics in this data, however tempting that narrative might
be from a bar chart alone.

**Survival analysis** (Cox Proportional Hazards, modeling time-to-churn
rather than a binary outcome): confirms the same ranking through an
entirely different method. `is_month_to_month` carries a hazard ratio of
**2.86** — a month-to-month customer's instantaneous churn risk is
essentially triple a longer-contract customer's, at any given tenure.
`num_complaints` (HR 1.19) and low `customer_satisfaction` (HR 0.90 per
point, i.e. protective) are the next-strongest drivers. Model concordance:
0.62 (0.5 = random ranking, 1.0 = perfect).

**A statistical-significance caution, reported rather than hidden:**
`monthlycharges` reaches p < 0.05 in both the significance tests and the
Cox model, but its hazard ratio is ≈0.998 per dollar — a customer would
need to pay $100/month more just to see their hazard drop ~18%. At
300K+ rows, statistical significance is easy to reach for practically
negligible effects; both numbers are reported together specifically so
this doesn't get miscommunicated as "monthly charges matter."

**Unsupervised segmentation** (K-means, no churn label involved in forming
the clusters): finds 4 usable customer segments differing in income,
tenure, usage, and bundle size, with churn rates ranging 9–11% across
clusters. Reported honestly: silhouette scores are modest (~0.14–0.16,
roughly flat across k=3–8) — the natural structure is a soft continuum,
not sharp, well-separated groups. k=4 is kept for interpretability, not
because the clustering is unambiguous.

## 4. Churn rate by segment

**By contract type:**

| Contract | Churn rate | n |
|---|---|---|
| Month-to-month | 27.0% | 6,146 |
| One-year | 12.7% | 169,255 |
| Two-year | 5.6% | 132,295 |

**By tenure bucket:**

| Tenure | Churn rate | n |
|---|---|---|
| New (0–6mo) | 10.3% | 67,667 |
| Established (6–24mo) | 10.1% | 126,303 |
| Loyal (24mo+) | 9.5% | 113,726 |

Tenure's effect is much weaker than contract type in this dataset — a
mild downward trend, not the sharp drop-off sometimes assumed.

**By service bundle size** (`total_active_services`, 0–8): churn rate
decreases steadily and almost monotonically from 12.4% (0 services) to
6.7% (all 8 services) — customers with more active subscriptions churn
less, consistent with a "stickiness" story, though this is also
correlated with contract type and tenure rather than necessarily causal.

## 5. Recommendation model behavior

Content-based k-NN over standardized demographic/account/usage profiles
(age, income, tenure, charges, satisfaction, usage, gender, education,
marital status, contract, payment method — service flags themselves are
deliberately excluded from the similarity profile). For each customer, the
20 nearest profile-neighbors' service subscriptions are used to rank the
customer's own *un-subscribed* services by neighbor popularity
(inverse-distance-weighted).

Example (customer `CUST0000099261`, an established 8-month tenure
customer, one-year contract): already has phone, internet, online backup,
and tech support; the model recommended, in order: **streaming TV** (score
0.457), **device protection** (0.360), **online security** (0.192) — a
plausible upsell path toward a more "fully bundled" profile matching what
similar customers in the neighborhood have.

## Limitations

See the [README's Limitations section](../README.md#limitations) — synthetic
data, DuckDB-over-Spark tradeoff, simulated batch arrivals, content-based-only
recommendations, batch-count-based retraining, and modest churn model
performance all apply to every finding in this report.
