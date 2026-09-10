# Findings — Telecom Customer Churn & Recommendation Platform

> **Data disclosure:** every number in this report comes from a **synthetic**
> dataset (Kaggle: `isandeep06/customer-churn-prediction-dataset-1m`, 1,000,000
> generated records). Nothing here describes a real telecom market or real
> customers — it describes patterns the data generator produced. Treat this
> as a demonstration of the analysis method, not a market research report.

**Snapshot as of:** all **13 batches ingested through the Airflow DAG** —
`customer_360` now holds the complete **1,000,000 rows** in the live
warehouse. The statistical analyses below were re-run against that full
dataset. The production churn model itself is trained on a 150,000-row
stratified sample of it, a deliberate memory trade-off documented in
`src/model/train_churn.py` (`MAX_TRAINING_ROWS`). Data drift across those batches is monitored
with PSI (Section 7). See the main [README.md](../README.md) for
architecture and verified build state.

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

**Current production model** (version `20260910T121802Z`, retrained
automatically by the Airflow DAG): LightGBM, `class_weight="balanced"`,
**sigmoid-calibrated** on a held-out split, decision threshold 0.105,
trained on a 150,000-row stratified sample:

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Retained (0) | 0.94 | 0.65 | 0.77 |
| Churned (1) | 0.16 | 0.60 | 0.25 |

ROC AUC 0.669. Accuracy 0.64 is reported only for completeness — it is
uninformative on a 90/10 split, where predicting "nobody churns" scores 0.90.

The threshold moved from 0.451 to 0.105 between this version and the one
reported earlier in this document, and the model itself did not get worse
— precision and recall at the operating point are essentially unchanged.
What changed is that the raw scores are now calibrated probabilities
rather than inflated ones, so the *same* decision now sits near the ~10%
base rate instead of near 0.5. See Section 6 for the full before/after.

**Honest verdict:** nothing tried meaningfully beats the original
RandomForest baseline. Observed test AUC across every configuration and
training size tried lands in a **0.65–0.68** band, with run-to-run
sampling variance (~0.03) about as large as the differences between
approaches — so claims of one model "beating" another here should be
treated cautiously. LightGBM was adopted for a small CV-confirmed edge
plus ~6x faster training. The practically useful changes were threshold
selection, now derived from expected value rather than a recall heuristic,
and probability calibration, which makes the scores themselves trustworthy
(both in Section 6). This ~0.68 ceiling looks like a genuine property of
the synthetic data's generated signal, not a modeling gap.

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
**2.77** — a month-to-month customer's instantaneous churn risk is
close to triple a longer-contract customer's, at any given tenure.
`num_complaints` (HR 1.20) and low `customer_satisfaction` (HR 0.90 per
point, i.e. protective) are the next-strongest drivers. Model concordance:
0.617 (0.5 = random ranking, 1.0 = perfect). Kaplan-Meier survival:
94.7% of customers are still active at 12 months tenure, 90.2% at 24
months, 82.1% at 48.

**A statistical-significance caution, reported rather than hidden:**
`monthlycharges` reaches p < 0.05 in both the significance tests and the
Cox model, but its hazard ratio is ≈0.9992 per dollar — a customer would
need to pay $100/month more just to see their hazard drop ~8%. At
1,000,000 rows, statistical significance is trivially easy to reach for
practically negligible effects; both numbers are reported together
specifically so this doesn't get miscommunicated as "monthly charges
matter."

**Per-customer explanations (SHAP)**: `TreeExplainer` on the production
LightGBM model, as a global summary plot and as individual waterfall plots
for specific high- and low-risk customers. This matters for a retention
team in a way global importances do not: it answers "why is *this*
customer flagged", which is what a retention agent actually needs before
making a call. The same computation runs live in the dashboard's At-Risk
Customers view, bounded to the rows on screen rather than the full 1M-row
table.

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

The recommender is rebuilt by the pipeline on the same trigger as the
churn model, over a bounded 150,000-profile **reference set** sampled from
`customer_360` (`MAX_REFERENCE_ROWS`). The cap bounds the artifact that the
API and dashboard must load into memory, not just training cost. It does
not limit who can receive recommendations: a customer outside the reference
set — about 85% of the 1,000,000, so the normal case — has their profile
projected into it at request time, giving 100% coverage.

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

### 5a. Is it actually better than not personalizing at all?

A single plausible-looking example is not an evaluation, and until now the
recommender had none - every other model in this project is measured
against a baseline and a held-out split; the recommender's metadata
recorded only structural facts (profile count, neighbor count). That gap
is closed by `src/model/evaluate_recommender.py`, a leave-one-out protocol:
hide one service a customer genuinely owns, present them to the model as
if they didn't own it, and check where the hidden service lands in the
ranking of everything else they don't own. Three rankers are scored on
identical splits - the production k-NN model, a **popularity** baseline
(rank by how common each service is overall), and a **random** floor -
over 4,918 evaluation customers **held out of the reference set entirely**
(scoring the model on profiles it was fitted on would measure memorization,
not generalization).

| Ranker | Hit@1 | Hit@2 | Hit@3 | MRR |
|---|---|---|---|---|
| k-NN (production) | 0.519 | 0.737 | 0.878 | 0.703 |
| Popularity baseline | 0.483 | 0.732 | 0.891 | 0.686 |
| Random | 0.245 | 0.478 | 0.681 | 0.498 |

**The honest result: k-NN beats popularity, but the margin is small, and
popularity is a genuinely hard baseline to clear here.** Both are far
ahead of random - recommending *something* sensible in this domain isn't
hard, because 8 services with skewed adoption rates (internet 84.8%,
phone 76.9%, ... device protection 29.6%) leaves little room for a naive
ranker to be *bad*. The interesting question is whether personalizing on
top of that popularity signal earns its complexity.

It does, but modestly, and the honest evidence required a significance
test rather than eyeballing the table above - a table alone cannot say
whether +2.5% MRR is signal or five thousand customers' worth of noise.
Both checks say it's real: a paired bootstrap (2,000 resamples,
customer-level, since both rankers are scored on the same customers) puts
the MRR gap at **+0.0170, 95% CI [+0.0105, +0.0237]** - entirely above
zero. An exact McNemar test on Hit@1 (knn-only hits: 525, popularity-only
hits: 348) gives **p = 2.3x10^-9**. Both tests answer the same question
from different angles - "is this real" (McNemar) and "how big is it"
(bootstrap) - and agree: k-NN's edge over popularity is real, not sampling
noise, but it is a small edge, not a transformative one. Reported plainly
rather than rounding "statistically significant" up to "big" or down to
"doesn't matter."

## 6. Turning the score into a decision: expected value

A churn probability is not a decision. Choosing *who to actually contact*
requires a threshold, and the earlier heuristic ("guarantee recall ≥ 0.60")
was a judgement call, not an economic one. `notebooks/threshold_and_business_value.ipynb`
replaces it with a cost/benefit derivation.

**The economics** (illustrative assumptions — this synthetic dataset has no
campaign-response data, so these would come from finance in reality): a
retention offer costs **$30**, works **30%** of the time, and a saved
customer is worth **12 months of their own monthly spend** (~$1,036 on
average). The asymmetry is what matters: a wasted offer costs tens of
dollars, a missed churner costs hundreds.

**Results on the 200,000-customer held-out test set:**

| Strategy | Flagged | Recall | Precision | Net value |
|---|---|---|---|---|
| Do nothing | 0 | — | — | $0 |
| Contact everyone | 200,000 | 1.000 | 0.099 | **$88,637** |
| Naive threshold (0.5) | 0 | 0.000 | — | $0 |
| Production threshold (0.105) | 75,182 | 0.616 | 0.163 | $1,419,012 |
| **EV-optimal (0.110)** | 69,980 | 0.588 | 0.167 | **$1,408,304** |

**The headline is unchanged: it's the targeting, not the threshold.**
Contacting everyone returns $89K; any sensibly-targeted strategy returns
~$1.4M. The gap between the production and EV-optimal thresholds
($1.408M–$1.419M) is small next to that, and, notably, the production
threshold edges out the EV-optimal one here — a reminder that
`optimal_threshold` searches over the discrete set of scores the model
actually produced, so its "optimum" can land a hair off the analytic peak.
A 0.75% difference on a $1.4M base is not a finding worth chasing further.

**A dramatic change from the previous version of this analysis: "naive
threshold 0.5" now flags nobody at all.** That is not a bug — it's what a
calibrated model looks like on a ~10%-imbalanced target. When the raw
scores were inflated by `class_weight="balanced"` (mean predicted score
0.41 against a 9.9% actual rate), enough customers cleared 0.5 to make that
threshold look almost reasonable by accident. A calibrated model reports
real probabilities, and for most customers the real probability of
churning is nowhere near 50% — so a threshold copied from a balanced
classification tutorial silently does nothing here. This is exactly the
failure mode calibration and business-threshold selection exist to catch
before it reaches production.

**The model is now calibrated, and the old miscalibration write-up below
is kept as before/after context rather than deleted, since the fix and
its motivation are part of the finding.**

*Before calibration:* the break-even *true* churn probability was 0.096,
but the profit curve peaked at a raw score of 0.470 — not a contradiction,
but `class_weight="balanced"` inflating minority-class scores (mean
predicted 0.413 vs actual 0.099) badly enough that the model's own Brier
score (0.196) was *worse* than just predicting the base rate every time
(0.089). Per-decile, predicted scores ran 3–5x higher than observed churn
rates.

*After calibration* (sigmoid/Platt scaling on a held-out split — see
`src/model/train_churn.py`): Brier score **0.0854**, finally *beating* the
always-base-rate baseline (0.0894) rather than losing to it. Mean predicted
probability **0.100** against an actual rate of **0.099** — no longer 4x
off, but matching almost exactly. The theoretical break-even (0.0965) and
the empirical profit-maximizing threshold (0.110) now sit **0.0135 apart**
instead of 0.096 vs 0.470 — close enough that either could be used directly
as the operating threshold, which was never true of the raw scores.
Calibration cost nothing on ranking quality (AUC 0.6693 either way, since
sigmoid scaling is monotonic) and the operating point at the chosen
threshold is unchanged (precision 0.160, recall 0.600) — only the number's
*meaning* changed, from an arbitrary score to a real probability.

The practical consequence, updated: **these scores can now be presented to
a stakeholder as real probabilities.** A calibrated score of 0.30 means
roughly a 30% real churn rate, not "somewhere between 6% and 30% depending
on which decile you're in." That upgrade — not the small AUC number itself
— is the more consequential outcome of this section.

**Robustness**, re-verified on the calibrated model across the same sweep
of offer costs ($10–$120) and success rates (10%–50%): all 12 combinations
remain profitable. The optimal threshold now ranges more widely (0.01–0.41)
than it did pre-calibration, which is the correct behavior, not a
regression — thresholds on calibrated probabilities should move with the
economics across that much wider a range of assumptions, whereas the old
compressed near-0.4–0.5 thresholds were an artifact of the scores all being
compressed into that same narrow band regardless of the underlying
economics.

## 7. Data drift monitoring

Every arriving batch is scored against a fixed baseline batch using the
**Population Stability Index** across 19 features (14 numeric, 5
categorical), with the standard interpretation bands: below 0.10 stable,
0.10-0.25 investigate, 0.25 and above retrain. Results are persisted to
`public.feature_drift` and shown on the dashboard's Pipeline Status view.

**Result: no drift, and that is the expected answer.** The maximum PSI
observed across all 12 batch pairs and all 19 features is **0.0006** —
roughly three orders of magnitude below the "investigate" band.

This is a property of how the data was constructed, not evidence that the
monitoring works. The 13 batches are sequential slices of a single
pre-shuffled 1,000,000-row file (verified during ingestion: churn rate
across 10 sequential slices all landed within 9.7-10.1%), so the batches
are statistically interchangeable by design. Any real PSI signal here
would indicate a bug in the batch splitter, not genuine market movement.

Because a negative result cannot demonstrate a detector works, the
detector's sensitivity is established separately in `tests/test_drift.py`
(18 tests), which injects shifts and asserts each is caught: mean shifts,
variance shifts with an unchanged mean, missingness jumps, previously
unseen categorical levels, and values outside the reference range.

The practical consequence for this project: retraining is wired to fire on
either a drift signal or a batch-count cadence, and on this dataset it is
always the **cadence** that fires. On a real data feed the drift trigger
would be the one carrying the signal.

## 8. AI Agent Layer - narrating the findings above, not producing new ones

Everything in this document up to here is a number a model, a statistical
test, or a query produced. `src/agents/` (Groq-backed, OpenAI-compatible
API) sits on top of that unchanged, turning it into plain English for a
retention team: SHAP attribution into a 2-3 sentence explanation, an
explanation plus the recommender's top suggestion into a drafted outreach
message, and old-vs-new retrain metrics plus drift results into a summary
paragraph. None of the three agents can change a prediction, a ranking, or
a metric - they read already-final output and describe it.

Verified against this platform's real data, not illustrative text: a real
customer (`CUST0000269643`, 32.6% churn probability) produced a real
explanation grounded in their actual SHAP factors, a real drafted outreach
message that correctly named the real recommended service ("Internet
Service"), and a real Airflow-triggered retrain produced a real summary
correctly identifying an F1/AUC regression (0.2528→0.2411, 0.6693→0.6564)
while correctly calling the smaller precision/recall changes "essentially
unchanged" rather than over-reading noise as a trend.

Full detail - the exact prompts, the guardrails (Postgres caching,
retry/backoff on real rate limits, fallback to raw data on any failure),
and the genuine friction hit along the way (the originally-chosen Groq
model IDs had been retired from Groq's catalog entirely; the replacement
reasoning models needed a token-budget fix after an empty-completion
failure; a cache read-path bug that would have crashed the first real
call) - is in the [README's AI Agent Layer section](../README.md#ai-agent-layer).

## Limitations

See the [README's Limitations section](../README.md#limitations) — synthetic
data, DuckDB-over-Spark tradeoff, simulated batch arrivals, content-based-only
recommendations, and modest churn model performance all apply to every
finding in this report.
