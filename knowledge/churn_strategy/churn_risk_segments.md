# Churn Risk Segments and Strategy

> Demonstration content authored for this project's RAG knowledge corpus.
> Segment definitions and cost assumptions are illustrative business
> reasoning meant to give the AI assistant something concrete to cite,
> not measured facts about a real customer base.

## 1. Why Segment-Level Strategy Exists

The churn model produces one probability per customer. This document
exists to translate that probability, combined with a handful of other
features, into a strategy question: is this customer worth an active
retention intervention, and if so, which kind. It intentionally
duplicates none of the model's own logic - it is business reasoning
layered on top of the model's output, not a restatement of it.

## 2. The Three Segments

### 2.1 Save-worthy

A customer is save-worthy when their predicted churn probability is above
the model's decision threshold and their expected 12-month value (roughly,
monthly charges times twelve) exceeds the cost of a typical retention
offer by a wide margin. This is the segment the retention playbook's
tiered responses are built for - see the retention corpus for what to
actually do once a customer lands here.

### 2.2 Marginal

A customer is marginal when they are flagged at risk but their expected
value only modestly exceeds offer cost, or when their risk factors are
ambiguous (a mix of positive and negative SHAP contributions with no
single dominant driver). The recommended default for marginal customers
is the lowest-cost intervention available - a service-tier review or an
automated email - rather than a high-touch phone call, since the expected
return does not clearly justify a specialist's time.

### 2.3 Not cost-effective to save

A customer is not cost-effective to save when the cost of any standard
retention offer would exceed their expected remaining value even if the
offer worked with certainty. This is not a judgment that the customer
does not matter - it is a statement that an active retention campaign is
the wrong tool for this customer, and that service-quality issues (if
any) are better addressed through the standard support path than a
special retention offer.

## 3. How This Interacts With the Model's Threshold

The model's decision threshold is chosen to guarantee a minimum recall on
churners, not to directly encode a cost-benefit tradeoff - see the
platform's own documentation on threshold selection for why. The
segments in section 2 are a second pass on top of that threshold: every
save-worthy and marginal customer is, by construction, already above the
threshold, but not every customer above the threshold is save-worthy.
This is why the AI assistant should never treat "flagged as at-risk" and
"worth an active intervention" as the same statement.

## 4. Explaining Risk Factors to a Non-Technical Audience

When a customer's SHAP factors are surfaced (see the model's own
explanation output), they should be framed as *directional evidence*,
not causal proof: a factor that increases predicted risk is a pattern the
model learned from historical data, not a guaranteed cause of this
specific customer's likelihood to leave. This distinction matters most
when a factor is something the business cannot easily change (e.g.
tenure) versus something it can (e.g. an unresolved complaint) - the
latter is what a retention action should target.
