# Discount and Offer Policy

> Demonstration content authored for this project's RAG knowledge corpus.
> Illustrative policy figures (offer costs, success rates) chosen to be
> consistent with the placeholder constants in
> src/model/threshold_analysis.py (OFFER_COST, P_OFFER_SUCCESS) - not
> measured business figures.

## 1. Standard Offer Cost Assumptions

For expected-value analysis of whether a retention offer is worth
extending, this platform uses a standard assumed offer cost and an
assumed probability that an extended offer successfully retains the
customer. These are illustrative planning assumptions, explicitly
documented as such wherever they are used (see
src/model/threshold_analysis.py's own module docstring) - they are not
measured outcomes from a real offer campaign, and any answer that cites
them should say so rather than presenting them as observed fact.

## 2. Who Can Approve an Offer

### 2.1 Standard offers

Offers within the standard cost assumption in section 1 (a contract
discount, a free month, a service-tier adjustment) can be extended by any
retention team member without separate approval, provided the customer
meets the criteria in the retention playbook.

### 2.2 Non-standard offers

Any offer priced above the standard assumption, or any offer type not
described in the retention playbook, requires a team-lead sign-off before
being extended. This exists specifically to prevent offer cost from
silently drifting upward without the expected-value assumptions in
section 1 being revisited.

## 3. Offer Frequency Limits

A customer should not receive more than one active retention offer within
a rolling 90-day window, regardless of how many times they are
re-flagged as at-risk within that window. Re-flagging without a change in
underlying behavior (no new complaint, no new support contact) is treated
as model noise around the decision threshold, not a new signal requiring
a new offer.

## 4. Relationship to the Churn Model's Threshold

This policy does not set or change the churn model's decision threshold.
The threshold is a modeling choice (see the platform's own documentation
on how it is chosen to guarantee a minimum recall), while this policy
governs what happens operationally once a customer crosses that
threshold. An AI assistant answering a question about "why was this
threshold chosen" should cite the model's own documentation, not this
policy - and a question about "what happens after a customer is flagged"
should cite this policy and the retention playbook, not the model's
threshold-selection reasoning.
