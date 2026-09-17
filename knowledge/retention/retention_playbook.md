# Retention Playbook

> Demonstration content authored for this project's RAG knowledge corpus.
> These are illustrative retention procedures written to exercise the AI
> assistant's retrieval and citation behavior against realistic-looking
> business documentation - they are not a real company's policy and
> should not be treated as one.

## 1. Purpose

This playbook defines when and how retention offers are extended to
at-risk customers identified by the churn model, and which offer applies
to which customer segment. It is the primary source the AI assistant
should cite when a user asks what retention action is recommended for a
customer or customer segment.

## 2. Risk Tiers and Standard Response

### 2.1 High risk, month-to-month contract

Customers on a month-to-month contract flagged above the model's decision
threshold are the highest-priority segment: this group churns at roughly
four times the rate of one- and two-year contract holders. The standard
response is a contract-conversion offer - one month free in exchange for
converting to a one-year contract - rather than a price discount, because
a locked-in contract addresses the underlying churn driver (no switching
cost) rather than only the symptom.

### 2.2 High risk, long-tenure customers

A customer flagged as high-risk despite two or more years of tenure is
treated as a service-quality signal, not a price signal. The recommended
action is a personal outreach call from a retention specialist within 48
hours, focused on diagnosing a specific service issue (drops in call
quality, billing disputes, or unresolved support tickets) rather than an
automated discount offer.

### 2.3 High risk, low monthly value

For customers below the median monthly charge who are flagged as
high-risk, the cost of an aggressive discount can exceed the customer's
remaining lifetime value. The recommended action is a service-tier
review: offer a right-sized downgrade (fewer add-ons, lower price) rather
than a discount on the current plan, since these customers are frequently
over-subscribed relative to actual usage.

## 3. Complaint-Driven Escalation

Any customer with three or more logged complaints in the trailing 90 days
is escalated regardless of their churn probability. The retention team's
service-level target is first contact within 24 hours of the third
complaint. This overrides the standard risk-tier response above - a
low-risk customer with repeated complaints is still escalated, because
complaint volume is a leading indicator the churn model has not yet
captured in that billing cycle.

## 4. What Retention Offers Do Not Cover

Retention offers described in this playbook apply to existing customers
identified by the churn model or the complaint-escalation rule above.
They are not acquisition offers, are not available to new signups, and
should not be described as guaranteed to prevent churn - they are
interventions with directional evidence behind them, not guarantees.

## 5. Measuring Offer Effectiveness

Offer effectiveness should be measured against a holdout group that
receives no offer, using the same threshold-based flagging the churn
model already produces. See the churn strategy corpus for how expected
value and offer cost are weighed against each other when deciding whether
an offer is worth extending at all.
