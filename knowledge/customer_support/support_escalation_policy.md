# Customer Support Escalation Policy

> Demonstration content authored for this project's RAG knowledge corpus.
> Illustrative support procedures, not a real company's policy.

## 1. Escalation Tiers

### 1.1 Tier 1: Standard Contact

Any single support contact (num_service_calls incrementing by one) is
handled at Tier 1 with no special routing. This is the default state for
the large majority of customer contacts and requires no retention-team
involvement.

### 1.2 Tier 2: Repeated Contact

A customer reaching three or more support contacts within a 90-day window
is routed to Tier 2, which includes a mandatory root-cause note on the
account (what the customer contacted about each time, not just how many
times). Tier 2 is the trigger referenced in the retention playbook's
complaint-driven escalation rule.

### 1.3 Tier 3: Complaint Escalation

A logged complaint (distinct from a routine support contact -
num_complaints, not num_service_calls) automatically escalates to Tier 3
regardless of contact count. Tier 3 requires acknowledgment within 24
hours and resolution or a documented retention referral within 5 business
days.

## 2. Relationship Between Support Contacts and Complaints

Support contacts and complaints are tracked as separate signals in this
platform's data model (num_service_calls vs. num_complaints) because they
carry different information: a support contact can be routine (a billing
question, a service change request) while a complaint specifically
indicates dissatisfaction. A customer can have many support contacts and
zero complaints, or few contacts and a complaint on the first one. Both
feed the churn model as independent features, and both should be
considered when an AI assistant explains why a customer is flagged as at
risk - citing only one when both are elevated understates the picture.

## 3. Late Payments

Late payments are tracked separately from both support contacts and
complaints. A pattern of late payments (the late_payments feature) is
treated as a financial-friction signal rather than a service-quality
signal, and the standard response is a billing-team outreach to discuss
payment plan or due-date options, not a retention-team contact - unless
the customer has also crossed the complaint threshold in section 1.3, in
which case the retention team leads and loops in billing.

## 4. What Support Escalation Does Not Do

Support escalation tiers describe internal routing and response-time
commitments. They do not, by themselves, change a customer's churn
probability - the churn model is retrained on a fixed cadence and does
not update in real time when a support ticket is filed. A customer who
was just escalated to Tier 3 may still show a stale, lower churn
probability until the next scheduled retrain incorporates their most
recent activity.
