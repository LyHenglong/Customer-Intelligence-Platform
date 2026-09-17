# Plans and Add-Ons

> Demonstration content authored for this project's RAG knowledge corpus.
> Illustrative plan structure aligned with the contract/payment_method
> categories present in marts.customer_360 - not a real pricing sheet.

## 1. Contract Types

### 1.1 Month-to-Month

No commitment, cancelable at any time. This is the contract type with the
highest observed churn rate in the platform's own segment analysis (see
the dashboard's Segments view) and the primary target of the retention
playbook's contract-conversion offer.

### 1.2 One-Year

A one-year commitment, typically at a modestly reduced effective rate
versus month-to-month. The standard conversion target for the retention
playbook's month-to-month offer.

### 1.3 Two-Year

A two-year commitment, the lowest-churn contract type observed in the
platform's data. Customers on two-year contracts who are nonetheless
flagged as high-risk are treated as a service-quality signal rather than
a price signal - see the churn strategy corpus's discussion of long-tenure
customers.

## 2. Payment Methods

Payment method (credit card, bank transfer, electronic check, or mailed
check, depending on which values are present in a given data batch) is
tracked as a churn model feature but is not, by itself, a lever the
retention team can change as part of an offer - it is a signal to
interpret, not an offer to make. A cluster of high-risk customers sharing
a payment method is worth a data-quality check (is this method associated
with a specific acquisition channel or promotion cohort) before it is
treated as a causal driver.

## 3. Paperless Billing

Paperless billing is an account setting, not a paid add-on. It is
included in the churn model's feature set because engagement with
account self-service tools (enabling paperless billing) correlates with
overall account engagement, not because paperless billing itself affects
service quality.

## 4. How Plans Relate to the Service Catalog

This document describes contract and billing structure; the telecom
services corpus describes what services and add-ons a customer can
subscribe to within a given plan. A recommendation from the content-based
recommender is always a service (see the service catalog), never a
contract-type or payment-method change - contract conversion is a
retention-playbook action, not a recommender output.
