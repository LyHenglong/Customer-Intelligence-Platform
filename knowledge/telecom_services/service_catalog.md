# Service Catalog

> Demonstration content authored for this project's RAG knowledge corpus.
> Service names and descriptions are illustrative and align with the
> service flags present in marts.customer_360 (has_phone_service,
> has_internet_service, has_online_security, has_online_backup,
> has_device_protection, has_tech_support, has_streaming_tv,
> has_streaming_movies) - not a real provider's offering.

## 1. Core Connectivity Services

### 1.1 Phone Service

Standard voice line service. The baseline service most customer records
in this dataset carry; its absence is unusual enough to be worth checking
for data quality issues before treating it as a real gap in a customer's
subscription.

### 1.2 Internet Service

Broadband internet, the entry point for every other add-on in this
catalog - online security, online backup, device protection, tech
support, and both streaming add-ons are only meaningful for a customer
who already has internet service. A customer flagged as high-risk with
internet service but none of the five add-ons below is a common upsell
target, not necessarily a churn risk in itself.

## 2. Protection Add-Ons

### 2.1 Online Security

Network-level protection against common threats (malicious sites,
phishing attempts) bundled at the account level rather than per device.

### 2.2 Online Backup

Automatic backup of account-linked data. Positioned as a low-friction
add-on: it requires no behavior change from the customer once enabled,
which is why it appears in outreach drafts for customers whose risk
factors suggest low engagement rather than price sensitivity.

### 2.3 Device Protection

Coverage for connected devices against damage or malfunction. Most
frequently recommended to customers with multiple active services, since
device protection's per-device value scales with how much hardware the
customer has connected.

### 2.4 Tech Support

Priority technical support queue. This is the add-on most directly tied
to the num_service_calls and num_complaints features the churn model
uses - a customer already generating a high volume of support contact is
a stronger candidate for this add-on than one with no support history at
all, since it is priced as a convenience, not a fix for an unrelated
problem.

## 3. Entertainment Add-Ons

### 3.1 Streaming TV

Live and on-demand television streaming, billed as an internet add-on.

### 3.2 Streaming Movies

On-demand movie streaming, billed as an internet add-on. Streaming TV and
Streaming Movies are priced and marketed as a pair - a customer with one
but not the other is the standard target for a "complete the pair"
recommendation, which is the most common output of the content-based
recommender for customers who already have four or more active services.

## 4. How This Catalog Relates to Recommendations

The recommender described elsewhere in this platform is content-based: it
looks at a customer's current service profile and demographic similarity
to other customers, then recommends the most common service among similar
customers that this customer does not yet have. This catalog exists so
the AI assistant can explain *what* a recommended service actually is
when it surfaces a recommendation - the recommender itself, not this
document, decides *which* service to recommend for a specific customer.
