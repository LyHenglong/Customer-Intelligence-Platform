-- Engagement / usage / support-interaction attributes.
select
    customer_id,
    customer_satisfaction,
    num_complaints,
    num_service_calls,
    late_payments,
    avg_monthly_gb,
    avg_gb_per_service,
    days_since_last_interaction,
    credit_score
from {{ source('warehouse', 'customers_cleaned') }}
