-- Churn label, kept as its own staging model so the ML-target column is
-- unambiguous and easy to exclude from feature sets by simply not joining it.
select
    customer_id,
    churn
from {{ source('warehouse', 'customers_cleaned') }}
