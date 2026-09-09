-- Demographic attributes, split out from the single wide source table so
-- the intermediate/marts layers can join conceptually distinct slices
-- instead of always reading one monolithic table.
select
    customer_id,
    signup_date,
    age,
    gender,
    annual_income,
    education,
    marital_status,
    dependents,
    senior_citizen
from {{ source('warehouse', 'customers_cleaned') }}
