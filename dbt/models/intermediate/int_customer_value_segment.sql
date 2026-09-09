-- Joins demographics + account to derive simple value-segmentation
-- attributes (income relative to spend, tenure bucket) used in the mart
-- and the dashboard's aggregate charts.
select
    dem.customer_id,
    dem.annual_income,
    dem.education,
    dem.marital_status,
    dem.age,
    dem.senior_citizen,
    acct.monthlycharges,
    acct.tenure,
    acct.tenure_years,
    case
        when dem.annual_income is not null and dem.annual_income > 0
        then round((acct.monthlycharges * 12) / dem.annual_income, 4)
        else null
    end as annual_spend_to_income_ratio,
    case
        when acct.tenure < 6 then 'new_0_6mo'
        when acct.tenure < 24 then 'established_6_24mo'
        else 'loyal_24mo_plus'
    end as tenure_bucket
from {{ ref('stg_customers__demographics') }} dem
join {{ ref('stg_customers__account') }} acct
    on dem.customer_id = acct.customer_id
