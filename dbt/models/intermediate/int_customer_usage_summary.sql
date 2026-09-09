-- Joins account (tenure) + usage/support signals to derive engagement-risk
-- metrics: complaint rate normalized by how long the customer has been
-- around, and a simple disengagement flag.
select
    usg.customer_id,
    usg.customer_satisfaction,
    usg.num_complaints,
    usg.num_service_calls,
    usg.late_payments,
    usg.avg_monthly_gb,
    usg.avg_gb_per_service,
    usg.days_since_last_interaction,
    usg.credit_score,
    case
        when acct.tenure > 0
        then round(usg.num_complaints / acct.tenure, 4)
        else usg.num_complaints
    end as complaints_per_tenure_month,
    (usg.days_since_last_interaction > 90) as is_disengaged
from {{ ref('stg_customers__usage') }} usg
join {{ ref('stg_customers__account') }} acct
    on usg.customer_id = acct.customer_id
