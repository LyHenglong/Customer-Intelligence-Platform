-- Joins account + service subscription flags to derive per-customer
-- service-value metrics used by both the mart and the recommender.
select
    svc.customer_id,
    svc.num_services,
    svc.total_active_services,
    svc.has_phone_service,
    svc.has_internet_service,
    svc.has_online_security,
    svc.has_online_backup,
    svc.has_device_protection,
    svc.has_tech_support,
    svc.has_streaming_tv,
    svc.has_streaming_movies,
    acct.contract,
    acct.monthlycharges,
    case
        when svc.total_active_services > 0
        then round(acct.monthlycharges / svc.total_active_services, 2)
        else null
    end as cost_per_active_service,
    (acct.contract = 'month_to_month') as is_month_to_month
from {{ ref('stg_customers__services') }} svc
join {{ ref('stg_customers__account') }} acct
    on svc.customer_id = acct.customer_id
