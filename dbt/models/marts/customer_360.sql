-- Customer 360: one row per customer, consolidating demographics, account
-- info, service subscriptions, usage/engagement, value segmentation, and
-- the churn label into a single table. This is the sole read path for both
-- ML training (churn + recommender) and the serving layer - nothing
-- downstream should need to join staging/intermediate models directly.
select
    dem.customer_id,
    dem.signup_date,
    dem.age,
    dem.gender,
    dem.annual_income,
    dem.education,
    dem.marital_status,
    dem.dependents,
    dem.senior_citizen,

    acct.tenure,
    acct.tenure_years,
    acct.contract,
    acct.payment_method,
    acct.paperless_billing,
    acct.monthlycharges,
    acct.totalcharges,

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
    isvc.cost_per_active_service,
    isvc.is_month_to_month,

    usg.customer_satisfaction,
    usg.num_complaints,
    usg.num_service_calls,
    usg.late_payments,
    usg.avg_monthly_gb,
    usg.avg_gb_per_service,
    usg.days_since_last_interaction,
    usg.credit_score,
    iusg.complaints_per_tenure_month,
    iusg.is_disengaged,

    ival.annual_spend_to_income_ratio,
    ival.tenure_bucket,

    chn.churn

from {{ ref('stg_customers__demographics') }} dem
join {{ ref('stg_customers__account') }} acct           on dem.customer_id = acct.customer_id
join {{ ref('stg_customers__services') }} svc            on dem.customer_id = svc.customer_id
join {{ ref('stg_customers__usage') }} usg                on dem.customer_id = usg.customer_id
join {{ ref('stg_customers__churn') }} chn                on dem.customer_id = chn.customer_id
join {{ ref('int_customer_service_summary') }} isvc       on dem.customer_id = isvc.customer_id
join {{ ref('int_customer_usage_summary') }} iusg         on dem.customer_id = iusg.customer_id
join {{ ref('int_customer_value_segment') }} ival         on dem.customer_id = ival.customer_id
