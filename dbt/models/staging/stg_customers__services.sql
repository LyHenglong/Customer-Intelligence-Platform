-- Service subscription flags.
select
    customer_id,
    num_services,
    total_active_services,
    has_phone_service,
    has_internet_service,
    has_online_security,
    has_online_backup,
    has_device_protection,
    has_tech_support,
    has_streaming_tv,
    has_streaming_movies
from {{ source('warehouse', 'customers_cleaned') }}
