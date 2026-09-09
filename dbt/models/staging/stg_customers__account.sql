-- Account / contract attributes.
select
    customer_id,
    tenure,
    tenure_years,
    contract,
    payment_method,
    paperless_billing,
    monthlycharges,
    totalcharges
from {{ source('warehouse', 'customers_cleaned') }}
