{#
    dbt's default behavior prefixes custom schemas with the profile target
    schema (e.g. "public_marts" instead of "marts"), which silently
    diverged from the staging/intermediate/marts schemas created in
    db/schema.sql. This override uses the custom schema name exactly as
    configured in dbt_project.yml, with no prefix.
#}
{% macro generate_schema_name(custom_schema_name, node) -%}
    {%- if custom_schema_name is none -%}
        {{ target.schema }}
    {%- else -%}
        {{ custom_schema_name | trim }}
    {%- endif -%}
{%- endmacro %}
