-- dim_retailer: the 5 supermarkets. Static — a real dimension table would
-- be overkill for 5 known values, but keeping it as a mart (rather than a
-- hardcoded Python list) means every retailer name a query might need
-- comes from the same source of truth as fact_price_daily's foreign key,
-- with zero risk of a hardcoded list silently drifting from the data
-- (e.g. a 6th retailer appearing in a future data refresh).
--
-- Parameters (bound via DuckDB's `?` positional parameter, see
-- scripts/materialize_marts.py): $1 = path to canonical_products_e5.parquet

SELECT DISTINCT
    supermarket AS retailer_name
FROM read_parquet(?)
ORDER BY retailer_name;
