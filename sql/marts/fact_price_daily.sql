-- fact_price_daily: one row per (canonical_name, supermarket, date) --
-- the grain every dashboard page and API endpoint actually queries at
-- (project_refactor.md §5).
--
-- AVG(prices), not a raw pass-through: verified directly against the real
-- data that 41.5% of rows (3,956,482 / 9,529,216) are duplicates at this
-- exact grain -- product matching legitimately clusters multiple distinct
-- retailer SKUs (different pack sizes of what's functionally the same
-- product) into one canonical_name, so a single (product, store, day) can
-- have several underlying price observations. AVG is the defensible
-- collapse: it's what "one row per grain" in the mart spec requires, and
-- it's more statistically sound than an arbitrary first-row-wins pick
-- (which is what the current Streamlit dashboard's own
-- drop_duplicates(keep="first") effectively does on the same duplicated
-- data -- not something this mart should perpetuate). MIN/MAX/n_listings
-- are kept alongside the average so a consumer that needs the spread
-- (e.g. a future "price range that day" feature) isn't forced to
-- re-aggregate from raw data.
--
-- Parameters: $1 = path to canonical_products_e5.parquet

SELECT
    canonical_name,
    supermarket,
    date,
    AVG(prices) AS avg_price,
    MIN(prices) AS min_price,
    MAX(prices) AS max_price,
    COUNT(*)    AS n_listings
FROM read_parquet(?)
GROUP BY canonical_name, supermarket, date;
