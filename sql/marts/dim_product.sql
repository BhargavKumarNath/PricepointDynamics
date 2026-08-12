-- dim_product: one row per canonical_name.
--
-- category and own_brand use mode() (the most frequent value per
-- canonical_name), not an arbitrary "first row wins" pick: verified
-- directly against the real data that ~20% of canonical products (13,776
-- / 68,596) have more than one distinct `category` label across their
-- constituent matched rows, and ~1% (746 / 68,596) have inconsistent
-- `own_brand` flags -- both are a direct consequence of product matching
-- merging multiple distinct retailer SKUs (different pack sizes, or a
-- generic product name matched across an own-brand and a branded listing)
-- into one canonical cluster. mode() is a defensible, deterministic
-- resolution; it is not claiming these products are perfectly
-- category-consistent, just picking the single best representative label
-- for a mart row that -- by definition -- can only carry one.
--
-- first_seen_date/last_seen_date are the observed date range in the
-- source data, not "when this product first went on sale" (this is a
-- historical Kaggle snapshot, not a live feed with true product-launch
-- dates).
--
-- Parameters: $1 = path to canonical_products_e5.parquet

SELECT
    canonical_name,
    mode(category)   AS category,
    mode(own_brand)  AS own_brand,
    MIN(date)        AS first_seen_date,
    MAX(date)        AS last_seen_date,
    COUNT(DISTINCT supermarket) AS n_retailers
FROM read_parquet(?)
GROUP BY canonical_name;
