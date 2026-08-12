"""Pandera schemas for data validation at each pipeline stage.

These schemas enforce column presence, dtype correctness, and value
range constraints on the 9.5M-record dataset during ingestion and
feature engineering.
"""

from __future__ import annotations

import pandera.pandas as pa
from pandera.pandas import Column, DataFrameSchema

# ---------------------------------------------------------------------------
# Raw ingestion schema — applied to each retailer CSV after initial load
# ---------------------------------------------------------------------------
RAW_DATA_SCHEMA = DataFrameSchema(
    columns={
        "date": Column(
            pa.DateTime,
            coerce=True,
            nullable=False,
            description="Date the price was recorded.",
        ),
        "product_name": Column(
            str,
            nullable=False,
            description="Retailer-specific product name.",
        ),
        "prices": Column(
            float,
            coerce=True,
            nullable=False,
            checks=[
                pa.Check.ge(0, error="Price must be non-negative."),
                pa.Check.le(10_000, error="Price exceeds £10,000 — likely a scraping error."),
            ],
            description="Product price in GBP.",
        ),
        "supermarket": Column(
            str,
            nullable=False,
            checks=pa.Check.isin(
                ["Tesco", "ASDA", "Morrisons", "Sains", "Aldi"],
                error="Unknown supermarket.",
            ),
            description="Retailer name.",
        ),
    },
    strict=False,  # Allow extra columns (e.g., category, unit)
    coerce=True,
    description="Schema for raw retailer CSV data.",
)


# ---------------------------------------------------------------------------
# Canonical products schema — post product-matching
# ---------------------------------------------------------------------------
CANONICAL_PRODUCTS_SCHEMA = DataFrameSchema(
    columns={
        "canonical_name": Column(str, nullable=False),
        "supermarket": Column(
            str,
            nullable=False,
            checks=pa.Check.isin(["Tesco", "ASDA", "Morrisons", "Sains", "Aldi"]),
        ),
        "prices": Column(float, nullable=False, checks=pa.Check.ge(0)),
        "date": Column(pa.DateTime, coerce=True, nullable=False),
        "own_brand": Column(bool, coerce=True, nullable=False),
    },
    strict=False,
    coerce=True,
    description="Schema for the canonical matched products dataset.",
)


# ---------------------------------------------------------------------------
# Feature-engineered schema — pre-training validation
# ---------------------------------------------------------------------------
FEATURE_DATA_SCHEMA = DataFrameSchema(
    columns={
        "price_lag_1d": Column(float, nullable=True),
        "price_rol_mean_7d": Column(float, nullable=True),
        "price_rol_max_7d": Column(float, nullable=True),
        "price_rol_min_7d": Column(float, nullable=True),
        "price_diff_1d": Column(float, nullable=True),
    },
    strict=False,  # Many more columns exist — check only critical ones
    coerce=True,
    description="Schema for feature-engineered training data.",
)
