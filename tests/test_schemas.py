"""Tests for Pandera data validation schemas."""

from __future__ import annotations


import pandera
import pytest

from pricepoint.schemas import RAW_DATA_SCHEMA, CANONICAL_PRODUCTS_SCHEMA


class TestRawDataSchema:
    """Tests for RAW_DATA_SCHEMA."""

    def test_valid_data_passes(self, sample_raw_df):
        result = RAW_DATA_SCHEMA.validate(sample_raw_df)
        assert len(result) == 3

    def test_missing_prices_column_fails(self, sample_raw_df):
        df = sample_raw_df.drop(columns=["prices"])
        with pytest.raises((pandera.errors.SchemaError, pandera.errors.SchemaErrors)):
            RAW_DATA_SCHEMA.validate(df, lazy=True)

    def test_negative_price_fails(self, sample_raw_df):
        df = sample_raw_df.copy()
        df.loc[0, "prices"] = -1.0
        with pytest.raises((pandera.errors.SchemaError, pandera.errors.SchemaErrors)):
            RAW_DATA_SCHEMA.validate(df, lazy=True)

    def test_extreme_price_fails(self, sample_raw_df):
        df = sample_raw_df.copy()
        df.loc[0, "prices"] = 99999.0
        with pytest.raises((pandera.errors.SchemaError, pandera.errors.SchemaErrors)):
            RAW_DATA_SCHEMA.validate(df, lazy=True)

    def test_unknown_supermarket_fails(self, sample_raw_df):
        df = sample_raw_df.copy()
        df.loc[0, "supermarket"] = "Lidl"
        with pytest.raises((pandera.errors.SchemaError, pandera.errors.SchemaErrors)):
            RAW_DATA_SCHEMA.validate(df, lazy=True)

    def test_extra_columns_allowed(self, sample_raw_df):
        df = sample_raw_df.copy()
        df["category"] = "Bakery"
        # strict=False means extra columns should be fine
        result = RAW_DATA_SCHEMA.validate(df)
        assert "category" in result.columns


class TestCanonicalProductsSchema:
    """Tests for CANONICAL_PRODUCTS_SCHEMA."""

    def test_valid_data_passes(self, sample_canonical_df):
        result = CANONICAL_PRODUCTS_SCHEMA.validate(sample_canonical_df)
        assert len(result) == 3

    def test_negative_price_fails(self, sample_canonical_df):
        df = sample_canonical_df.copy()
        df.loc[0, "prices"] = -5.0
        with pytest.raises((pandera.errors.SchemaError, pandera.errors.SchemaErrors)):
            CANONICAL_PRODUCTS_SCHEMA.validate(df, lazy=True)
