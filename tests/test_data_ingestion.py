"""Tests for the Polars-backed ingestion pipeline (Phase 2 rewrite).

data_ingestion.py previously had zero test coverage. These tests cover
both the new Polars-native load_raw_csvs/clean_raw_data functions and a
tolerance-based regression comparison against the pre-Phase-2 pandas
reference implementation, per project_refactor.md Phase 2's testing
requirement.
"""

from __future__ import annotations

import json
import time

import pandas as pd
import polars as pl
import pytest

import pricepoint.data_ingestion as data_ingestion_module
from pricepoint.data_ingestion import clean_raw_data, load_raw_csvs, run_ingestion, validate_data


def _old_clean_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """Pre-Phase-2 pandas reference implementation of clean_raw_data.

    Kept only in this test module as the "old" side of an old-vs-new
    regression comparison for the Polars rewrite, per Phase 2's testing
    requirement -- not used in production code.
    """
    df = df.copy()
    df.columns = df.columns.str.lower().str.replace(r"[^\w]+", "_", regex=True).str.strip("_")
    df = df.rename(columns={"names": "product_name"})
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], format="%Y%m%d", errors="coerce")
    for col in df.select_dtypes(include=["object", "str"]).columns:
        try:
            df[col] = df[col].str.strip()
        except AttributeError:
            pass
    if "prices" in df.columns:
        df["prices"] = pd.to_numeric(df["prices"], errors="coerce")
        df = df.dropna(subset=["prices"])
    if "product_name" in df.columns:
        df = df.dropna(subset=["product_name"])
    return df


@pytest.fixture
def raw_csv_dir(tmp_path):
    """A directory of small synthetic retailer CSVs mirroring the real raw format."""
    csv_content = (
        "supermarket,prices_(£),prices_unit_(£),unit,names,date,category,own_brand\n"
        "Tesco, 1.50 ,0.30,100g,  Baked Beans ,20240101,tinned,False\n"
        "Tesco,2.00,0.40,100g,Tomato Soup,20240102,tinned,True\n"
        "Tesco,,0.10,100g,Bad Row No Price,20240103,tinned,False\n"
        "Tesco,3.00,0.50,100g,,20240104,tinned,False\n"
    )
    csv_path = tmp_path / "All_Data_Tesco.csv"
    csv_path.write_text(csv_content)
    return tmp_path


class TestLoadRawCsvs:
    def test_returns_polars_dataframe(self, raw_csv_dir):
        class _FakeDataConfig:
            raw_dir = raw_csv_dir
            raw_files = ["All_Data_Tesco.csv"]

        class _FakeSettings:
            data = _FakeDataConfig()

        result = load_raw_csvs(_FakeSettings())
        assert isinstance(result, pl.DataFrame)
        assert len(result) == 4

    def test_missing_file_raises_if_none_found(self, tmp_path):
        class _FakeDataConfig:
            raw_dir = tmp_path
            raw_files = ["does_not_exist.csv"]

        class _FakeSettings:
            data = _FakeDataConfig()

        with pytest.raises(FileNotFoundError):
            load_raw_csvs(_FakeSettings())

    def test_skips_missing_file_if_others_exist(self, raw_csv_dir):
        class _FakeDataConfig:
            raw_dir = raw_csv_dir
            raw_files = ["All_Data_Tesco.csv", "does_not_exist.csv"]

        class _FakeSettings:
            data = _FakeDataConfig()

        result = load_raw_csvs(_FakeSettings())
        assert len(result) == 4


class TestCleanRawData:
    @pytest.fixture
    def raw_pl_df(self) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "supermarket": ["Tesco", "Tesco", "Tesco", "Tesco"],
                "prices_(£)": [1.50, 2.00, None, 3.00],
                "prices_unit_(£)": [0.30, 0.40, 0.10, 0.50],
                "unit": ["100g", "100g", "100g", "100g"],
                "names": [" Baked Beans ", "Tomato Soup", "Bad Row No Price", None],
                "date": [20240101, 20240102, 20240103, 20240104],
                "category": ["tinned", "tinned", "tinned", "tinned"],
                "own_brand": [False, True, False, False],
            }
        )

    def test_column_names_normalised(self, raw_pl_df):
        result = clean_raw_data(raw_pl_df)
        assert "prices" in result.columns
        assert "product_name" in result.columns
        assert "prices_(£)" not in result.columns
        assert "names" not in result.columns

    def test_null_prices_dropped(self, raw_pl_df):
        result = clean_raw_data(raw_pl_df)
        assert result["prices"].null_count() == 0

    def test_null_product_name_dropped(self, raw_pl_df):
        result = clean_raw_data(raw_pl_df)
        assert result["product_name"].null_count() == 0

    def test_whitespace_stripped(self, raw_pl_df):
        result = clean_raw_data(raw_pl_df)
        names = result["product_name"].to_list()
        assert "Baked Beans" in names
        assert " Baked Beans " not in names

    def test_date_coerced_to_date_type(self, raw_pl_df):
        result = clean_raw_data(raw_pl_df)
        assert result["date"].dtype == pl.Date

    def test_own_brand_stays_boolean_not_stripped(self, raw_pl_df):
        """own_brand is Boolean-typed by Polars, so the whitespace-strip
        step (which only targets Utf8 columns) must never touch it."""
        result = clean_raw_data(raw_pl_df)
        assert result["own_brand"].dtype == pl.Boolean

    def test_row_count_after_cleaning(self, raw_pl_df):
        # 4 rows in -> 1 null price, 1 null name -> 2 valid rows remain
        result = clean_raw_data(raw_pl_df)
        assert len(result) == 2

    def test_matches_pandas_reference_implementation(self, raw_pl_df):
        """Tolerance-based old-vs-new regression: the Polars rewrite must
        produce the same cleaned rows as the pre-Phase-2 pandas version."""
        raw_pd_df = raw_pl_df.to_pandas()

        new_result = clean_raw_data(raw_pl_df).to_pandas()
        new_result["date"] = pd.to_datetime(new_result["date"])
        old_result = _old_clean_raw_data(raw_pd_df)

        cols = sorted(old_result.columns)
        new_sorted = new_result[cols].sort_values(cols).reset_index(drop=True)
        old_sorted = old_result[cols].sort_values(cols).reset_index(drop=True)

        assert len(new_sorted) == len(old_sorted)
        for col in cols:
            if pd.api.types.is_float_dtype(old_sorted[col]):
                pd.testing.assert_series_equal(
                    new_sorted[col].reset_index(drop=True),
                    old_sorted[col].reset_index(drop=True),
                    check_names=False,
                    rtol=1e-6,
                )
            else:
                assert new_sorted[col].astype(str).tolist() == old_sorted[col].astype(str).tolist(), (
                    f"Mismatch in column {col!r}"
                )


class TestValidateData:
    def test_valid_cleaned_data_passes(self, sample_raw_df):
        result = validate_data(sample_raw_df)
        assert len(result) == len(sample_raw_df)


class _FakeDataConfig:
    def __init__(self, raw_dir, raw_files, interim_dir):
        self.raw_dir = raw_dir
        self.raw_files = raw_files
        self.interim_dir = interim_dir


class _FakeIngestionSettings:
    def __init__(self, raw_dir, raw_files, interim_dir):
        self.data = _FakeDataConfig(raw_dir, raw_files, interim_dir)


class TestRunIngestionSkipIfUnchanged:
    """Integration tests for the manifest-based skip-if-unchanged wiring.

    Phase 1 originally shipped `has_sources_changed()` fully implemented
    and unit-tested (see test_manifest.py) but never actually called it
    from run_ingestion/run.py -- the CLI always re-ran ingestion
    regardless of whether the raw CSVs had changed. These tests cover the
    actual wiring, not just the underlying comparison function.
    """

    @pytest.fixture
    def settings(self, tmp_path, raw_csv_dir):
        interim_dir = tmp_path / "interim"
        return _FakeIngestionSettings(
            raw_dir=raw_csv_dir,
            raw_files=["All_Data_Tesco.csv"],
            interim_dir=interim_dir,
        )

    def test_second_run_skips_when_unchanged(self, settings, monkeypatch):
        first_path = run_ingestion(settings)
        assert first_path.exists()

        # If load_raw_csvs is called on the second run, the skip logic
        # failed to trigger -- fail loudly rather than silently re-running.
        def _fail_if_called(*args, **kwargs):
            raise AssertionError("load_raw_csvs was called on an unchanged re-run; skip logic did not trigger")

        monkeypatch.setattr(data_ingestion_module, "load_raw_csvs", _fail_if_called)

        second_path = run_ingestion(settings)
        assert second_path == first_path

    def test_force_reruns_even_when_unchanged(self, settings, monkeypatch):
        run_ingestion(settings)

        calls = []
        original = data_ingestion_module.load_raw_csvs
        monkeypatch.setattr(
            data_ingestion_module, "load_raw_csvs", lambda *a, **kw: (calls.append(1), original(*a, **kw))[1]
        )

        run_ingestion(settings, force=True)
        assert len(calls) == 1, "force=True must re-run ingestion even when sources are unchanged"

    def test_reruns_when_source_file_modified(self, settings, raw_csv_dir):
        first_path = run_ingestion(settings)
        first_manifest = json.loads(first_path.with_suffix(first_path.suffix + ".manifest.json").read_text())

        # Modify the raw CSV -- mtime must advance past the manifest's
        # generated_at for has_sources_changed to detect it.
        time.sleep(0.01)
        csv_path = raw_csv_dir / "All_Data_Tesco.csv"
        csv_path.write_text(csv_path.read_text() + "Tesco,4.00,0.60,100g,New Product,20240105,tinned,False\n")

        second_path = run_ingestion(settings)
        second_manifest = json.loads(second_path.with_suffix(second_path.suffix + ".manifest.json").read_text())
        assert second_manifest["generated_at"] != first_manifest["generated_at"]
        assert second_manifest["row_count"] > first_manifest["row_count"]
