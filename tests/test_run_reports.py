"""Tests for pricepoint/run_reports.py -- the per-stage JSON run report
writer (project_refactor.md §14). Exercised directly against synthetic
data; wiring into the pipeline stages themselves is covered by
tests/test_pipeline_integration.py, which asserts real reports land on
disk during a genuine ingest -> match -> features -> train run.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from pricepoint.run_reports import default_report_dir, write_run_report


@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", None]})


class TestWriteRunReport:
    def test_report_file_written(self, tmp_path, sample_df):
        report_dir = tmp_path / "_run_reports"
        report_path = write_run_report("ingestion", report_dir, rows_in=5, df_out=sample_df, duration_seconds=1.234)

        assert report_path.exists()
        assert report_path.parent == report_dir
        assert report_path.name.startswith("ingestion_")
        assert report_path.suffix == ".json"

    def test_report_contents(self, tmp_path, sample_df):
        report_dir = tmp_path / "_run_reports"
        report_path = write_run_report(
            "feature_engineering", report_dir, rows_in=5, df_out=sample_df, duration_seconds=2.5
        )

        report = json.loads(report_path.read_text())
        assert report["stage"] == "feature_engineering"
        assert report["rows_in"] == 5
        assert report["rows_out"] == 3
        assert report["row_delta"] == -2
        assert report["null_counts"] == {"b": 1}
        assert report["duration_seconds"] == 2.5
        assert isinstance(report["schema_hash"], str) and len(report["schema_hash"]) == 16
        assert "generated_at" in report

    def test_no_null_columns_omitted_from_null_counts(self, tmp_path):
        df = pd.DataFrame({"a": [1, 2, 3]})
        report_path = write_run_report("ingestion", tmp_path, rows_in=3, df_out=df, duration_seconds=0.1)
        report = json.loads(report_path.read_text())
        assert report["null_counts"] == {}

    def test_extra_fields_merged_in(self, tmp_path, sample_df):
        report_path = write_run_report(
            "training",
            tmp_path,
            rows_in=3,
            df_out=sample_df,
            duration_seconds=0.1,
            extra={"test_rows": 10, "mae": 0.15},
        )
        report = json.loads(report_path.read_text())
        assert report["test_rows"] == 10
        assert report["mae"] == 0.15

    def test_two_reports_for_the_same_stage_do_not_collide(self, tmp_path, sample_df):
        """Reports are historical/additive, not overwritten like a manifest
        -- back-to-back runs must not clobber each other's file."""
        first = write_run_report("ingestion", tmp_path, rows_in=3, df_out=sample_df, duration_seconds=0.1)
        second = write_run_report("ingestion", tmp_path, rows_in=3, df_out=sample_df, duration_seconds=0.1)
        assert first != second
        assert first.exists()
        assert second.exists()

    def test_large_row_loss_logs_a_warning(self, tmp_path, sample_df, caplog):
        with caplog.at_level("WARNING"):
            write_run_report("ingestion", tmp_path, rows_in=100, df_out=sample_df, duration_seconds=0.1)
        assert any("dropped" in record.message.lower() for record in caplog.records)

    def test_small_row_loss_does_not_warn(self, tmp_path, sample_df, caplog):
        with caplog.at_level("WARNING"):
            write_run_report("ingestion", tmp_path, rows_in=3, df_out=sample_df, duration_seconds=0.1)
        assert not any("dropped" in record.message.lower() for record in caplog.records)

    def test_row_growth_never_warns(self, tmp_path, sample_df, caplog):
        """A stage that adds rows (or an unchanged rows_in=0 edge case)
        must never be flagged as a "loss"."""
        with caplog.at_level("WARNING"):
            write_run_report("feature_engineering", tmp_path, rows_in=0, df_out=sample_df, duration_seconds=0.1)
        assert not any("dropped" in record.message.lower() for record in caplog.records)


class TestDefaultReportDir:
    def test_derives_from_raw_dir_parent(self, tmp_path):
        class _FakeDataConfig:
            raw_dir = tmp_path / "data" / "00_raw"

        class _FakeSettings:
            data = _FakeDataConfig()

        assert default_report_dir(_FakeSettings()) == tmp_path / "data" / "_run_reports"
