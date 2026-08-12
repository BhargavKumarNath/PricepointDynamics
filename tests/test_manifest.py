"""Tests for pricepoint/manifest.py -- the manifest.json sidecar generator.

Exercised directly against a synthetic DataFrame rather than through the
full ingestion/feature-engineering pipelines, since those require the raw
retailer CSVs (not present in this environment -- see project_v2.md Phase 1
Progress Log for the data-availability blocker). This still proves the
manifest-writing logic itself is correct; wiring is verified by inspection
of data_ingestion.py::run_ingestion and feature_engineering.py::run_feature_engineering.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from pricepoint.manifest import write_manifest


@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})


class TestWriteManifest:
    def test_manifest_file_written_alongside_artifact(self, tmp_path, sample_df):
        output_path = tmp_path / "output.parquet"
        output_path.write_bytes(b"not a real parquet file, just needs to exist")

        manifest_path = write_manifest(output_path, sample_df, ["source.csv"], stage="ingestion")

        assert manifest_path == tmp_path / "output.parquet.manifest.json"
        assert manifest_path.exists()

    def test_manifest_contents(self, tmp_path, sample_df):
        output_path = tmp_path / "output.parquet"
        manifest_path = write_manifest(output_path, sample_df, ["a.csv", "b.csv"], stage="feature_engineering")

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        assert manifest["stage"] == "feature_engineering"
        assert manifest["row_count"] == 3
        assert manifest["column_count"] == 2
        assert manifest["columns"] == ["a", "b"]
        assert manifest["source_files"] == ["a.csv", "b.csv"]
        assert "schema_hash" in manifest
        assert "generated_at" in manifest
        # git_sha may be None outside a git repo, but the key must exist.
        assert "git_sha" in manifest

    def test_schema_hash_changes_when_schema_changes(self, tmp_path, sample_df):
        path_a = tmp_path / "a.parquet"
        path_b = tmp_path / "b.parquet"

        manifest_a = json.loads(write_manifest(path_a, sample_df, [], stage="test").read_text(encoding="utf-8"))

        different_df = sample_df.copy()
        different_df["c"] = [1.0, 2.0, 3.0]
        manifest_b = json.loads(write_manifest(path_b, different_df, [], stage="test").read_text(encoding="utf-8"))

        assert manifest_a["schema_hash"] != manifest_b["schema_hash"]

    def test_schema_hash_stable_for_same_schema(self, tmp_path, sample_df):
        path_a = tmp_path / "a.parquet"
        path_b = tmp_path / "b.parquet"

        manifest_a = json.loads(write_manifest(path_a, sample_df, [], stage="test").read_text(encoding="utf-8"))
        manifest_b = json.loads(write_manifest(path_b, sample_df.copy(), [], stage="test").read_text(encoding="utf-8"))

        assert manifest_a["schema_hash"] == manifest_b["schema_hash"]
