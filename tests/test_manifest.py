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
import time
from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from pricepoint.manifest import has_sources_changed, load_manifest, write_manifest


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


class TestLoadManifest:
    """Tests for load_manifest function."""

    def test_load_existing_manifest(self, tmp_path):
        """Load a valid manifest file."""
        manifest_data = {
            "stage": "test",
            "output_file": "test.parquet",
            "row_count": 100,
            "schema_hash": "abc123",
        }
        manifest_path = tmp_path / "test.manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest_data, f)

        result = load_manifest(manifest_path)
        assert result is not None
        assert result["stage"] == "test"
        assert result["row_count"] == 100

    def test_load_nonexistent_manifest(self, tmp_path):
        """Return None for a manifest that doesn't exist."""
        manifest_path = tmp_path / "nonexistent.manifest.json"
        result = load_manifest(manifest_path)
        assert result is None

    def test_load_invalid_json(self, tmp_path):
        """Return None for an invalid JSON manifest."""
        manifest_path = tmp_path / "invalid.manifest.json"
        with open(manifest_path, "w") as f:
            f.write("{invalid json")

        result = load_manifest(manifest_path)
        assert result is None


class TestHasSourcesChanged:
    """Tests for has_sources_changed function."""

    def test_no_manifest_means_sources_changed(self, tmp_path):
        """If no manifest exists, sources are considered changed."""
        source = tmp_path / "source.parquet"
        source.write_text("test data")
        manifest = tmp_path / "test.manifest.json"

        assert has_sources_changed([source], manifest) is True

    def test_unchanged_sources(self, tmp_path):
        """If source files haven't been modified, sources are unchanged."""
        source = tmp_path / "source.parquet"
        source.write_text("test data")
        manifest = tmp_path / "test.manifest.json"

        # Write a manifest with a future timestamp (newer than the source)
        manifest_data = {
            "stage": "test",
            "source_files": [str(source)],
            "generated_at": (datetime.now(UTC) + timedelta(hours=1)).isoformat(),
        }
        with open(manifest, "w") as f:
            json.dump(manifest_data, f)

        # Sleep briefly to ensure timestamps differ
        time.sleep(0.01)

        # Source is older than manifest, so unchanged
        assert has_sources_changed([source], manifest) is False

    def test_modified_source_detected(self, tmp_path):
        """If a source file is modified after the manifest, sources are changed."""
        source = tmp_path / "source.parquet"
        source.write_text("original data")
        manifest = tmp_path / "test.manifest.json"

        # Write a manifest with a past timestamp
        manifest_data = {
            "stage": "test",
            "source_files": [str(source)],
            "generated_at": (datetime.now(UTC) - timedelta(hours=1)).isoformat(),
        }
        with open(manifest, "w") as f:
            json.dump(manifest_data, f)

        # Sleep to ensure timestamps differ, then modify the source
        time.sleep(0.01)
        source.write_text("modified data")

        # Source is newer than manifest, so changed
        assert has_sources_changed([source], manifest) is True
