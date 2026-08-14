"""End-to-end pipeline-integration test.

Runs the *real* ingestion -> matching -> feature-engineering -> training
stage functions against a small, hand-crafted fixture CSV
(``tests/fixtures/sample_raw.csv``), with ``generate_embeddings``
monkeypatched to small deterministic vectors so this test never triggers
a real SBERT/``intfloat/e5-large`` download -- an explicitly flagged trap
in project_refactor.md §13 for any pipeline-integration test that calls
``find_canonical_matches`` end-to-end.

This is deliberately the one test in the suite that drives every stage
function for real (not a fixture-shaped DataFrame handed directly to a
single function, as the per-module unit tests already do) -- it exists to
catch integration bugs at the seams between stages that unit tests, by
construction, cannot see.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from pricepoint import product_matching
from pricepoint.config import Settings, load_settings
from pricepoint.data_ingestion import run_ingestion
from pricepoint.feature_engineering import run_feature_engineering
from pricepoint.product_matching import run_matching
from pricepoint.training import run_training

FIXTURE_CSV = Path(__file__).parent / "fixtures" / "sample_raw.csv"

# Deterministic embeddings for the 5 distinct normalised product names the
# fixture CSV produces (see that file's generation history / row
# comments below for what each represents). Only the two differently
# -worded banana names are close enough (cosine similarity > the real
# config's 0.95 threshold) to cluster together; every other pair is
# orthogonal. This exercises the real threshold + mutual-nearest
# -neighbour clustering logic in `cluster_by_similarity` without any of
# the actual SBERT model.
_FIXTURE_EMBEDDINGS: dict[str, np.ndarray] = {
    "value bananas": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
    "6 bananas value pack": np.array([0.99, 0.1411, 0.0, 0.0], dtype=np.float32),
    "organic avocado twin pack": np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
    "finest chocolate digestives": np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32),
    "free sample item": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
}


def _fake_generate_embeddings(
    product_names: pd.Series, model_name: str = "intfloat/e5-large", batch_size: int = 256
) -> np.ndarray:
    vectors = np.stack([_FIXTURE_EMBEDDINGS[name] for name in product_names])
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return (vectors / norms).astype(np.float32)


@pytest.fixture
def settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Settings:
    """Real, fully-validated Settings pointed at a tmp_path tree, with only
    the paths/hyperparameters that need to differ from production
    overridden -- matching threshold, mutual_k, rolling windows, and lag
    days are all left at their real config.yaml values so this test
    exercises the actual production configuration, not a stand-in."""
    real = load_settings()

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    shutil.copy(FIXTURE_CSV, raw_dir / "sample_raw.csv")

    data = real.data.model_copy(
        update={
            "raw_dir": raw_dir,
            "interim_dir": tmp_path / "interim",
            "processed_dir": tmp_path / "processed",
            "raw_files": ["sample_raw.csv"],
        }
    )
    model = real.model.model_copy(
        update={
            "output_dir": tmp_path / "models",
            # Tiny/fast hyperparameters for a tiny fixture -- production's
            # min_child_samples=100 would leave every tree a single root
            # leaf on ~100 training rows. Mirrors the same override
            # convention already used in tests/api/conftest.py.
            "lgbm_params": {
                "objective": "mae",
                "n_estimators": 30,
                "num_leaves": 7,
                "min_child_samples": 1,
                "verbose": -1,
            },
        }
    )
    overridden = real.model_copy(update={"data": data, "model": model})

    monkeypatch.setattr(product_matching, "generate_embeddings", _fake_generate_embeddings)
    return overridden


class TestPipelineIntegration:
    def test_full_pipeline_end_to_end(self, settings: Settings) -> None:
        # --- Ingestion ---------------------------------------------------
        interim_path = run_ingestion(settings)
        assert interim_path.exists()
        interim_df = pd.read_parquet(interim_path)

        # Fixture has 127 raw rows; exactly one has a missing product name
        # and must be dropped (RAW_DATA_SCHEMA requires product_name
        # non-nullable), the rest must survive.
        assert len(interim_df) == 126
        assert not interim_df["product_name"].isna().any()
        # Zero price is explicitly valid (RAW_DATA_SCHEMA only rejects
        # negative prices) and must survive cleaning -- only NULL prices
        # are dropped.
        assert (interim_df["prices"] == 0).any()

        ingestion_manifest = interim_path.with_suffix(interim_path.suffix + ".manifest.json")
        assert ingestion_manifest.exists()

        # --- Matching ------------------------------------------------------
        canonical_path = run_matching(settings)
        assert canonical_path.exists()
        canonical_df = pd.read_parquet(canonical_path)
        assert len(canonical_df) == len(interim_df)

        banana_rows = canonical_df[canonical_df["product_name"].isin(["Value Bananas 6 Pack", "6 Bananas Value Pack"])]
        assert len(banana_rows) == 50
        assert banana_rows["canonical_name"].nunique() == 1, "differently-worded names should merge via embeddings"
        assert set(banana_rows["supermarket"]) == {"ASDA", "Tesco"}

        avocado_rows = canonical_df[canonical_df["product_name"] == "Organic Avocado Twin Pack"]
        assert avocado_rows["canonical_name"].nunique() == 1, (
            "identical names across retailers should merge via exact dedup"
        )
        assert set(avocado_rows["supermarket"]) == {"ASDA", "Morrisons"}

        digestives_rows = canonical_df[canonical_df["product_name"] == "Finest Chocolate Digestives"]
        assert digestives_rows["canonical_name"].nunique() == 1
        assert digestives_rows["canonical_name"].iloc[0] not in set(banana_rows["canonical_name"])
        assert digestives_rows["canonical_name"].iloc[0] not in set(avocado_rows["canonical_name"])

        matching_manifest = canonical_path.with_suffix(canonical_path.suffix + ".manifest.json")
        assert matching_manifest.exists()

        # --- Feature engineering --------------------------------------------
        feature_path = run_feature_engineering(settings)
        assert feature_path.exists()
        feature_df = pd.read_parquet(feature_path)
        assert len(feature_df) == len(canonical_df)
        for col in [
            "price_rol_mean_7d",
            "price_rol_std_7d",
            "price_lag_1d",
            "price_lag_7d",
            "price_diff_1d",
            "price_vs_market_avg",
            "price_rank",
            "is_cheapest_in_market",
            "day_of_week_sin",
            "day_of_week_cos",
        ]:
            assert col in feature_df.columns

        # Products with same-day competitors (bananas, avocados) get a
        # real leave-one-out comparison; the chocolate digestives /
        # free-sample-item singletons have none, per-row, on every date.
        merged_names = banana_rows["canonical_name"].tolist() + avocado_rows["canonical_name"].tolist()
        competitor_rows = feature_df[feature_df["canonical_name"].isin(merged_names)]
        assert competitor_rows["price_vs_market_avg"].notna().any()

        # --- Training --------------------------------------------------------
        model_path = run_training(settings)
        assert model_path.exists()
        model = joblib.load(model_path)
        assert hasattr(model, "predict")

        metrics_path = settings.model.output_dir / "metrics.json"
        assert metrics_path.exists()
        metrics = json.loads(metrics_path.read_text())
        assert "MAE" in metrics and "RMSE" in metrics
        assert metrics["n_train_rows"] > 0
        assert metrics["n_test_rows"] > 0

    def test_second_run_skips_unchanged_stages(self, settings: Settings) -> None:
        """The manifest-based skip-if-unchanged behaviour (established in
        Phase 1 for each stage independently) must also hold across the
        full chain: re-running every stage with unchanged inputs should
        not rewrite any output."""
        interim_path = run_ingestion(settings)
        canonical_path = run_matching(settings)
        feature_path = run_feature_engineering(settings)
        model_path = run_training(settings)

        interim_mtime = interim_path.stat().st_mtime_ns
        canonical_mtime = canonical_path.stat().st_mtime_ns
        feature_mtime = feature_path.stat().st_mtime_ns

        assert run_ingestion(settings) == interim_path
        assert interim_path.stat().st_mtime_ns == interim_mtime

        assert run_matching(settings) == canonical_path
        assert canonical_path.stat().st_mtime_ns == canonical_mtime

        assert run_feature_engineering(settings) == feature_path
        assert feature_path.stat().st_mtime_ns == feature_mtime

        # Training has no skip-if-unchanged manifest of its own (it always
        # retrains) -- re-running it should still succeed without error.
        assert run_training(settings) == model_path
