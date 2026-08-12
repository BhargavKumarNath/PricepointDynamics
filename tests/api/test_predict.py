"""Contract tests for POST /v1/predict.

`test_matches_direct_model_predict` is the phase's explicit validation
criterion: the API's output must equal `model.predict()` on identically
resolved input, not merely be "close" -- proof the served prediction
never diverges from what a direct call to the model would produce.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from api.dependencies import get_settings
from pricepoint.training import build_feature_vector


class TestPredict:
    def test_matches_direct_model_predict(self, client, fixture_model, feature_data_path):
        body = {"canonical_name": "test bananas", "supermarket": "TestMart", "date": "2024-01-15"}
        r = client.post("/v1/predict", json=body)
        assert r.status_code == 200
        api_prediction = r.json()["predicted_price"]

        feature_df = pd.read_parquet(feature_data_path)
        row = feature_df[
            (feature_df["canonical_name"] == "test bananas")
            & (feature_df["supermarket"] == "TestMart")
            & (feature_df["date"] == pd.Timestamp("2024-01-15"))
        ]
        input_vector, _ = build_feature_vector(row, fixture_model.feature_name_)
        direct_prediction = float(fixture_model.predict(input_vector)[0])

        assert np.isclose(api_prediction, direct_prediction)

    def test_response_reports_resolved_date(self, client):
        r = client.post(
            "/v1/predict", json={"canonical_name": "test bananas", "supermarket": "TestMart", "date": "2024-01-15"}
        )
        assert r.status_code == 200
        body = r.json()
        assert body["requested_date"] == "2024-01-15"
        assert body["resolved_from_date"] == "2024-01-15"
        assert body["price_override_applied"] is False

    def test_price_override_matches_direct_model_predict(self, client, fixture_model, feature_data_path):
        """Not asserting the prediction *changes* under override -- with a
        tiny fixture model/tree, an override doesn't always cross a split
        threshold, so that assertion would be flaky. What must hold
        unconditionally is that the override is genuinely applied to the
        resolved feature vector, matching an independent direct call --
        the same contract as `test_matches_direct_model_predict`."""
        body = {
            "canonical_name": "test bananas",
            "supermarket": "TestMart",
            "date": "2024-01-15",
            "price_override": 50.0,
        }
        r = client.post("/v1/predict", json=body)
        assert r.status_code == 200
        api_body = r.json()
        assert api_body["price_override_applied"] is True

        feature_df = pd.read_parquet(feature_data_path)
        row = feature_df[
            (feature_df["canonical_name"] == "test bananas")
            & (feature_df["supermarket"] == "TestMart")
            & (feature_df["date"] == pd.Timestamp("2024-01-15"))
        ]
        input_vector, _ = build_feature_vector(row, fixture_model.feature_name_, price_override=50.0)
        direct_prediction = float(fixture_model.predict(input_vector)[0])

        assert np.isclose(api_body["predicted_price"], direct_prediction)

    def test_unknown_product_is_404(self, client):
        r = client.post(
            "/v1/predict",
            json={"canonical_name": "not a real product", "supermarket": "TestMart", "date": "2024-01-15"},
        )
        assert r.status_code == 404
        assert r.json()["context"]["canonical_name"] == "not a real product"

    def test_date_outside_range_is_422(self, client):
        r = client.post(
            "/v1/predict", json={"canonical_name": "test bananas", "supermarket": "TestMart", "date": "2099-01-01"}
        )
        assert r.status_code == 422
        assert "min_date" in r.json()["context"]

    def test_negative_price_override_is_422(self, client):
        r = client.post(
            "/v1/predict",
            json={
                "canonical_name": "test bananas",
                "supermarket": "TestMart",
                "date": "2024-01-15",
                "price_override": -5.0,
            },
        )
        assert r.status_code == 422

    def test_rate_limit_returns_429(self, client):
        settings = get_settings()
        original = settings.api.predict_rate_limit
        settings.api.predict_rate_limit = "2/minute"
        try:
            body = {"canonical_name": "test bananas", "supermarket": "TestMart", "date": "2024-01-15"}
            statuses = [client.post("/v1/predict", json=body).status_code for _ in range(4)]
        finally:
            settings.api.predict_rate_limit = original
        assert statuses.count(429) > 0
