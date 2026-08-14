"""Contract tests for GET /v1/products/{canonical_id}/history.

The one endpoint promoted back from §25.2's superseded §8.1 list --
UI_refactor.md's Price Predictor redesign needed a real historical price
chart. Mirrors tests/api/test_predict.py's pattern: fixture data, real
app, dependency_overrides.
"""

from __future__ import annotations


class TestProductHistory:
    def test_returns_full_history_across_retailers(self, client):
        r = client.get("/v1/products/test bananas/history")
        assert r.status_code == 200
        body = r.json()
        assert body["canonical_name"] == "test bananas"
        assert len(body["history"]) == 2
        assert {point["supermarket"] for point in body["history"]} == {"TestMart", "OtherMart"}

    def test_filters_to_one_retailer(self, client):
        r = client.get("/v1/products/test bananas/history", params={"supermarket": "TestMart"})
        assert r.status_code == 200
        body = r.json()
        assert len(body["history"]) == 1
        assert body["history"][0]["supermarket"] == "TestMart"
        assert body["history"][0]["avg_price"] == 1.19

    def test_ordered_by_date(self, client):
        r = client.get("/v1/products/test bananas/history")
        dates = [point["date"] for point in r.json()["history"]]
        assert dates == sorted(dates)

    def test_unknown_product_is_404(self, client):
        r = client.get("/v1/products/not a real product/history")
        assert r.status_code == 404
        body = r.json()
        assert body["context"]["canonical_name"] == "not a real product"

    def test_known_product_unknown_retailer_is_404(self, client):
        r = client.get("/v1/products/test bananas/history", params={"supermarket": "NoSuchStore"})
        assert r.status_code == 404

    def test_missing_marts_return_structured_404_not_bare_500(self, client, tmp_path):
        from api.dependencies import get_warehouse
        from api.main import app
        from pricepoint.warehouse import Warehouse

        empty_dir = tmp_path / "no_marts_here"
        empty_dir.mkdir()

        def _empty_warehouse():
            warehouse = Warehouse(empty_dir)
            try:
                yield warehouse
            finally:
                warehouse.close()

        app.dependency_overrides[get_warehouse] = _empty_warehouse
        r = client.get("/v1/products/test bananas/history")

        assert r.status_code == 404
        body = r.json()
        assert "detail" in body
        assert "context" in body

    def test_bare_pricepoint_error_returns_structured_500_not_bare_stack_trace(self, client):
        """A `PricePointError` not caught by any of the more specific
        handlers (`ArtifactNotFoundError` -> 404, `DataValidationError` ->
        422) must still fall through to `_pricepoint_error_handler` in
        `api/main.py` and produce a structured 500 body, not an unhandled
        exception / bare stack trace leak (project_refactor.md §16). No
        existing route naturally raises a bare `PricePointError`, so this
        drives it via a dependency override, same technique as the 404
        case above."""
        from api.dependencies import get_warehouse
        from api.main import app
        from pricepoint.exceptions import PricePointError

        def _broken_warehouse():
            raise PricePointError("boom", context={"reason": "simulated failure"})
            yield  # pragma: no cover -- makes this a generator, never reached

        app.dependency_overrides[get_warehouse] = _broken_warehouse
        r = client.get("/v1/products/test bananas/history")

        assert r.status_code == 500
        body = r.json()
        assert body["detail"] == "boom"
        assert body["context"] == {"reason": "simulated failure"}
