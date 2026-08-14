"""Service function backing api/routers/products.py.

Thin wrapper around `Warehouse.get_product_history` (built and tested in
Phase 3, never previously wired to a live endpoint) -- promoted per
UI_refactor.md's decision to give the Price Predictor page a real
historical price chart, project_refactor.md §25.2's named "arbitrary
historical drill-down" trigger for this exact endpoint.
"""

from __future__ import annotations

from api.schemas.products import ProductHistoryPoint, ProductHistoryResponse
from pricepoint.exceptions import ArtifactNotFoundError
from pricepoint.warehouse import Warehouse


def get_product_history(warehouse: Warehouse, canonical_name: str, supermarket: str | None) -> ProductHistoryResponse:
    """Daily price history for one product, for charting -- backs
    `GET /v1/products/{canonical_id}/history`.

    Raises
    ------
    ArtifactNotFoundError
        If the product (optionally restricted to one retailer) has no
        history in the mart at all -- never a silently-empty 200.
    """
    df = warehouse.get_product_history(canonical_name, supermarket=supermarket)
    if df.empty:
        raise ArtifactNotFoundError(
            f"No price history found for {canonical_name!r}" + (f" at {supermarket!r}" if supermarket else "") + ".",
            context={"canonical_name": canonical_name, "supermarket": supermarket},
        )
    history = [ProductHistoryPoint(**row) for row in df.to_dict(orient="records")]
    return ProductHistoryResponse(canonical_name=canonical_name, history=history)
