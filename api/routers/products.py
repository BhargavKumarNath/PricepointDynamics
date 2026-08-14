"""GET /v1/products/{canonical_id}/history."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Query

from api.dependencies import get_warehouse
from api.schemas.products import ProductHistoryResponse
from api.services.products import get_product_history
from pricepoint.warehouse import Warehouse

router = APIRouter(prefix="/v1/products", tags=["products"])


@router.get("/{canonical_id}/history", response_model=ProductHistoryResponse)
def history(
    canonical_id: str,
    supermarket: Annotated[str | None, Query(description="Restrict history to one retailer.")] = None,
    warehouse: Warehouse = Depends(get_warehouse),
) -> ProductHistoryResponse:
    return get_product_history(warehouse, canonical_id, supermarket)
