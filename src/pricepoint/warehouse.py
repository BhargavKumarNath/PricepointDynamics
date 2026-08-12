"""Thin DuckDB query wrapper over the materialized analytics marts.

Every query here reads the `data/03_marts/*.parquet` files produced by
``scripts/materialize_marts.py`` directly (DuckDB's predicate/projection
pushdown over Parquet -- see project_refactor.md §6/§7), not the full
9.5M-row ``02_processed/`` files. This is what makes DuckDB genuinely
load-bearing rather than the previously-dead ``_get_duckdb_conn()`` stub
in ``dashboard/data_loader.py`` (defect #4).

Query functions here are grounded in the dashboard's actual current
pandas-based query patterns (Market Overview, Basket Analysis pages), not
speculative -- see project_refactor.md Phase 3's Progress Log entry for
the concrete evidence each one is based on.
"""

from __future__ import annotations

import logging
from pathlib import Path
from types import TracebackType

import duckdb
import pandas as pd

logger = logging.getLogger(__name__)


class MartNotFoundError(FileNotFoundError):
    """Raised when a required mart Parquet file doesn't exist on disk."""


class Warehouse:
    """Query interface over the materialized marts, backed by DuckDB.

    Opens one in-memory DuckDB connection per instance and reuses it
    across queries (a connection is cheap to create but each query still
    benefits from not re-establishing one). Marts are read directly from
    their Parquet files on every query -- DuckDB's projection/predicate
    pushdown means this is not equivalent to loading the file into
    memory first.

    Use as a context manager to ensure the connection is closed:

        with Warehouse(settings.marts.output_dir) as wh:
            overview = wh.get_market_overview()

    Parameters
    ----------
    marts_dir : Path
        Directory containing the materialized mart Parquet files
        (``dim_retailer.parquet``, ``dim_product.parquet``,
        ``fact_price_daily.parquet``).
    """

    def __init__(self, marts_dir: Path) -> None:
        self._marts_dir = Path(marts_dir)
        self._conn = duckdb.connect(database=":memory:")

    def close(self) -> None:
        """Close the underlying DuckDB connection."""
        self._conn.close()

    def __enter__(self) -> Warehouse:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.close()

    def _mart_path(self, name: str) -> Path:
        """Resolve and validate a mart's Parquet path.

        Raises
        ------
        MartNotFoundError
            If the mart hasn't been materialized yet.
        """
        path = self._marts_dir / f"{name}.parquet"
        if not path.exists():
            raise MartNotFoundError(
                f"Mart {name!r} not found at {path}. Run `python run.py marts` "
                "(or `uv run python scripts/materialize_marts.py`) first."
            )
        return path

    def search_products(self, query: str, limit: int = 20) -> pd.DataFrame:
        """Search canonical products by name, for the predictor's product selector.

        Grounds `GET /v1/products/search?q=` (project_refactor.md §8.1).

        Parameters
        ----------
        query : str
            Case-insensitive substring to match against `canonical_name`.
        limit : int
            Maximum number of results.

        Returns
        -------
        pd.DataFrame
            Columns: canonical_name, category, own_brand, n_retailers,
            first_seen_date, last_seen_date.
        """
        path = self._mart_path("dim_product")
        sql = """
            SELECT canonical_name, category, own_brand, n_retailers, first_seen_date, last_seen_date
            FROM read_parquet(?)
            WHERE canonical_name ILIKE '%' || ? || '%'
            ORDER BY canonical_name
            LIMIT ?
        """
        return self._conn.execute(sql, [str(path), query, limit]).fetchdf()

    def get_market_overview(self) -> pd.DataFrame:
        """Per-retailer summary: portfolio size, own-brand mix, price distribution.

        Grounds `GET /v1/market/overview` (project_refactor.md §8.1) --
        replaces `dashboard/pages/01_market_overview.py`'s
        ``df.groupby('supermarket')`` aggregates (portfolio size,
        own-brand %) computed after loading the full canonical dataset
        into pandas.

        ``own_brand_pct`` and the price-distribution columns are
        deliberately *row-weighted* over ``fact_price_daily`` (one row
        per day a product was observed), not de-duplicated to one row per
        product first -- this matches the existing dashboard's
        ``df.groupby('supermarket')['own_brand'].mean()`` over its
        full, row-level (not per-product) canonical data, where a
        product tracked across more days already counts more. This is a
        pre-existing statistical convention (arguably debatable, but not
        a bug introduced here) inherited intentionally rather than
        silently changed -- revisit only as a deliberate choice, not a
        side effect of building this mart.

        Returns
        -------
        pd.DataFrame
            One row per retailer: portfolio_size, own_brand_pct,
            min_price, price_p25, price_median, price_p75, max_price.
        """
        fact_path = self._mart_path("fact_price_daily")
        dim_path = self._mart_path("dim_product")
        sql = """
            SELECT
                f.supermarket,
                COUNT(DISTINCT f.canonical_name)                          AS portfolio_size,
                AVG(CASE WHEN d.own_brand THEN 1.0 ELSE 0.0 END) * 100     AS own_brand_pct,
                MIN(f.avg_price)::DOUBLE                                  AS min_price,
                quantile_cont(f.avg_price, 0.25)::DOUBLE                  AS price_p25,
                MEDIAN(f.avg_price)::DOUBLE                               AS price_median,
                quantile_cont(f.avg_price, 0.75)::DOUBLE                  AS price_p75,
                MAX(f.avg_price)::DOUBLE                                  AS max_price
            FROM read_parquet(?) f
            JOIN read_parquet(?) d ON f.canonical_name = d.canonical_name
            GROUP BY f.supermarket
            ORDER BY f.supermarket
        """
        return self._conn.execute(sql, [str(fact_path), str(dim_path)]).fetchdf()

    def get_basket_cost(self, canonical_names: list[str]) -> pd.DataFrame:
        """Latest-date price + coverage per retailer for a set of products.

        Grounds `GET /v1/market/basket?basket=` (project_refactor.md
        §8.1) -- replaces `dashboard/pages/02_Basket_analysis.py`'s
        pivot-table-and-sum pattern over the full canonical dataset,
        loaded once and cached in Streamlit's process memory.

        Parameters
        ----------
        canonical_names : list[str]
            Canonical product names making up the basket. Products not
            present in the mart (e.g. never matched, or discontinued)
            are silently excluded from the result, not zero-filled --
            callers should compare `items_found` per retailer against
            `len(canonical_names)` to detect partial coverage
            (project_refactor.md §8.2's "never zero-fill silently"
            principle applies here too).

        Returns
        -------
        pd.DataFrame
            One row per retailer that stocks at least one basket item:
            supermarket, basket_cost, items_found.
        """
        if not canonical_names:
            return pd.DataFrame(columns=["supermarket", "basket_cost", "items_found"])

        path = self._mart_path("fact_price_daily")
        sql = """
            WITH latest AS (
                SELECT MAX(date) AS max_date FROM read_parquet(?)
            )
            SELECT
                f.supermarket,
                SUM(f.avg_price)::DOUBLE AS basket_cost,
                COUNT(*)                 AS items_found
            FROM read_parquet(?) f, latest
            WHERE f.date = latest.max_date
              AND f.canonical_name IN (SELECT UNNEST(?::VARCHAR[]))
            GROUP BY f.supermarket
            ORDER BY f.supermarket
        """
        return self._conn.execute(sql, [str(path), str(path), canonical_names]).fetchdf()
