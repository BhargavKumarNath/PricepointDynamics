"""Build the analytics marts (data/03_marts/) from 02_processed/ via DuckDB.

This is the "write" counterpart to ``warehouse.py``'s "read" side: this
module builds the mart Parquet files from ``sql/marts/*.sql``;
``warehouse.py`` queries them. Kept as two modules rather than one,
following the same separation this project already uses elsewhere
(``data_ingestion.py`` builds, nothing else in ``src/pricepoint`` re-reads
its output directly except the next pipeline stage) -- a single
`Warehouse` class doing both materialization and querying would mix two
different lifecycles (build-once-per-refresh vs. query-per-request) in
one interface.

Reuses this project's existing manifest-based skip-if-unchanged pattern
(``manifest.py``, first wired up in Phase 1): each mart is only rebuilt
if its source Parquet file has changed since that mart's manifest was
last written, unless ``force=True``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import duckdb
import pandas as pd

from pricepoint.config import Settings
from pricepoint.manifest import has_sources_changed, write_manifest

logger = logging.getLogger(__name__)

# src/pricepoint/marts.py -> src/pricepoint -> src -> <project root> -> sql/marts
SQL_DIR = Path(__file__).resolve().parents[2] / "sql" / "marts"

# Materialization order matters only in that dim_retailer/dim_product are
# cheap sanity-checks before the larger fact_price_daily build -- there is
# no foreign-key enforcement between the mart Parquet files (DuckDB reads
# them independently; referential integrity is a query-time join
# assumption, not a stored constraint), so order does not affect
# correctness.
MART_NAMES = ["dim_retailer", "dim_product", "fact_price_daily"]


def materialize_mart(
    name: str,
    source_path: Path,
    output_dir: Path,
    conn: duckdb.DuckDBPyConnection,
    force: bool = False,
) -> Path:
    """Build one mart from its SQL definition and write it to Parquet.

    Parameters
    ----------
    name : str
        Mart name; must match a `sql/marts/{name}.sql` file.
    source_path : Path
        Path to the source Parquet file (canonical_products_e5.parquet).
    output_dir : Path
        Directory to write `{name}.parquet` into.
    conn : duckdb.DuckDBPyConnection
        Shared DuckDB connection.
    force : bool
        If True, rebuild even if the source is unchanged since the last
        recorded manifest.

    Returns
    -------
    Path
        Path to the materialized mart Parquet file.
    """
    sql_path = SQL_DIR / f"{name}.sql"
    if not sql_path.exists():
        raise FileNotFoundError(f"No SQL definition found for mart {name!r} at {sql_path}")

    output_path = output_dir / f"{name}.parquet"
    manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")

    if not force and output_path.exists() and not has_sources_changed([source_path], manifest_path):
        logger.info("Mart '%s' unchanged since last run; skipping. Output: %s", name, output_path)
        return output_path

    query = sql_path.read_text(encoding="utf-8").rstrip().rstrip(";")

    logger.info("Materializing mart '%s' from %s …", name, sql_path.name)
    # COPY streams DuckDB's query result directly to Parquet -- the mart
    # (at most ~68K rows for dim_product, far smaller for the others) is
    # never pulled into a Python-side pandas DataFrame during the build
    # itself, even though the underlying scan is over 9.5M source rows.
    conn.execute(f"COPY ({query}) TO '{output_path}' (FORMAT PARQUET)", [str(source_path)])

    # Read the (small) mart back to generate its manifest -- consistent
    # with every other pipeline stage's write_manifest() call, and cheap
    # here specifically because marts are the whole point of being small.
    mart_df = pd.read_parquet(output_path, engine="pyarrow")
    write_manifest(output_path, mart_df, [source_path], stage=f"mart:{name}")

    logger.info("Mart '%s' materialized: %s rows -> %s", name, f"{len(mart_df):,}", output_path)
    return output_path


def run_materialize_marts(settings: Settings, force: bool = False) -> dict[str, Path]:
    """Materialize all 3 marts from `02_processed/`.

    Parameters
    ----------
    settings : Settings
        Application settings.
    force : bool
        If True, rebuild every mart even if unchanged.

    Returns
    -------
    dict[str, Path]
        Mapping of mart name to its output Parquet path.
    """
    source_path = settings.data.processed_dir / settings.marts.source_filename
    if not source_path.exists():
        raise FileNotFoundError(f"Canonical products not found at {source_path}. Run matching first.")

    output_dir = settings.marts.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    conn = duckdb.connect(database=":memory:")
    try:
        return {name: materialize_mart(name, source_path, output_dir, conn, force=force) for name in MART_NAMES}
    finally:
        conn.close()
