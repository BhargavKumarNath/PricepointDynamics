#!/usr/bin/env python3
"""Download raw supermarket price data from Kaggle.

This script uses the Kaggle API to download this project's raw retailer
CSVs. Requires Kaggle API credentials
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def get_project_root() -> Path:
    """Locate the project root by finding pyproject.toml."""
    current = Path(__file__).resolve().parent.parent
    if (current / "pyproject.toml").exists():
        return current
    raise FileNotFoundError("Could not locate project root (pyproject.toml not found)")


def download_dataset(output_dir: Path, dataset: str) -> None:
    """Download the Kaggle dataset to the specified directory.

    Parameters
    ----------
    output_dir : Path
        Directory to download files to (will be created if needed).
    dataset : str
        Kaggle dataset identifier in format 'owner/dataset-name'.

    Raises
    ------
    RuntimeError
        If the Kaggle API call fails.
    """
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        raise ImportError("kaggle package not installed. Install it with: uv sync --extra kaggle") from None

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Connecting to Kaggle API …")
    api = KaggleApi()
    # authenticate() tries access token -> legacy key -> OAuth -> anonymous
    # (see this module's docstring); on total failure it prints its own
    # setup instructions and calls sys.exit(1) itself, so no separate
    # credential pre-check is maintained here.
    api.authenticate()
    logger.info("Kaggle API authenticated as %s. ✓", api.config_values.get("username", "?"))

    logger.info("Downloading dataset '%s' to %s …", dataset, output_dir)
    api.dataset_download_files(
        dataset,
        path=str(output_dir),
        unzip=False,  # We'll handle unzipping separately if needed
    )
    logger.info("Dataset download complete.")


def extract_and_validate(output_dir: Path) -> None:
    """Extract any .zip files and validate expected CSV files exist.

    Parameters
    ----------
    output_dir : Path
        Directory containing the downloaded files.
    """
    import zipfile

    output_dir = Path(output_dir)

    # Unzip any .zip files
    for zip_path in output_dir.glob("*.zip"):
        logger.info("Extracting %s …", zip_path.name)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(output_dir)
        zip_path.unlink()  # Remove the zip after extraction
        logger.info("Extraction complete.")

    # Validate expected files
    csv_files = list(output_dir.glob("*.csv"))
    logger.info("Found %d CSV files in %s", len(csv_files), output_dir)

    if not csv_files:
        raise RuntimeError(f"No CSV files found in {output_dir} after extraction")

    for csv_path in csv_files:
        logger.info("  - %s (%s rows)", csv_path.name, _count_rows(csv_path))

    logger.info("Validation complete. Data is ready for ingestion.")


def _count_rows(csv_path: Path) -> int:
    """Count rows in a CSV file (excluding header)."""
    try:
        with open(csv_path, encoding="utf-8") as f:
            return sum(1 for _ in f) - 1
    except Exception:
        return -1


_VERIFIED_DEFAULT_DATASET = "declanmcalinden/time-series-uk-supermarket-data"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download this project's raw retailer CSVs from Kaggle.",
    )
    parser.add_argument(
        "--dataset",
        default=os.getenv("KAGGLE_DATASET_SLUG", _VERIFIED_DEFAULT_DATASET),
        help="Kaggle dataset identifier, 'owner/dataset-name' "
        f"(or set KAGGLE_DATASET_SLUG). Default: {_VERIFIED_DEFAULT_DATASET} "
        "-- see this script's module docstring for how that default was verified.",
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = _parse_args()

    project_root = get_project_root()
    output_dir = project_root / "data" / "00_raw"

    logger.info("PricePoint Dynamics — Kaggle Dataset Downloader")
    logger.info("Project root: %s", project_root)
    logger.info("Output directory: %s", output_dir)
    logger.info("Dataset: %s", args.dataset)

    try:
        download_dataset(output_dir, dataset=args.dataset)
        extract_and_validate(output_dir)
        logger.info("✓ All data downloaded and ready for ingestion.")
    except (FileNotFoundError, RuntimeError, ImportError) as e:
        logger.error("Failed: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
