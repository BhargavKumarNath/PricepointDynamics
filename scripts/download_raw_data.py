#!/usr/bin/env python3
"""Download raw supermarket price data from Kaggle.

This script uses the Kaggle API to download the dataset containing raw
supermarket price CSVs. Requires a valid Kaggle API token configured at
~/.kaggle/kaggle.json.

Setup:
    1. Install kaggle: pip install kaggle
    2. Get an API key from https://www.kaggle.com/account
    3. Save it to ~/.kaggle/kaggle.json
    4. Ensure permissions: chmod 600 ~/.kaggle/kaggle.json
    5. Run: python scripts/download_raw_data.py

Environment variable override:
    KAGGLE_USERNAME and KAGGLE_KEY can be set instead of using the config file.
"""

from __future__ import annotations

import json
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


def setup_kaggle_api() -> None:
    """Verify Kaggle API credentials are available.

    Raises
    ------
    FileNotFoundError
        If neither env vars nor ~/.kaggle/kaggle.json are configured.
    RuntimeError
        If the API token is invalid.
    """
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_json = kaggle_dir / "kaggle.json"

    # Check env vars first (higher priority)
    if os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY"):
        logger.info("Using Kaggle credentials from environment variables.")
        return

    # Check config file
    if not kaggle_json.exists():
        raise FileNotFoundError(
            f"Kaggle API credentials not found at {kaggle_json}.\n"
            "Setup instructions:\n"
            "  1. Go to https://www.kaggle.com/account\n"
            "  2. Click 'Create new token' to download kaggle.json\n"
            "  3. Place it at ~/.kaggle/kaggle.json\n"
            "  4. Run: chmod 600 ~/.kaggle/kaggle.json"
        )

    logger.info("Using Kaggle credentials from %s", kaggle_json)

    # Verify the JSON is valid
    try:
        with open(kaggle_json) as f:
            creds = json.load(f)
            if not all(k in creds for k in ["username", "key"]):
                raise ValueError("Missing 'username' or 'key' in kaggle.json")
    except (json.JSONDecodeError, ValueError) as e:
        raise RuntimeError(f"Invalid Kaggle credentials file: {e}") from e


def download_dataset(output_dir: Path, dataset: str = "bhargavkumarnath/uk-supermarket-prices") -> None:
    """Download the Kaggle dataset to the specified directory.

    Parameters
    ----------
    output_dir : Path
        Directory to download files to (will be created if needed).
    dataset : str
        Kaggle dataset identifier in format 'username/dataset-name'.

    Raises
    ------
    RuntimeError
        If the Kaggle API call fails.
    """
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        raise ImportError("kaggle package not installed. Install it with: pip install kaggle") from None

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Connecting to Kaggle API …")
    api = KaggleApi()
    api.authenticate()
    logger.info("Kaggle API authenticated. ✓")

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


def main() -> None:
    """Main entry point."""
    project_root = get_project_root()
    output_dir = project_root / "data" / "00_raw"

    logger.info("PricePoint Dynamics — Kaggle Dataset Downloader")
    logger.info("Project root: %s", project_root)
    logger.info("Output directory: %s", output_dir)

    try:
        setup_kaggle_api()
        download_dataset(output_dir)
        extract_and_validate(output_dir)
        logger.info("✓ All data downloaded and ready for ingestion.")
    except (FileNotFoundError, RuntimeError, ImportError) as e:
        logger.error("Failed: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
