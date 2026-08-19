"""
Download and validate the CT Kidney dataset from Kaggle.

Usage:
    python -m src.data.download
"""
import os
import sys
import zipfile
import shutil
from pathlib import Path
from collections import Counter
from dotenv import load_dotenv
load_dotenv()

import yaml

from src.utils.logger import get_logger

logger = get_logger(__name__)

KAGGLE_DATASET = "nazmul0087/ct-kidney-dataset-normal-cyst-tumor-and-stone"
CONFIG_PATH = Path("config/config.yaml")


def load_config() -> dict:
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def download_dataset(raw_dir: Path) -> Path:
    """Download and unzip the Kaggle dataset into raw_dir. Idempotent."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    zip_path = raw_dir / "ct-kidney-dataset.zip"

    marker = raw_dir / ".download_complete"
    if marker.exists():
        logger.info("Dataset already downloaded (marker found), skipping download.")
        return raw_dir

    logger.info(f"Downloading dataset '{KAGGLE_DATASET}' via Kaggle API...")
    exit_code = os.system(
        f'kaggle datasets download -d {KAGGLE_DATASET} -p "{raw_dir}"'
    )
    if exit_code != 0:
        logger.error("Kaggle download failed. Check kaggle.json credentials.")
        sys.exit(1)

    # kaggle CLI names the zip after the dataset slug
    downloaded_zip = raw_dir / "ct-kidney-dataset-normal-cyst-tumor-and-stone.zip"
    if not downloaded_zip.exists():
        # fall back: find any zip just downloaded
        zips = list(raw_dir.glob("*.zip"))
        if not zips:
            logger.error("No zip file found after download.")
            sys.exit(1)
        downloaded_zip = zips[0]

    logger.info(f"Extracting {downloaded_zip} ...")
    with zipfile.ZipFile(downloaded_zip, "r") as zf:
        zf.extractall(raw_dir)

    downloaded_zip.unlink()
    marker.touch()
    logger.info("Download and extraction complete.")
    return raw_dir


def inspect_structure(raw_dir: Path, max_depth: int = 3) -> None:
    """Print the actual folder tree so we can see real structure before assuming."""
    logger.info("Inspecting raw dataset structure:")
    for root, dirs, files in os.walk(raw_dir):
        depth = len(Path(root).relative_to(raw_dir).parts)
        if depth > max_depth:
            dirs[:] = []
            continue
        indent = "  " * depth
        logger.info(f"{indent}{Path(root).name}/  ({len(files)} files)")


def find_class_dirs(raw_dir: Path, expected_classes: list[str]) -> dict[str, Path]:
    """
    Locate the actual folder for each expected class, case-insensitively,
    wherever it is nested. Fails loudly if a class can't be found.
    """
    found = {}
    expected_lower = {c.lower(): c for c in expected_classes}

    for root, dirs, _ in os.walk(raw_dir):
        for d in dirs:
            key = d.lower()
            if key in expected_lower:
                canonical = expected_lower[key]
                found[canonical] = Path(root) / d

    missing = set(expected_classes) - set(found.keys())
    if missing:
        logger.error(f"Missing expected classes in dataset: {missing}")
        logger.error("Run inspect_structure() output above to see actual folder names.")
        sys.exit(1)

    return found


def report_class_distribution(class_dirs: dict[str, Path]) -> dict[str, int]:
    valid_ext = {".jpg", ".jpeg", ".png"}
    counts = Counter()
    for class_name, class_path in class_dirs.items():
        n = sum(1 for f in class_path.iterdir() if f.suffix.lower() in valid_ext)
        counts[class_name] = n

    logger.info("Class distribution:")
    logger.info(f"{'Class':<12}{'Images'}")
    logger.info("-" * 20)
    for class_name, count in counts.items():
        logger.info(f"{class_name:<12}{count}")

    return dict(counts)


def check_patient_identifiers(class_dirs: dict[str, Path]) -> None:
    """
    Sample filenames from each class and print them so we can visually check
    for patient-level identifiers (e.g. repeated patient IDs across files).
    This is a manual-inspection aid, not an automated guarantee.
    """
    logger.info("Sampling filenames to check for patient identifiers:")
    for class_name, class_path in class_dirs.items():
        files = list(class_path.iterdir())[:5]
        logger.info(f"  {class_name}: {[f.name for f in files]}")
    logger.info(
        "Review the filenames above manually. If they contain repeated patient "
        "IDs or scan-session prefixes, note this for Phase 5 (patient-level split). "
        "If filenames look like independent per-image IDs with no patient grouping, "
        "we'll split at image level and document that limitation."
    )


def main():
    config = load_config()
    raw_dir = Path(config["data"]["raw_dir"])
    expected_classes = config["data"]["expected_classes"]

    download_dataset(raw_dir)
    inspect_structure(raw_dir)

    class_dirs = find_class_dirs(raw_dir, expected_classes)
    logger.info(f"Resolved class directories: {class_dirs}")

    counts = report_class_distribution(class_dirs)
    check_patient_identifiers(class_dirs)

    total = sum(counts.values())
    logger.info(f"Total images: {total}")
    if total == 0:
        logger.error("No images found — dataset download/extraction likely failed.")
        sys.exit(1)

    logger.info("Dataset download and validation complete.")


if __name__ == "__main__":
    main()