"""
Validate the raw CT Kidney dataset: corrupted images, formats, dimensions,
class imbalance. Writes a summary report.

Usage:
    python -m src.data.validate
"""
import json
from pathlib import Path
from collections import Counter, defaultdict

import yaml
from PIL import Image, UnidentifiedImageError

# pyrefly: ignore [missing-import]
from src.utils.logger import get_logger

logger = get_logger(__name__)

CONFIG_PATH = Path("config/config.yaml")
VALID_EXT = {".jpg", ".jpeg", ".png"}


def load_config() -> dict:
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def find_class_dirs(raw_dir: Path, expected_classes: list[str]) -> dict[str, Path]:
    """Same resolution logic as download.py — locate each class folder wherever nested."""
    import os
    found = {}
    expected_lower = {c.lower(): c for c in expected_classes}
    for root, dirs, _ in os.walk(raw_dir):
        for d in dirs:
            key = d.lower()
            if key in expected_lower:
                found[expected_lower[key]] = Path(root) / d
    return found


def validate_class(class_name: str, class_dir: Path) -> dict:
    result = {
        "class": class_name,
        "total_files": 0,
        "valid_images": 0,
        "corrupted": [],
        "unsupported_format": [],
        "duplicate_filenames": [],
        "dimensions": Counter(),
    }

    seen_names = set()
    all_files = list(class_dir.iterdir())
    result["total_files"] = len(all_files)

    for f in all_files:
        if f.suffix.lower() not in VALID_EXT:
            result["unsupported_format"].append(f.name)
            continue

        if f.name in seen_names:
            result["duplicate_filenames"].append(f.name)
        seen_names.add(f.name)

        try:
            with Image.open(f) as img:
                img.verify()
            # re-open after verify() (which invalidates the file handle) to read size
            with Image.open(f) as img:
                result["dimensions"][img.size] += 1
            result["valid_images"] += 1
        except (UnidentifiedImageError, OSError):
            result["corrupted"].append(f.name)

    # Counter isn't JSON-serializable directly
    result["dimensions"] = {f"{w}x{h}": c for (w, h), c in result["dimensions"].items()}
    return result


def check_class_imbalance(counts: dict[str, int]) -> dict:
    total = sum(counts.values())
    max_class = max(counts, key=counts.get)
    min_class = min(counts, key=counts.get)
    ratio = counts[max_class] / counts[min_class]

    imbalance_report = {
        "total_images": total,
        "class_percentages": {c: round(n / total * 100, 2) for c, n in counts.items()},
        "majority_class": max_class,
        "minority_class": min_class,
        "imbalance_ratio": round(ratio, 2),
    }

    if ratio > 3:
        logger.warning(
            f"Significant class imbalance detected: {max_class} is {ratio:.1f}x "
            f"larger than {min_class}. Will need class weighting during training."
        )

    return imbalance_report


def main():
    config = load_config()
    raw_dir = Path(config["data"]["raw_dir"])
    expected_classes = config["data"]["expected_classes"]

    class_dirs = find_class_dirs(raw_dir, expected_classes)
    if not class_dirs:
        logger.error("No class directories found. Run download.py first.")
        return

    per_class_results = {}
    counts = {}

    for class_name, class_dir in class_dirs.items():
        logger.info(f"Validating class: {class_name}")
        result = validate_class(class_name, class_dir)
        per_class_results[class_name] = result
        counts[class_name] = result["valid_images"]

        if result["corrupted"]:
            logger.warning(f"  {len(result['corrupted'])} corrupted files in {class_name}")
        if result["unsupported_format"]:
            logger.warning(
                f"  {len(result['unsupported_format'])} unsupported-format files in {class_name}"
            )
        if result["duplicate_filenames"]:
            logger.warning(
                f"  {len(result['duplicate_filenames'])} duplicate filenames in {class_name}"
            )
        logger.info(f"  {result['valid_images']} valid images, dimensions: {result['dimensions']}")

    imbalance = check_class_imbalance(counts)

    summary = {
        "per_class": per_class_results,
        "class_distribution": counts,
        "imbalance": imbalance,
    }

    out_path = Path("reports/metrics/dataset_summary.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Summary written to {out_path}")
    logger.info("Class distribution:")
    logger.info(f"{'Class':<12}{'Images'}")
    logger.info("-" * 20)
    for c, n in counts.items():
        logger.info(f"{c:<12}{n}")
    logger.info(f"Imbalance ratio (majority/minority): {imbalance['imbalance_ratio']}")

    total_issues = sum(
        len(r["corrupted"]) + len(r["unsupported_format"])
        for r in per_class_results.values()
    )
    if total_issues > 0:
        logger.warning(f"Validation finished with {total_issues} problem file(s) — review above.")
    else:
        logger.info("Validation passed with no corrupted or unsupported files.")


if __name__ == "__main__":
    main()