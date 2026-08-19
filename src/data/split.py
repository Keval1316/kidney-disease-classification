"""
Create stratified train/val/test splits at image level.
No patient identifiers exist in this dataset (confirmed in Phase 3), so
splitting is done per-image. This is a documented limitation: if two visually
similar slices from the same original scan session ended up in different
splits, that could inflate apparent performance slightly. See README Limitations.

Usage:
    python -m src.data.split
"""
import json
import random
from pathlib import Path

import yaml

from src.utils.logger import get_logger

logger = get_logger(__name__)

CONFIG_PATH = Path("config/config.yaml")
VALID_EXT = {".jpg", ".jpeg", ".png"}


def load_config() -> dict:
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def find_class_dirs(raw_dir: Path, expected_classes: list[str]) -> dict[str, Path]:
    import os
    found = {}
    expected_lower = {c.lower(): c for c in expected_classes}
    for root, dirs, _ in os.walk(raw_dir):
        for d in dirs:
            key = d.lower()
            if key in expected_lower:
                found[expected_lower[key]] = Path(root) / d
    return found


def stratified_split(
    class_dirs: dict[str, Path],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict:
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"

    rng = random.Random(seed)
    splits = {"train": [], "val": [], "test": []}

    for class_name, class_dir in class_dirs.items():
        files = sorted(
            str(f) for f in class_dir.iterdir() if f.suffix.lower() in VALID_EXT
        )
        rng.shuffle(files)

        n = len(files)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        # remainder goes to test, avoids rounding losing images

        train_files = files[:n_train]
        val_files = files[n_train:n_train + n_val]
        test_files = files[n_train + n_val:]

        splits["train"].extend([(f, class_name) for f in train_files])
        splits["val"].extend([(f, class_name) for f in val_files])
        splits["test"].extend([(f, class_name) for f in test_files])

        logger.info(
            f"{class_name}: total={n}, train={len(train_files)}, "
            f"val={len(val_files)}, test={len(test_files)}"
        )

    # shuffle each split so classes are interleaved, not blocked
    for key in splits:
        rng.shuffle(splits[key])

    return splits


def verify_no_leakage(splits: dict) -> None:
    train_set = {f for f, _ in splits["train"]}
    val_set = {f for f, _ in splits["val"]}
    test_set = {f for f, _ in splits["test"]}

    overlap_tv = train_set & val_set
    overlap_tt = train_set & test_set
    overlap_vt = val_set & test_set

    if overlap_tv or overlap_tt or overlap_vt:
        logger.error(
            f"LEAKAGE DETECTED: train/val={len(overlap_tv)}, "
            f"train/test={len(overlap_tt)}, val/test={len(overlap_vt)}"
        )
        raise RuntimeError("Data leakage between splits — aborting.")
    logger.info("No leakage between splits — verified disjoint file sets.")


def main():
    config = load_config()
    raw_dir = Path(config["data"]["raw_dir"])
    expected_classes = config["data"]["expected_classes"]
    seed = config["project"]["seed"]
    train_ratio = config["split"]["train_ratio"]
    val_ratio = config["split"]["val_ratio"]
    test_ratio = config["split"]["test_ratio"]

    class_dirs = find_class_dirs(raw_dir, expected_classes)
    if not class_dirs:
        logger.error("No class directories found. Run download.py first.")
        return

    splits = stratified_split(class_dirs, train_ratio, val_ratio, test_ratio, seed)
    verify_no_leakage(splits)

    out_path = Path("data/processed/splits.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(splits, f, indent=2)

    logger.info(f"Splits written to {out_path}")
    logger.info(
        f"Totals — train: {len(splits['train'])}, "
        f"val: {len(splits['val'])}, test: {len(splits['test'])}"
    )


if __name__ == "__main__":
    main()