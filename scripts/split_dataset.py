"""
Split dataset into train/validation/test sets.

Creates a splits.json manifest mapping each split to its window IDs.
"""
import sys
import json
import random
from pathlib import Path
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config import DATA_DIR, RANDOM_SEED

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Split ratios
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1


def main():
    logger.info("=" * 70)
    logger.info("Splitting Dataset into Train/Val/Test")
    logger.info("=" * 70)

    windows_dir = DATA_DIR / "processed" / "windows"

    # Find all window IDs
    window_files = sorted(windows_dir.glob("window_*.json"))
    window_ids = [f.stem.replace("window_", "") for f in window_files]

    total = len(window_ids)
    logger.info(f"Found {total} windows")

    if total == 0:
        logger.error("No window files found!")
        return 1

    # Shuffle with fixed seed for reproducibility
    random.seed(RANDOM_SEED)
    random.shuffle(window_ids)

    # Split
    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)

    splits = {
        "train": sorted(window_ids[:train_end]),
        "val": sorted(window_ids[train_end:val_end]),
        "test": sorted(window_ids[val_end:]),
    }

    logger.info(f"Train: {len(splits['train'])} windows")
    logger.info(f"Val:   {len(splits['val'])} windows")
    logger.info(f"Test:  {len(splits['test'])} windows")

    # Save manifest
    output_path = DATA_DIR / "processed" / "splits.json"
    with open(output_path, 'w') as f:
        json.dump(splits, f, indent=2)

    logger.info(f"Saved splits to: {output_path}")

    logger.info("=" * 70)
    logger.info("Done!")
    logger.info("=" * 70)
    return 0


if __name__ == "__main__":
    exit(main())
