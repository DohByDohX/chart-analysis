"""
Generate input/target chart image pairs for vision-to-vision training.

For each window JSON:
- Target: Full 128-candle chart (all candles visible)
- Input: First 123 candles rendered in 128-candle layout (last 5 positions empty)
"""
import sys
import json
import gc
from pathlib import Path
import logging
from tqdm import tqdm
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.chart_renderer import ChartRenderer
from config import (
    WINDOW_SIZE, CONTEXT_CANDLES, IMAGE_SIZE,
    INPUT_IMAGES_DIR, TARGET_IMAGES_DIR, DATA_DIR
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

BATCH_SIZE = 50  # Process in batches to manage memory


def load_window_df(window_path: Path) -> pd.DataFrame:
    """Load a window JSON and return the input_window as a DataFrame."""
    with open(window_path, 'r') as f:
        data = json.load(f)
    return pd.DataFrame(data['input_window'])


def main():
    logger.info("=" * 70)
    logger.info("Generating Input/Target Chart Image Pairs")
    logger.info("=" * 70)
    logger.info(f"Window size: {WINDOW_SIZE} candles")
    logger.info(f"Context candles: {CONTEXT_CANDLES}")
    logger.info(f"Masked candles: {WINDOW_SIZE - CONTEXT_CANDLES}")
    logger.info(f"Image size: {IMAGE_SIZE[0]}x{IMAGE_SIZE[1]}")
    logger.info(f"Pixels per candle: {IMAGE_SIZE[0] / WINDOW_SIZE}")

    # Directories
    windows_dir = DATA_DIR / "processed" / "windows"
    INPUT_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    TARGET_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    # Find all window JSON files
    window_files = sorted(windows_dir.glob("window_*.json"))
    logger.info(f"Found {len(window_files)} window files")

    if not window_files:
        logger.error("No window files found!")
        return 1

    # Initialize renderer
    renderer = ChartRenderer(output_size=IMAGE_SIZE)

    # Process in batches
    total_rendered = 0
    for batch_start in range(0, len(window_files), BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, len(window_files))
        batch_num = batch_start // BATCH_SIZE + 1
        total_batches = (len(window_files) - 1) // BATCH_SIZE + 1

        logger.info(f"Batch {batch_num}/{total_batches}")

        for window_path in tqdm(
            window_files[batch_start:batch_end],
            desc=f"Batch {batch_num}",
            leave=False
        ):
            # Extract window ID from filename (e.g., "window_00042.json" -> "00042")
            window_id = window_path.stem.replace("window_", "")

            # Load full 128-candle window
            full_df = load_window_df(window_path)

            # Render target: all 128 candles
            target_image = renderer.render_window(full_df, total_candles=WINDOW_SIZE)
            target_path = TARGET_IMAGES_DIR / f"target_{window_id}.png"
            renderer.save_image(target_image, target_path)

            # Render input: first 123 candles in 128-candle layout
            context_df = full_df.iloc[:CONTEXT_CANDLES]
            input_image = renderer.render_window(
                context_df, total_candles=WINDOW_SIZE
            )
            input_path = INPUT_IMAGES_DIR / f"input_{window_id}.png"
            renderer.save_image(input_image, input_path)

            total_rendered += 1

        # Free memory after each batch
        gc.collect()

    logger.info("=" * 70)
    logger.info(f"Done! Rendered {total_rendered} pairs")
    logger.info(f"Target images: {TARGET_IMAGES_DIR}")
    logger.info(f"Input images:  {INPUT_IMAGES_DIR}")
    logger.info("=" * 70)
    return 0


if __name__ == "__main__":
    exit(main())
