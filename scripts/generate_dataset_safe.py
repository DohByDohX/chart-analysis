"""
I/O-optimized dataset generation script with multiple safeguards.

Strategies to prevent I/O bottleneck crashes:
1. Tiny batches (10 instead of 50)
2. I/O delays between batches (2 seconds)
3. Aggressive matplotlib cleanup
4. Memory monitoring with auto-pause
5. One symbol at a time with full cleanup
"""
import sys
from pathlib import Path
import logging
from tqdm import tqdm
import time
import gc

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.window_generator import WindowGenerator
from src.data.chart_renderer import ChartRenderer
from src.data.tokenizer import CandleTokenizer
from config import SAMPLE_STOCKS, WINDOW_SIZE, DATA_DIR, SAMPLES_PER_STOCK

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_memory():
    """Check available memory."""
    try:
        import psutil
        mem = psutil.virtual_memory()
        available_gb = mem.available / 1e9
        percent_used = mem.percent
        return available_gb, percent_used
    except:
        return None, None


def main():
    logger.info("=" * 70)
    logger.info("I/O-Optimized Dataset Generation")
    logger.info("=" * 70)
    
    # Check initial memory
    avail_mem, mem_percent = check_memory()
    if avail_mem:
        logger.info(f"Initial Memory: {avail_mem:.1f} GB available ({mem_percent:.1f}% used)")
    
    # Directories
    raw_dir = DATA_DIR / "raw"
    windows_dir = DATA_DIR / "processed" / "windows"
    images_dir = DATA_DIR / "processed" / "images"
    tokenizer_dir = DATA_DIR / "processed" / "tokenizer"
    
    windows_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize components
    logger.info("Initializing components...")
    window_gen = WindowGenerator(data_dir=raw_dir, input_size=WINDOW_SIZE)
    tokenizer = CandleTokenizer()
    
    # Generate windows for each symbol
    total_windows = 0
    
    # CRITICAL: Process ONE symbol at a time with full cleanup
    for symbol_idx, symbol in enumerate(SAMPLE_STOCKS):
        logger.info(f"\n{'='*70}")
        logger.info(f"Processing Symbol {symbol_idx+1}/{len(SAMPLE_STOCKS)}: {symbol}")
        logger.info(f"{'='*70}")
        
        csv_path = raw_dir / f"{symbol}.csv"
        
        if not csv_path.exists():
            logger.warning(f"CSV not found for {symbol}, skipping...")
            continue
        
        # Create NEW renderer for each symbol to avoid memory accumulation
        renderer = ChartRenderer()
        
        # Load CSV data
        try:
            import pandas as pd
            import matplotlib
            matplotlib.use('Agg')  # Force non-interactive backend
            import matplotlib.pyplot as plt
            
            data = pd.read_csv(csv_path, index_col=0, parse_dates=True)
            
            # Use SAMPLES_PER_STOCK from config
            num_windows = SAMPLES_PER_STOCK
            logger.info(f"Generating {num_windows} windows for {symbol}...")
            
            windows = window_gen.generate_random_windows(
                data=data,
                num_samples=num_windows,
                symbol=symbol
            )
            logger.info(f"Generated {len(windows)} windows for {symbol}")
            
            # CRITICAL: TINY batches to reduce I/O pressure
            BATCH_SIZE = 10  # Reduced from 50 to 10
            IO_DELAY = 2.0   # 2 second delay between batches
            
            for batch_start in range(0, len(windows), BATCH_SIZE):
                batch_end = min(batch_start + BATCH_SIZE, len(windows))
                batch_num = batch_start // BATCH_SIZE + 1
                total_batches = (len(windows) - 1) // BATCH_SIZE + 1
                
                logger.info(f"  Batch {batch_num}/{total_batches} (windows {batch_start}-{batch_end-1})")
                
                for i in range(batch_start, batch_end):
                    window_data = windows[i]
                    window_id = f"{total_windows + i:05d}"
                
                    # Convert DataFrames to dict for JSON serialization
                    window_to_save = {
                        'symbol': window_data['symbol'],
                        'start_date': window_data['start_date'],
                        'end_date': window_data['end_date'],
                        'input_window': window_data['input'].to_dict(orient='list'),
                        'target_window': window_data['target'].to_dict(orient='list')
                    }
                    
                    # Save window data to JSON
                    import json
                    window_path = windows_dir / f"window_{window_id}.json"
                    with open(window_path, 'w') as f:
                        json.dump(window_to_save, f)
                    
                    # Render and save image
                    image = renderer.render_window(window_data['input'])
                    image_path = images_dir / f"chart_{window_id}.png"
                    renderer.save_image(image, image_path)
                    
                    # Explicitly clear matplotlib after EACH image
                    plt.close('all')
                
                # CRITICAL: I/O delay to let storage controller recover
                if batch_num < total_batches:  # Don't delay after last batch
                    logger.info(f"  Pausing {IO_DELAY}s for I/O recovery...")
                    time.sleep(IO_DELAY)
                
                # Force aggressive garbage collection
                gc.collect()
                
                # Check memory status
                avail_mem, mem_percent = check_memory()
                if avail_mem:
                    logger.info(f"  Memory: {avail_mem:.1f} GB available ({mem_percent:.1f}% used)")
                    
                    # SAFETY: If memory is getting low, do extra cleanup
                    if mem_percent > 85:
                        logger.warning("High memory usage detected! Extra cleanup...")
                        plt.close('all')
                        gc.collect()
                        time.sleep(3)  # Extra delay
            
            total_windows += len(windows)
            
            # CRITICAL: Full cleanup after each symbol
            logger.info(f"Completed {symbol}. Performing full cleanup...")
            del renderer  # Delete renderer object
            del windows   # Free windows list
            del data      # Free CSV data
            plt.close('all')
            gc.collect()
            time.sleep(2)  # Brief pause between symbols
            
        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}")
            import traceback
            traceback.print_exc()
            
            # Cleanup even on error
            plt.close('all')
            gc.collect()
            continue
    
    logger.info(f"\nTotal windows generated: {total_windows}")
    
    # Save tokenizer vocabulary
    tokenizer_path = tokenizer_dir / "vocabulary.json"
    tokenizer.save_vocabulary(str(tokenizer_path))
    logger.info(f"Saved tokenizer to {tokenizer_path}")
    
    # Final memory check
    avail_mem, mem_percent = check_memory()
    if avail_mem:
        logger.info(f"Final Memory: {avail_mem:.1f} GB available ({mem_percent:.1f}% used)")
    
    logger.info("\n" + "=" * 70)
    logger.info("Dataset generation complete!")
    logger.info(f"Windows: {windows_dir}")
    logger.info(f"Images: {images_dir}")
    logger.info("=" * 70)

if __name__ == "__main__":
    main()
