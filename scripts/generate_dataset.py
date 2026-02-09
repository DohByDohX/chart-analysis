import sys
from pathlib import Path
import logging
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.window_generator import WindowGenerator
from src.data.chart_renderer import ChartRenderer
from src.data.tokenizer import CandleTokenizer
from config import SAMPLE_STOCKS, WINDOW_SIZE, DATA_DIR

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    logger.info("=" * 70)
    logger.info("Generating Training Dataset")
    logger.info("=" * 70)
    
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
    renderer = ChartRenderer()
    tokenizer = CandleTokenizer()
    
    # Generate windows for each symbol
    total_windows = 0
    
    for symbol in SAMPLE_STOCKS:
        logger.info(f"\nProcessing {symbol}...")
        csv_path = raw_dir / f"{symbol}.csv"
        
        if not csv_path.exists():
            logger.warning(f"CSV not found for {symbol}, skipping...")
            continue
        
        # Load CSV data
        try:
            import pandas as pd
            data = pd.read_csv(csv_path, index_col=0, parse_dates=True)
            
            # Generate random windows (e.g., 100 per symbol)
            num_windows = 100
            windows = window_gen.generate_random_windows(
                data=data,
                num_samples=num_windows,
                symbol=symbol
            )
            logger.info(f"Generated {len(windows)} windows for {symbol}")
            
            # Save windows and render images
            for i, window_data in enumerate(tqdm(windows, desc=f"Rendering {symbol}")):
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
                
                # Render and save image (use the input DataFrame)
                image = renderer.render_window(window_data['input'])
                image_path = images_dir / f"chart_{window_id}.png"
                renderer.save_image(image, image_path)
            
            total_windows += len(windows)
            
        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}")
            continue
    
    logger.info(f"\nTotal windows generated: {total_windows}")
    
    # Save tokenizer vocabulary
    tokenizer_path = tokenizer_dir / "vocabulary.json"
    tokenizer.save_vocabulary(str(tokenizer_path))
    logger.info(f"Saved tokenizer to {tokenizer_path}")
    
    logger.info("\n" + "=" * 70)
    logger.info("Dataset generation complete!")
    logger.info(f"Windows: {windows_dir}")
    logger.info(f"Images: {images_dir}")
    logger.info("=" * 70)

if __name__ == "__main__":
    main()
