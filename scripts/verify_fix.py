import sys
from pathlib import Path
import shutil

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.chart_renderer import ChartRenderer
from src.data.window_generator import WindowGenerator
from config import RAW_DATA_DIR, IMAGE_SIZE, WINDOW_SIZE, PREDICTION_HORIZON, RANDOM_SEED

def test_memory_fix():
    print("Testing memory fix...")
    renderer = ChartRenderer(output_size=IMAGE_SIZE)
    generator = WindowGenerator(
        data_dir=RAW_DATA_DIR,
        input_size=WINDOW_SIZE,
        target_size=PREDICTION_HORIZON,
        random_seed=RANDOM_SEED
    )
    windows = generator.generate_from_symbol("AAPL", num_samples=5)
    
    output_dir = Path("temp_test_images")
    output_dir.mkdir(exist_ok=True)
    
    try:
        # Test with return_images=False (default)
        print("Running render_batch with return_images=False...")
        images = renderer.render_batch(windows, output_dir=output_dir, save_images=False, return_images=False)
        
        if len(images) == 0:
            print("[PASS] render_batch returned empty list as expected")
        else:
            print(f"[FAIL] render_batch returned {len(images)} images when return_images=False")
            return 1
            
        # Test with return_images=True
        print("Running render_batch with return_images=True...")
        images = renderer.render_batch(windows, output_dir=output_dir, save_images=False, return_images=True)
        
        if len(images) == 5:
            print(f"[PASS] render_batch returned {len(images)} images as expected")
        else:
            print(f"[FAIL] render_batch returned {len(images)} images when return_images=True")
            return 1
            
        print("Memory fix verification complete!")
        return 0
        
    finally:
        # Cleanup
        if output_dir.exists():
            shutil.rmtree(output_dir)

if __name__ == "__main__":
    exit(test_memory_fix())
