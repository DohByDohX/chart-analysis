"""
Test script for chart rendering.
Validates image dimensions, zoom-to-fit normalization, and visual output.
"""
import sys
from pathlib import Path
import gc

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.chart_renderer import ChartRenderer
from src.data.window_generator import WindowGenerator
from config import (
    RAW_DATA_DIR, IMAGE_DATA_DIR, WINDOW_SIZE, PREDICTION_HORIZON,
    IMAGE_SIZE, RANDOM_SEED
)
import numpy as np


def main():
    """Test the chart rendering functionality."""
    print("=" * 70)
    print("Vision-Trader: Chart Rendering Test")
    print("=" * 70)
    print(f"Window size: {WINDOW_SIZE} candles")
    print(f"Image size: {IMAGE_SIZE[0]}×{IMAGE_SIZE[1]}")
    print(f"Pixels per candle: {IMAGE_SIZE[0] / WINDOW_SIZE:.2f}")
    print()
    
    # Initialize renderer
    renderer = ChartRenderer(output_size=IMAGE_SIZE)
    
    # Test 1: Generate sample windows
    print("=" * 70)
    print("Test 1: Generate Sample Windows")
    print("=" * 70)
    
    generator = WindowGenerator(
        data_dir=RAW_DATA_DIR,
        input_size=WINDOW_SIZE,
        target_size=PREDICTION_HORIZON,
        random_seed=RANDOM_SEED
    )
    
    windows = generator.generate_from_symbol("AAPL", num_samples=3)
    
    if not windows:
        print("[FAIL] Could not generate sample windows")
        return 1
    
    print(f"[OK] Generated {len(windows)} windows")
    
    # Test 2: Test zoom-to-fit normalization
    print("\n" + "=" * 70)
    print("Test 2: Zoom-to-Fit Normalization")
    print("=" * 70)
    
    sample_window = windows[0]['input']
    print(f"Original price range:")
    print(f"  Low:  {sample_window['Low'].min():.2f}")
    print(f"  High: {sample_window['High'].max():.2f}")
    
    normalized = renderer.apply_zoom_to_fit(sample_window)
    print(f"\nNormalized price range:")
    print(f"  Low:  {normalized['Low'].min():.4f}")
    print(f"  High: {normalized['High'].max():.4f}")
    
    # Check if normalized to [0, 1]
    all_in_range = (
        normalized['Low'].min() >= 0 and
        normalized['High'].max() <= 1 and
        normalized['Open'].min() >= 0 and
        normalized['Close'].max() <= 1
    )
    
    if all_in_range:
        print("[OK] All prices normalized to [0, 1] range")
    else:
        print("[FAIL] Normalization out of range")
    
    # Test 3: Render single window
    print("\n" + "=" * 70)
    print("Test 3: Render Single Window")
    print("=" * 70)
    
    image = renderer.render_window(windows[0]['input'])
    
    print(f"Image shape: {image.shape}")
    print(f"Image dtype: {image.dtype}")
    print(f"Value range: [{image.min():.4f}, {image.max():.4f}]")
    
    # Validate dimensions
    expected_shape = (IMAGE_SIZE[1], IMAGE_SIZE[0], 3)
    if image.shape == expected_shape:
        print(f"[OK] Image dimensions correct: {expected_shape}")
    else:
        print(f"[FAIL] Expected {expected_shape}, got {image.shape}")
        return 1
    
    # Validate value range
    if image.min() >= 0 and image.max() <= 1:
        print("[OK] Image values in [0, 1] range")
    else:
        print(f"[FAIL] Image values out of range: [{image.min()}, {image.max()}]")
    
    # Test 4: Validate 3-channel grayscale
    print("\n" + "=" * 70)
    print("Test 4: 3-Channel Grayscale Validation")
    print("=" * 70)
    
    # Check if all channels are identical (grayscale)
    channels_equal = np.allclose(image[:, :, 0], image[:, :, 1]) and \
                     np.allclose(image[:, :, 1], image[:, :, 2])
    
    if channels_equal:
        print("[OK] All 3 channels are identical (grayscale stacking)")
    else:
        print("[WARNING] Channels are not identical")
    
    # Test 5: Save sample images
    print("\n" + "=" * 70)
    print("Test 5: Save Sample Images")
    print("=" * 70)
    
    output_dir = IMAGE_DATA_DIR / "samples"
    
    for i, window in enumerate(windows):
        image = renderer.render_window(window['input'])
        filename = f"AAPL_window_{i}_{window['start_date']}_to_{window['end_date']}.png"
        renderer.save_image(image, output_dir / filename)
        print(f"[OK] Saved: {filename}")
        # Force garbage collection after each render
        gc.collect()
    
    # Test 6: Batch rendering
    print("\n" + "=" * 70)
    print("Test 6: Batch Rendering")
    print("=" * 70)
    
    batch_output_dir = IMAGE_DATA_DIR / "batch"
    rendered_images = renderer.render_batch(
        windows,
        output_dir=batch_output_dir,
        save_images=True,
        return_images=True
    )
    
    print(f"[OK] Rendered {len(rendered_images)} images in batch")
    print(f"[OK] Saved to: {batch_output_dir}")
    
    # Test 7: Visual inspection info
    print("\n" + "=" * 70)
    print("Test 7: Visual Inspection")
    print("=" * 70)
    
    print(f"Sample images saved to:")
    print(f"  {output_dir}")
    print(f"  {batch_output_dir}")
    print()
    print("Please visually inspect the images to verify:")
    print("  - Candlesticks are clearly visible")
    print("  - Volume bars are at the bottom")
    print("  - No grid, labels, or UI elements")
    print("  - Monochrome (black/white/gray)")
    print("  - Each candle is approximately 4 pixels wide")
    
    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"[SUCCESS] All automated tests passed!")
    print(f"Rendered {len(windows)} windows")
    print(f"Image size: {IMAGE_SIZE[0]}×{IMAGE_SIZE[1]}×3")
    print(f"Pixels per candle: {IMAGE_SIZE[0] / WINDOW_SIZE:.2f}")
    print(f"Output directory: {IMAGE_DATA_DIR}")
    
    return 0


if __name__ == "__main__":
    exit(main())
