"""
Test script for the ChartDataset.
Verifies RAM pre-loading, tensor shapes, and data normalization.
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import torch
import logging
import json
from src.data.dataset import ChartDataset
from config import PROCESSED_DATA_DIR

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    print("=" * 70)
    print("ChartDataset Test")
    print("=" * 70)
    
    # 1. Load splits
    splits_path = PROCESSED_DATA_DIR / "splits.json"
    if not splits_path.exists():
        logger.error(f"Splits file not found: {splits_path}")
        return 1
        
    with open(splits_path, 'r') as f:
        splits = json.load(f)
        
    # Use validation set (smaller)
    val_ids = splits.get("val", [])
    if not val_ids:
        logger.error("No validation IDs found")
        return 1
        
    print(f"Found {len(val_ids)} validation samples")
    
    # Limit to 10 samples for quick testing
    test_ids = val_ids[:10]
    print(f"Testing with {len(test_ids)} samples")
    
    # 2. Initialize Dataset (RAM Pre-load)
    print("\n" + "=" * 70)
    print("Test 1: Initialization & RAM Pre-load")
    print("=" * 70)
    
    try:
        dataset = ChartDataset(
            split_name="val_test",
            window_ids=test_ids,
            preload_ram=True
        )
        print("[OK] Dataset initialized with RAM pre-load")
    except Exception as e:
        print(f"[FAIL] Initialization failed: {e}")
        return 1
        
    # 3. Check Length
    print("\n" + "=" * 70)
    print("Test 2: Dataset Length")
    print("=" * 70)
    
    if len(dataset) == 10:
        print(f"[OK] Dataset length correct: {len(dataset)}")
    else:
        print(f"[FAIL] Expected 10, got {len(dataset)}")
        return 1
        
    # 4. Check Item Output
    print("\n" + "=" * 70)
    print("Test 3: Get Item (Shape & Range)")
    print("=" * 70)
    
    input_tensor, target_tensor = dataset[0]
    
    print(f"Input shape: {input_tensor.shape}")
    print(f"Target shape: {target_tensor.shape}")
    
    # Output should be (3, 512, 512)
    expected_shape = (3, 512, 512)
    if input_tensor.shape == expected_shape and target_tensor.shape == expected_shape:
        print(f"[OK] Shapes match expected {expected_shape}")
    else:
        print(f"[FAIL] Shape mismatch")
        return 1
        
    # Check Range (Should be [0, 1] for ToTensor)
    print(f"Input range: [{input_tensor.min():.4f}, {input_tensor.max():.4f}]")
    print(f"Target range: [{target_tensor.min():.4f}, {target_tensor.max():.4f}]")
    
    if input_tensor.min() >= 0 and input_tensor.max() <= 1.0:
        print("[OK] Values normalized to [0, 1]")
    else:
        print("[FAIL] Values out of range [0, 1]")
        return 1
        
    # 5. Check Performance (Memory Load)
    print("\n" + "=" * 70)
    print("Test 4: Loading Speed")
    print("=" * 70)
    
    # Iterate through all
    for i in range(len(dataset)):
        _ = dataset[i]
        
    print("[OK] Iteration successful")
    
    print("\n[SUCCESS] ChartDataset verified!")
    return 0

if __name__ == "__main__":
    exit(main())
