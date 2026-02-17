
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import random

# Use absolute paths
BASE_DIR = Path(r"d:\Projects\ChartAnalysis")
DATA_DIR = BASE_DIR / "data" / "processed" / "input_images"
TARGET_DIR = BASE_DIR / "data" / "processed" / "target_images"

print(f"Data Dir: {DATA_DIR}")
print(f"Target Dir: {TARGET_DIR}")

if not DATA_DIR.exists():
    print(f"ERROR: Data dir does not exist: {DATA_DIR}")
    sys.exit(1)

input_files = list(DATA_DIR.glob("input_*.png"))
print(f"Found {len(input_files)} input files.")

if len(input_files) == 0:
    print("No input files found. Exiting.")
    sys.exit(1)

# Inspect a few random files
random.shuffle(input_files)
subset = input_files[:5]

for p in subset:
    try:
        wid = p.stem.split('_')[1]
        target_path = TARGET_DIR / f"target_{wid}.png"
        
        print(f"Checking pair: {p.name} -> {target_path.name}")
        if not target_path.exists():
            print(f"  MISSING TARGET: {target_path}")
            continue
            
        img_in = Image.open(p).convert('RGB')
        img_tgt = Image.open(target_path).convert('RGB')
        
        arr_in = np.array(img_in)
        arr_tgt = np.array(img_tgt)
        
        print(f"  Input Shape: {arr_in.shape}, Min: {arr_in.min()}, Max: {arr_in.max()}, Mean: {arr_in.mean():.2f}")
        print(f"  Target Shape: {arr_tgt.shape}, Min: {arr_tgt.min()}, Max: {arr_tgt.max()}, Mean: {arr_tgt.mean():.2f}")
        
        # Check for solid colors (variance)
        if np.std(arr_in) < 1:
             print("  WARNING: Input image has extremely low variance (solid color)")
        if np.std(arr_tgt) < 1:
             print("  WARNING: Target image has extremely low variance (solid color)")
             
    except Exception as e:
        print(f"  Error processing {p.name}: {e}")
