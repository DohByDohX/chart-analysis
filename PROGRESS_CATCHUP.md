# Progress Catchup: VisionTrader Project

This document provides a summary of the project state as of February 16, 2026, to facilitate easy resumption of work.

## Current Project State

### Core Architecture: Vision-to-Vision Regression
We have moved away from token-based prediction to a **Vision-to-Vision** approach. The model, `VisionTrader`, uses a pre-trained Vision Transformer (ViT) to extract features from 123 context candles and a CNN decoder to predict the remaining 5 masked candles.

- **Encoder**: `google/vit-base-patch32-384` (first 10 layers frozen).
- **Decoder**: 5-layer CNN upsampler (16x16 -> 512x512).
- **Stitching**: The model preserves the input history exactly (492px) and only predicts the future section (20px).
- **Loss**: Combined VGG Perceptual + SSIM loss with a 10x multiplier on the "future" region.

### Data Status
- **Dataset Size**: ~2,800 image pairs (standardized to 512x512 PNGs).
- **Split**: 85% Train / 15% Val (managed via `data/processed/splits.json`).
- **Formatting**: Input images have the last 5 candles masked (blacked out); target images contain the ground truth.

### Training Progress (Run: `version_02.16`)
- **Epochs**: 25 planned, 24 completed (stopped early at 15).
- **Final Metrics**: 
  - Train Loss: ~28.4
  - Validation Loss: ~25.5
- **Performance**: The model perfectly reconstructs chart structure and successfully predicts trend direction, though individual candle details (wicks) are still somewhat smoothed.

## How to Resume

### 1. Training
To continue training or start a new run:
```bash
python scripts/train.py --run-name your_run_name --epochs 25 --preload
```
Arguments:
- `--run-name`: Folder name for logs and checkpoints.
- `--epochs`: Number of training iterations.
- `--preload`: Loads images to RAM (faster, requires ~2GB RAM).
- `--resume`: Path to a `.pth` file to continue from.

### 2. Evaluation & Visualization
Check `data/logs/[run_name]` for `viz_epoch_N.png` files to see model predictions vs. targets.
Check `data/checkpoints/[run_name]/best_model.pth` for the weights with the lowest validation loss.

## Recommended Next Steps
1. **Increase Data**: 2,800 samples is a starting point; increasing to 10k-50k samples will likely improve candle detail.
2. **Decoder Upgrade**: Consider a Transformer-based decoder (e.g., using Cross-Attention with Encoder features) to improve high-frequency detail (wicks and shadows).
3. **Task Integration**: Re-examine the `gap-encoding` logic if returning to tokens for complex pattern recognition.

---
*Last Updated: 2026-02-16*
