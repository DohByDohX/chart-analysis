# TODO: Vision-to-Vision VisionTrader

## Phase 1: Data Preparation

- [ ] **Target Chart Renderer**: Modify chart renderer to produce target images (5 future candles) using same rendering style as input charts
- [ ] **Paired Dataset Generator**: Generate input/target image pairs from existing windows data
  - [ ] Input: 128-candle chart image (already exists for 5,000 windows)
  - [ ] Target: 5-candle chart image (new — render from existing window JSONs)
- [ ] **Validate Pairs**: Visual inspection of input/target pairs for consistency
- [ ] **Dataset Split**: Organize train/validation/test splits

## Phase 2: Model Architecture

- [ ] **Vision Encoder**: Set up ViT encoder for 512x512 input charts
  - [ ] Evaluate ViT-Base (384) vs ViT-Large (512) vs resize strategy
  - [ ] Freeze vs fine-tune decision
- [ ] **Image Decoder**: Build CNN-based decoder to generate future chart images
  - [ ] Design upsampling architecture (features → image)
  - [ ] Add residual blocks for detail refinement
- [ ] **End-to-End Model**: Combine encoder + decoder, verify forward pass

## Phase 3: Loss Functions & Training

- [ ] **Perceptual Loss**: Implement VGG-based feature loss
  - [ ] Extract features from multiple VGG layers
  - [ ] Weight different layers appropriately
- [ ] **SSIM Loss**: Implement structural similarity loss
- [ ] **Combined Loss**: Tune balance between perceptual and SSIM
- [ ] **Training Script**: Build training loop with:
  - [ ] Mixed precision (AMP)
  - [ ] Cosine annealing LR schedule
  - [ ] Early stopping
  - [ ] Checkpoint saving
  - [ ] Visualization of predictions during training
- [ ] **Train Baseline**: Train with perceptual + SSIM loss

## Phase 4: OHLCV Extraction Pipeline

- [ ] **Candle Detection**: Detect candlestick bodies and wicks from generated images
  - [ ] Color-based segmentation (green/red bodies)
  - [ ] Edge detection for wick lines
- [ ] **Price Calibration**: Map pixel Y-coordinates to price values
  - [ ] Use input chart's known range for calibration
- [ ] **Volume Extraction**: Detect and measure volume bars
- [ ] **Validation**: Ensure extracted OHLCV satisfies constraints (High >= Close/Open >= Low)
- [ ] **Test on Real Charts**: Verify extraction accuracy on actual rendered charts

## Phase 5: Evaluation & Refinement

- [ ] **Visual Quality Metrics**: SSIM, FID scores on test set
- [ ] **Prediction Accuracy**: MAPE, direction accuracy from extracted OHLCV
- [ ] **Human Evaluation**: Manual review of generated chart continuations
- [ ] **Error Analysis**: Identify failure modes (blurry outputs, impossible candles, etc.)
- [ ] **Optional: Adversarial Loss**: Add GAN discriminator for sharper outputs
- [ ] **Compare with Regression Baseline**: Quantitative comparison

## Phase 6: Inference & Visualization

- [ ] **Prediction Pipeline**: End-to-end: chart image → predicted future chart → extracted OHLCV
- [ ] **Attention Visualization**: Which parts of input chart influenced the prediction
- [ ] **Side-by-Side Comparison**: Show input chart + predicted vs actual future
- [ ] **Backtesting**: Compare predictions against historical outcomes