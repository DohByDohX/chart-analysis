# PRD: Vision-Trader

## Pure Visual Chart Prediction Using Vision-to-Vision Learning

### 1. Product Vision

Build an AI system that predicts future stock price action the way a human trader does — by looking at charts. The model takes a candlestick chart image as input and generates a chart image of predicted future candles as output. No numerical targets, no tokenization — **pure visual reasoning**.

### 2. Core Philosophy

* **Learn like humans:** Traders read charts visually. They see patterns (double tops, head & shoulders, breakouts), not numbers. Our model should do the same.
* **No numerical supervision:** The model never sees raw OHLCV numbers during training. It learns entirely from image pairs.
* **Numbers are post-processing:** OHLCV values are extracted from generated chart images only when needed for trade execution — just like a human reading a chart.

### 3. System Architecture

```
Input: Historical Chart Image (128 candles, 512x512 PNG)
    |
    v
Vision Encoder (ViT) — understands patterns
    |
    v
Image Decoder (CNN upsampler) — imagines the future
    |
    v
Output: Predicted Future Chart Image (5 candles)
    |
    v (post-processing, only when needed)
OHLCV Extraction Pipeline — reads the chart
    |
    v
[Open, High, Low, Close, Volume] x 5 candles
```

### 4. Data Specification

| Attribute | Value |
| --- | --- |
| **Input** | 512x512 monochrome candlestick chart (128 candles + volume bars) |
| **Output** | Chart image showing 5 predicted future candles |
| **Preprocessing** | Zoom-to-fit normalization per window; no indicators; no grid |
| **Data Sources** | S&P 500 + NASDAQ historical daily OHLCV (25 tickers, diverse sectors) |
| **Dataset Size** | 5,000 input/target image pairs (expandable) |
| **Rendering** | Custom PIL-based renderer; consistent style between input and target |

### 5. Model Components

| Component | Role |
| --- | --- |
| **Vision Encoder** | Pretrained ViT (e.g., ViT-Base or ViT-Large). Extracts visual features from input chart. |
| **Image Decoder** | CNN-based upsampler. Generates future chart image from encoded features. |
| **Perceptual Loss** | VGG-based feature loss. Compares semantic meaning, not raw pixels. |
| **SSIM Loss** | Structural similarity. Penalizes structural differences while tolerating minor pixel shifts. |
| **OHLCV Extractor** | Computer vision pipeline. Detects candle bodies, wicks, and volume bars from generated images. |

### 6. Training Strategy

* **Loss function:** Perceptual loss (VGG features) + SSIM (structural similarity). No pixel-level MSE.
* **Why perceptual loss:** A candle shifted 1 pixel is semantically identical but has huge pixel error. Perceptual loss captures meaning, not position.
* **Optional GAN:** Adversarial loss can be added later to sharpen outputs.
* **No tokenization, no regression heads, no numerical targets.**

### 7. Success Metrics

| Metric | Target | Description |
| --- | --- | --- |
| **SSIM** | > 0.7 | Structural similarity between predicted and actual charts |
| **Visual Realism** | Pass human evaluation | Generated charts look like real continuations |
| **Direction Accuracy** | > 55% | Predicted trend direction matches actual (extracted via OHLCV pipeline) |
| **MAPE** | < 5% | Mean Absolute Percentage Error of extracted OHLCV vs actual |
| **Extraction Reliability** | > 90% | OHLCV extraction succeeds on generated images |

### 8. Constraints

* **No hidden data:** Zero technical indicators. Model derives all intent from pure price and volume geometry.
* **Zoom-to-fit normalization:** Model learns relative patterns, not absolute prices.
* **System resources:** Follow OSVariables.md guidelines (8GB RAM laptop, no GPU).
* **Chart rendering consistency:** Input and target charts must use identical rendering style.

### 9. Out of Scope (For Now)

* Real-time prediction / live trading
* Multi-timeframe analysis
* Intraday data (using daily only)
* Technical indicator overlays
* Model serving / API deployment