# PRD: Project "Vision-Trader"

## Generative Vision Transformer for Stock Pattern Forecasting

### 1. Product Vision

To develop a generative AI system that interprets stock market price action visually (as a human trader would) and predicts future market behavior by generating a sequence of future "candle tokens."

### 2. Core Requirements & Logic

* **Visual Input:** 100-period monochrome candlestick charts with volume bars at the bottom.
* **No "Hidden" Data:** Zero technical indicators (RSI, MACD, etc.). The model must derive all intent from pure price and volume geometry.
* **Normalization:** All snapshots are "Zoom-to-Fit," ensuring the model learns relative patterns and volatility regimes rather than absolute price values.
* **Tokenized Output:** Instead of pixels, the model outputs a categorical "code" representing the OHLCV characteristics of the next candle.

### 3. Technical Architecture

| Component | Specification |
| --- | --- |
| **Data Engine** | Python-based generator using `mplfinance` or custom `PIL` scripts to create denoised, grayscale snapshots. |
| **Model Type** | **Encoder-Decoder Vision Transformer (ViT).** |
| **Encoder** | Pre-trained ViT (e.g., ViT-Base) using the "Stacking Trick" (3-channel grayscale) to process the 100-period context. |
| **Decoder** | Autoregressive Transformer decoder that predicts one candle token at a time. |
| **Vocabulary** | A "Candle Code" dictionary (e.g., Code 102 = "Large Bullish Body, Small Wick, High Volume"). |

### 4. Data Specification

* **Input Dimensions:**  pixels (standard for pre-trained ViTs).
* **Sequence Length:** 100 input candles  5–10 predicted output candles.
* **Preprocessing:** * Remove all UI/Grid elements.
* Scale OHLC to  relative to the 100-period window.
* Generate synthetic training data via sliding windows across S&P 500 / NASDAQ historical CSVs.



### 5. Success Metrics (The "Checklist")

* **Token Accuracy:** Percent of correctly predicted candle "types" (Direction and Size).
* **Visual Consistency:** When the tokens are rendered back into a chart, do the predicted candles form a logical continuation of the trend?
* **Edge Case Resilience:** Ability to recognize "reversal" patterns (e.g., Dojis at peaks) vs. "continuation" patterns.

---

### 6. Implementation Roadmap

1. **Phase 1 (Data):** Script to convert historical CSV data into the 3-channel "Stacked" grayscale snapshots and generate the Token Vocabulary.
2. **Phase 2 (Training):** Fine-tune the ViT Encoder and train the Autoregressive Decoder.
3. **Phase 3 (Inference):** Build a loop where the model predicts a token, we render it, and feed the new image back in for the next step.