### **Phase 1: Data Engineering & Tokenization**

* [ ] **Dataset Acquisition**: Download historical daily or intraday OHLCV data (S&P 500, NASDAQ) in CSV format.
* [ ] **Sliding Window Generator**: Script a windowing function to create 100-period sequences (Input) paired with the subsequent 10-period sequences (Target).
* [ ] **Candle Tokenizer**:
* [ ] Define the "Candle Code" vocabulary (mapping body size, wick length, direction, and volume to unique integers).
* [ ] Create a script to convert raw OHLCV values into these categorical tokens.


* [ ] **Image Engine**:
* [ ] Develop a Python script to render clean, monochrome  candlestick charts.
* [ ] Implement the "Zoom-to-Fit" logic (min-max normalization per window).
* [ ] Implement the "Stacking Trick" (3-channel grayscale) for model compatibility.


* [ ] **Final Pipeline**: Save images and their corresponding token labels into a structured format (e.g., HDF5 or a directory of PNGs with a mapping JSON).

### **Phase 2: Model Development & Training**

* [ ] **Encoder Setup**: Initialize a pre-trained Vision Transformer (ViT-Base) and verify it accepts the 3-channel stacked grayscale input.
* [ ] **Decoder Architecture**:
* [ ] Design the autoregressive Transformer decoder.
* [ ] Implement the Cross-Attention layer between the ViT Encoder output and the Decoder.


* [ ] **Training Loop**:
* [ ] Set up a Cross-Entropy loss function for the token prediction.
* [ ] Implement "Teacher Forcing" for more efficient training of the sequence decoder.
* [ ] Log training and validation loss to monitor for over-fitting.


* [ ] **Evaluation**: Create a validation script to measure "Token Accuracy" on unseen stock symbols.

### **Phase 3: Inference & Visual Generation**

* [ ] **Autoregressive Loop**: Build the inference function where the model predicts token , renders it into the image, and uses that new image to predict .
* [ ] **Rendering Pipeline**: Build a utility to convert the model's predicted tokens back into a visual candlestick chart for human review.
* [ ] **Attention Mapping**: Integrate a visualization tool to see which patches of the 100-period input chart influenced the 5-10 candle forecast.
* [ ] **Backtesting**: Compare the generated "future" charts against the actual historical outcomes to assess real-world predictive utility.